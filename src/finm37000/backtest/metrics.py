"""Gross vs net performance and turnover over a period of a backtest."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import BacktestResult

TRADING_DAYS_PER_YEAR = 252
SESSION_TZ = "America/Chicago"
SESSION_ROLL = pd.Timedelta(hours=7)


@dataclass(frozen=True)
class PeriodSummary:
    """Dollar performance of one bar-end period, split into gross and costs.

    ``gross_pnl`` is before slippage and fees, ``net_pnl`` after both, so
    ``net_pnl == gross_pnl - slippage_cost - fees``. Turnover counts every
    contract bought or sold; a reversal from -1 to +1 counts as two.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    contracts_traded: int
    fills: int
    gross_pnl: float
    slippage_cost: float
    fees: float
    net_pnl: float
    net_max_drawdown: float
    time_in_market: float
    daily_net_sharpe: float

    @property
    def gross_pnl_per_contract(self) -> float:
        """Gross P&L earned per contract traded; NaN if nothing traded."""
        if not self.contracts_traded:
            return float("nan")
        return self.gross_pnl / self.contracts_traded

    @property
    def cost_per_contract(self) -> float:
        """Slippage plus fees per contract traded; NaN if nothing traded."""
        if not self.contracts_traded:
            return float("nan")
        return (self.slippage_cost + self.fees) / self.contracts_traded


def summarize_period(
    result: BacktestResult,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> PeriodSummary:
    """Summarize bar ends in ``[start, end)`` of a full-sample backtest.

    P&L is the change in marked cumulative values from the last bar before
    ``start``, so a position carried into the period is charged or credited
    only with its in-period price moves. Fills belong to the bar whose open
    executed them. Bounds default to the whole history.
    """
    history = result.history
    if history.empty:
        msg = "Backtest history is empty."
        raise ValueError(msg)
    index = pd.DatetimeIndex(history.index)
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index < end
    if not mask.any():
        msg = "No bars in the requested period."
        raise ValueError(msg)
    config = result.config
    # A fill changes the position at that bar, so |change| is its quantity.
    traded = history["position"].diff().fillna(history["position"]).abs()
    slippage = (traded * config.slippage_ticks * config.tick_size
                * config.dollars_per_price_unit)
    fees = history["cumulative_fees"].diff().fillna(history["cumulative_fees"])
    net = history["net_pnl"]
    first = int(np.argmax(mask))
    before = float(net.iloc[first - 1]) if first else 0.0
    period_net = net[mask] - before
    net_pnl = float(period_net.iloc[-1])
    slippage_cost = float(slippage[mask].sum())
    period_fees = float(fees[mask].sum())
    drawdown = period_net.cummax().clip(lower=0.0) - period_net
    # CME sessions open at 17:00 Chicago time and count toward the next day.
    session = (index[mask].tz_convert(SESSION_TZ) + SESSION_ROLL).normalize()
    daily = period_net.groupby(session).last().diff(1)
    daily.iloc[0] = period_net.groupby(session).last().iloc[0]
    std = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    sharpe = (float(daily.mean()) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
              if std > 0 else float("nan"))
    return PeriodSummary(
        start=index[mask][0], end=index[mask][-1], bars=int(mask.sum()),
        contracts_traded=int(traded[mask].sum()),
        fills=int((traded[mask] > 0).sum()),
        gross_pnl=net_pnl + slippage_cost + period_fees,
        slippage_cost=slippage_cost, fees=period_fees, net_pnl=net_pnl,
        net_max_drawdown=float(drawdown.max()),
        time_in_market=float((history["position"][mask] != 0).mean()),
        daily_net_sharpe=sharpe,
    )
