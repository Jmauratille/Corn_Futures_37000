"""Single-contract, bar-based backtesting with next-observed-open fills."""

from dataclasses import asdict, replace
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from .execution import NextOpenExecution
from .models import BacktestConfig, BacktestResult, Fill, Order, PortfolioState
from .strategy import Strategy


def validate_bars(bars: pd.DataFrame) -> None:
    """Require nonempty, chronological, timezone-aware bar-end OHLC data."""
    if bars.empty:
        msg = "bars must not be empty."
        raise ValueError(msg)
    if not isinstance(bars.index, pd.DatetimeIndex):
        msg = "bars must have a DatetimeIndex of bar-end timestamps."
        raise ValueError(msg)
    if (bars.index.tz is None or bars.index.hasnans
            or not bars.index.is_unique or not bars.index.is_monotonic_increasing):
        msg = "Bar ends must be timezone-aware, unique, and increasing."
        raise ValueError(msg)
    columns = ["open", "high", "low", "close"]
    missing = set(columns) - set(bars.columns)
    if missing:
        msg = f"Missing OHLC columns: {sorted(missing)}"
        raise ValueError(msg)
    for column in columns:
        values = bars[column]
        if (not pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)
                or not np.isfinite(values.to_numpy(dtype=float)).all()):
            msg = f"{column} must contain finite real prices."
            raise ValueError(msg)
    if ((bars["low"] > bars[["open", "close"]].min(axis=1)).any()
            or (bars["high"] < bars[["open", "close"]].max(axis=1)).any()
            or (bars["low"] > bars["high"]).any()):
        msg = "OHLC prices are inconsistent."
        raise ValueError(msg)
    if "symbol" in bars and (bars["symbol"].isna().any()
                              or len(bars["symbol"].unique()) != 1):
        msg = "bars must contain exactly one nonmissing symbol."
        raise ValueError(msg)


def _validate_config(config: BacktestConfig) -> None:
    for name in ("initial_capital", "dollars_per_price_unit", "tick_size"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            msg = f"{name} must be finite and positive."
            raise ValueError(msg)
    if not np.isfinite(config.fee_per_contract) or config.fee_per_contract < 0:
        msg = "fee_per_contract must be finite and nonnegative."
        raise ValueError(msg)
    for name, minimum in (("slippage_ticks", 0), ("max_position", 1)):
        value = getattr(config, name)
        if (isinstance(value, bool) or not isinstance(value, Integral)
                or value < minimum):
            msg = f"{name} must be an integer >= {minimum}."
            raise ValueError(msg)


def apply_fill(
    portfolio: PortfolioState, fill: Fill, config: BacktestConfig,
) -> PortfolioState:
    """Apply one fill; realized P&L is gross of fees, in dollars.

    Same-side additions use weighted average entry. Opposite-side fills
    realize P&L on the closed quantity. Reversals open the residual at the
    fill price. Unrealized P&L is recomputed separately at the bar close.
    """
    old = portfolio.position
    quantity = fill.quantity
    new = old + quantity
    if quantity == 0:
        msg = "A fill must have nonzero quantity."
        raise ValueError(msg)
    if abs(new) > config.max_position:
        msg = "Fill exceeds max_position."
        raise ValueError(msg)
    realized = portfolio.realized_pnl
    entry = portfolio.average_entry_price
    if old == 0 or old * quantity > 0:
        entry = (abs(old) * entry + abs(quantity) * fill.price) / abs(new)
    else:
        closed = min(abs(old), abs(quantity))
        direction = 1 if old > 0 else -1
        realized += (closed * direction * (fill.price - entry)
                     * config.dollars_per_price_unit)
        if new == 0:
            entry = 0.0
        elif old * new < 0:
            entry = fill.price
    return replace(portfolio, position=new, average_entry_price=entry,
                   realized_pnl=realized, unrealized_pnl=0.0,
                   cumulative_fees=portfolio.cumulative_fees + fill.fees)


def mark_to_market(
    portfolio: PortfolioState, close_price: float, config: BacktestConfig,
) -> PortfolioState:
    """Value the open futures exposure without deducting contract notional."""
    unrealized = (portfolio.position * (close_price - portfolio.average_entry_price)
                  * config.dollars_per_price_unit)
    return replace(portfolio, unrealized_pnl=unrealized)


def make_history_record(
    timestamp: pd.Timestamp, portfolio: PortfolioState, config: BacktestConfig,
) -> dict:
    """Build one history row with net P&L and equity for the bar close."""
    record = {"timestamp": timestamp, **asdict(portfolio)}
    net_pnl = (portfolio.realized_pnl + portfolio.unrealized_pnl
               - portfolio.cumulative_fees)
    record.update(net_pnl=net_pnl, equity=config.initial_capital + net_pnl)
    return record


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    *,
    bar_duration: str,
) -> BacktestResult:
    """Process closed bars and fill signals at the next observed bar's open.

    bar_duration MUST match the frequency used to build the input bars.
    The input index labels bar END; modeled execution time is next bar end
    minus duration. This interval-start timestamp is an approximation: OHLC
    data do not identify when the first trade actually occurred or guarantee
    that an order could execute there. Empty intervals remain omitted.

    Strategy history ends at the current close. The final signal is recorded
    but cannot fill without a later bar. Final exposure stays marked at the
    last close; no automatic liquidation, margin calls, funding, or exchange
    variation-margin cash movements are modeled. No data are downloaded.
    """
    validate_bars(bars)
    _validate_config(config)
    offset = to_offset(bar_duration)
    if not isinstance(offset, Tick) or offset.nanos <= 0:
        msg = "bar_duration must be a positive fixed duration."
        raise ValueError(msg)
    duration = pd.Timedelta(offset.nanos, unit="ns")
    # validate_bars guarantees a DatetimeIndex; restate it for the type checker.
    bar_ends = pd.DatetimeIndex(bars.index)
    starts = bar_ends - duration
    if len(bars) > 1 and (starts[1:] < bar_ends[:-1]).any():
        msg = "Bars overlap: check bar_duration against bar-end timestamps."
        raise ValueError(msg)
    if "symbol" in bars and bars["symbol"].iloc[0] != config.symbol:
        msg = "bars symbol does not match config.symbol."
        raise ValueError(msg)

    portfolio = PortfolioState()
    execution = NextOpenExecution(config)
    pending: Order | None = None
    records: list[dict] = []
    fills: list[dict] = []
    for i, (_, bar) in enumerate(bars.iterrows()):
        timestamp = bar_ends[i]
        # This order was decided at the PREVIOUS observed bar's close.
        if pending is not None:
            fill = execution.fill(pending, starts[i], float(bar["open"]))
            portfolio = apply_fill(portfolio, fill, config)
            fills.append(asdict(fill))
            pending = None

        portfolio = mark_to_market(portfolio, float(bar["close"]), config)
        # A copy prevents an accidental strategy mutation from altering input bars.
        # Typed as object so the defensive runtime check below is not elided:
        # Strategy is a Protocol, and a faulty implementation could return a
        # non-integer despite the declared return type.
        signal: object = strategy.on_bar(bars.iloc[:i + 1].copy(), portfolio)
        if isinstance(signal, bool) or not isinstance(signal, Integral):
            msg = "Strategy must return an integer target position."
            raise ValueError(msg)
        target = int(signal)
        if abs(target) > config.max_position:
            msg = "Strategy target exceeds max_position."
            raise ValueError(msg)
        quantity = target - portfolio.position
        if quantity:
            pending = Order(signal_time=timestamp, quantity=quantity)
        record = make_history_record(timestamp, portfolio, config)
        record.update(close=float(bar["close"]), target_position=target,
                      order_quantity=quantity)
        records.append(record)

    history = pd.DataFrame(records).set_index("timestamp")
    trades = pd.DataFrame(fills, columns=[
        "signal_time", "execution_time", "quantity", "price", "fees",
    ])
    return BacktestResult(config=config, history=history, trades=trades)
