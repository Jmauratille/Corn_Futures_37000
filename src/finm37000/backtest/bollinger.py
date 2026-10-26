"""Middle-line mean reversion; run with python -m finm37000.backtest.bollinger."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finm37000 import bollinger_bands
from finm37000.backtest.models import PortfolioState


@dataclass(frozen=True)
class BollingerMeanReversion:
    """Target +1 below the rolling mean, -1 above it, otherwise hold.

    The caller supplies one contract's chronological CLOSED bars only.
    The current closed bar is included in the rolling window. No training
    labels or future bars should be passed to this strategy.
    """

    window: int = 20

    def __post_init__(self) -> None:
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            raise ValueError("window must be an integer of at least 2.")

    def on_bar(self, history: pd.DataFrame, portfolio: PortfolioState) -> int:
        """Return a target position, never an incremental order quantity."""
        if portfolio.position not in (-1, 0, 1):
            raise ValueError("Expected a portfolio position of -1, 0, or 1.")
        if "close" not in history:
            raise KeyError("history must contain close.")
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("history must have a DatetimeIndex.")
        if (history.index.hasnans or not history.index.is_unique
                or not history.index.is_monotonic_increasing):
            raise ValueError("history timestamps must be unique, increasing, and nonmissing.")
        if history.empty:
            return portfolio.position
        if "symbol" in history and (history["symbol"].isna().any()
                                     or history["symbol"].nunique() != 1):
            raise ValueError("history must contain exactly one nonmissing contract symbol.")
        close = history["close"].iloc[-self.window:]
        if (not pd.api.types.is_numeric_dtype(close)
                or pd.api.types.is_bool_dtype(close)
                or pd.api.types.is_complex_dtype(close)):
            raise ValueError("close must contain real numeric prices.")
        # Missing/bad prices in the active window mean no new signal.
        if len(close) < self.window or not np.isfinite(close.to_numpy(dtype=float)).all():
            return portfolio.position
        middle = bollinger_bands(close, window=self.window)["middle"].iloc[-1]
        latest = close.iloc[-1]
        if latest < middle:
            return 1
        if latest > middle:
            return -1
        return portfolio.position


def main() -> None:
    """Synthetic signal check; does not simulate fills or calculate P&L."""
    strategy = BollingerMeanReversion(window=3)
    rows = []
    for scenario, closes in {
        "warmup": [500.0, 500.0],
        "below middle": [500.0, 500.0, 499.0],
        "above middle": [500.0, 500.0, 501.0],
        "equal to middle": [500.0, 500.0, 500.0],
    }.items():
        history = pd.DataFrame({"close": closes}, index=pd.date_range(
            "2026-10-01 14:05", periods=len(closes), freq="5min", tz="UTC"))
        for position in (-1, 0, 1):
            target = strategy.on_bar(history, PortfolioState(position=position))
            rows.append({"scenario": scenario, "position": position,
                         "target": target, "order_quantity": target - position})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
