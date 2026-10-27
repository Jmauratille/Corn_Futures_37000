"""Middle-line mean reversion; run with python -m finm37000.backtest.bollinger."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finm37000 import bollinger_bands
from finm37000.backtest.models import PortfolioState

MIN_WINDOW = 2
ALLOWED_POSITIONS = (-1, 0, 1)


@dataclass(frozen=True)
class BollingerMeanReversion:
    """Target +1 below the rolling mean, -1 above it, otherwise hold.

    The caller supplies one contract's chronological CLOSED bars only.
    The current closed bar is included in the rolling window. No training
    labels or future bars should be passed to this strategy.
    """

    window: int = 20

    def __post_init__(self) -> None:
        """Validate that ``window`` is an integer of at least two."""
        if (isinstance(self.window, bool) or not isinstance(self.window, int)
                or self.window < MIN_WINDOW):
            msg = "window must be an integer of at least 2."
            raise ValueError(msg)

    def on_bar(self, history: pd.DataFrame, portfolio: PortfolioState) -> int:
        """Return a target position, never an incremental order quantity."""
        if portfolio.position not in ALLOWED_POSITIONS:
            msg = "Expected a portfolio position of -1, 0, or 1."
            raise ValueError(msg)
        if "close" not in history:
            msg = "history must contain close."
            raise KeyError(msg)
        if not isinstance(history.index, pd.DatetimeIndex):
            msg = "history must have a DatetimeIndex."
            raise ValueError(msg)
        if (history.index.hasnans or not history.index.is_unique
                or not history.index.is_monotonic_increasing):
            msg = "history timestamps must be unique, increasing, and nonmissing."
            raise ValueError(msg)
        if history.empty:
            return portfolio.position
        if "symbol" in history and (history["symbol"].isna().any()
                                     or len(history["symbol"].unique()) != 1):
            msg = "history must contain exactly one nonmissing contract symbol."
            raise ValueError(msg)
        close = history["close"].iloc[-self.window:]
        if (not pd.api.types.is_numeric_dtype(close)
                or pd.api.types.is_bool_dtype(close)
                or pd.api.types.is_complex_dtype(close)):
            msg = "close must contain real numeric prices."
            raise ValueError(msg)
        # Missing/bad prices in the active window mean no new signal.
        if (len(close) < self.window
                or not np.isfinite(close.to_numpy(dtype=float)).all()):
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
