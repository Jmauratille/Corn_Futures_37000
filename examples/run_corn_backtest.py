"""Synthetic end-to-end demo; prices and contract settings are illustrative."""

import pandas as pd

from finm37000.backtest.bollinger import BollingerMeanReversion
from finm37000.backtest.engine import run_backtest
from finm37000.backtest.models import BacktestConfig


def main() -> None:
    """Run a synthetic Bollinger backtest and print fills, history and P&L."""
    closes = [500., 501., 499., 498., 501., 503., 500., 497.]
    opens = [500.] + closes[:-1]
    bars = pd.DataFrame({
        "symbol": "TEST", "open": opens, "close": closes,
        "high": [max(o, c) for o, c in zip(opens, closes, strict=True)],
        "low": [min(o, c) for o, c in zip(opens, closes, strict=True)],
    }, index=pd.date_range("2026-10-01 14:05", periods=len(closes), freq="5min",
                           tz="UTC"))
    # Synthetic settings; replace with the units and costs of your real contract.
    config = BacktestConfig(symbol="TEST", initial_capital=10000.,
                            dollars_per_price_unit=10., tick_size=0.25,
                            fee_per_contract=1., slippage_ticks=1, max_position=1)
    result = run_backtest(bars, BollingerMeanReversion(window=3), config,
                          bar_duration="5min")
    print("Fills:")
    print(result.trades.to_string(index=False))
    print("\nPortfolio at each bar close:")
    print(result.history[["close", "position", "target_position", "net_pnl", "equity"]])
    print(f"\nFinal marked P&L: ${result.history.net_pnl.iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
