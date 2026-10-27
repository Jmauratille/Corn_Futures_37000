"""Get one target position from locally stored corn trades; no orders are sent."""

import argparse

import pandas as pd

from finm37000.backtest.bollinger import BollingerMeanReversion
from finm37000.backtest.models import PortfolioState
from finm37000.bars import build_analysis_bars
from finm37000.data_store import load_history


def main() -> None:
    """Print the latest Bollinger target position from stored corn trades."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ZCH7")
    parser.add_argument("--end", required=True,
                        help="Actual exclusive data coverage endpoint, with timezone.")
    parser.add_argument("--rule", default="5min")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--position", type=int, choices=(-1, 0, 1), default=0)
    args = parser.parse_args()
    trades = load_history(args.symbol)
    # Databento commonly indexes by receive time; bars require event time.
    # Preserve ts_event if present. Never relabel receive timestamps as events.
    bars = build_analysis_bars(trades, rule=args.rule, end=pd.Timestamp(args.end))
    if bars.empty:
        msg = "No completed bars before --end."
        raise ValueError(msg)
    strategy = BollingerMeanReversion(window=args.window)
    portfolio = PortfolioState(position=args.position)
    target = strategy.on_bar(bars, portfolio)
    middle = bars["close"].rolling(args.window).mean().iloc[-1]
    print(f"Latest completed bar: {bars.index[-1]}")
    print(f"Observed bars: {len(bars)}; close: {bars['close'].iloc[-1]}; "
          f"middle: {middle}")
    print(f"Current position: {portfolio.position}; target: {target}")
    print(f"Order quantity for subsequent execution: {target - portfolio.position}")


if __name__ == "__main__":
    main()
