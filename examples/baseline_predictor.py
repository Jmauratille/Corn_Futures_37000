"""Evaluate the baseline return predictor on a chronological holdout.

Run from the repository root. With stored trades (see pull_data)::

    python examples/baseline_predictor.py --symbol ZCH7 --bar 5min --horizon 3

Without a Databento key, ``--synthetic`` runs on random-walk bars, which
should show no edge over the zero-return benchmark::

    python examples/baseline_predictor.py --synthetic

Settings are fixed on the command line before the run; do not rerun with
different settings to improve holdout numbers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finm37000 import load_history, make_ohlcv, run_baseline

TICK = 0.25  # ZC minimum price increment, in cents per bushel.


def synthetic_bars(n: int = 2000, seed: int = 37000) -> pd.DataFrame:
    """Random-walk bars rounded to the corn tick, so zero returns occur."""
    rng = np.random.default_rng(seed)
    close = 450 + TICK * np.cumsum(rng.choice([-1, 0, 0, 1], n))
    return pd.DataFrame(
        {"symbol": "SYNTH", "close": close, "volume": rng.integers(1, 200, n)},
        index=pd.date_range("2026-01-05 08:35", periods=n, freq="5min", tz="UTC"),
    )


def stored_bars(symbol: str, rule: str) -> pd.DataFrame:
    """End-labeled bars of one contract's stored trades, on event time."""
    trades = load_history(symbol)
    trades = trades.loc[trades["symbol"] == symbol]
    trades.index = pd.DatetimeIndex(trades["ts_event"])
    trades.index.name = "ts_event"
    return make_ohlcv(
        trades.drop(columns="ts_event"), rule, index_name="ts_event", label="right"
    )


def main() -> None:
    """Parse arguments, run the baseline and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="ZCH7")
    parser.add_argument("--bar", default="5min", help="Bar duration, e.g. 5min.")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, help="Also write the report here.")
    args = parser.parse_args()

    bars = synthetic_bars() if args.synthetic else stored_bars(args.symbol, args.bar)
    report = run_baseline(
        bars,
        window=args.window,
        horizon=args.horizon,
        test_fraction=args.test_fraction,
        gap=args.gap,
        ridge=args.ridge,
    )
    report["settings"]["bar"] = "5min (synthetic)" if args.synthetic else args.bar
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
