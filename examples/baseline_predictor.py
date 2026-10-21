"""Evaluate the baseline return predictor on a chronological holdout.

Run from the repository root. With stored trades (see pull_data)::

    python examples/baseline_predictor.py --symbol ZCH7 --bar 5min --horizon 3 \
        --end 2026-10-01T00:00:00Z

``--end`` is the exclusive end of the stored data's coverage. Bars are built
with :func:`finm37000.build_analysis_bars`, so only completed bars are used.

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

from finm37000 import build_analysis_bars, load_history, run_baseline

TICK = 0.25  # ZC minimum price increment, in cents per bushel.


def synthetic_bars(n: int = 2000, seed: int = 37000) -> pd.DataFrame:
    """Random-walk bars rounded to the corn tick, so zero returns occur."""
    rng = np.random.default_rng(seed)
    close = 450 + TICK * np.cumsum(rng.choice([-1, 0, 0, 1], n))
    return pd.DataFrame(
        {"symbol": "SYNTH", "close": close, "volume": rng.integers(1, 200, n)},
        index=pd.date_range("2026-01-05 08:35", periods=n, freq="5min", tz="UTC"),
    )


def stored_bars(symbol: str, rule: str, end: pd.Timestamp) -> pd.DataFrame:
    """Completed, end-labeled event-time bars of one contract's stored trades."""
    trades = load_history(symbol)
    trades = trades.loc[trades["symbol"] == symbol]
    return build_analysis_bars(trades, rule, end=end)


def main() -> None:
    """Parse arguments, run the baseline and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="ZCH7")
    parser.add_argument("--bar", default="5min", help="Bar duration, e.g. 5min.")
    parser.add_argument(
        "--end", help="Exclusive, timezone-aware end of the stored data coverage."
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, help="Also write the report here.")
    args = parser.parse_args()

    if args.synthetic:
        bars = synthetic_bars()
    elif args.end is None:
        parser.error("--end is required with stored data; use the coverage end.")
    else:
        bars = stored_bars(args.symbol, args.bar, pd.Timestamp(args.end))
    report = run_baseline(
        bars,
        window=args.window,
        horizon=args.horizon,
        test_fraction=args.test_fraction,
        gap=args.gap,
        ridge=args.ridge,
    )
    report["settings"]["bar"] = "5min (synthetic)" if args.synthetic else args.bar
    report["settings"]["end"] = None if args.synthetic else args.end
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
