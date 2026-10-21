"""Build completed analysis bars from synthetic trades or an existing local file."""

import argparse

import numpy as np
import pandas as pd

from finm37000 import (
    build_analysis_bars,
    build_prediction_dataset,
    chronological_split,
    load_history,
)


def main() -> None:
    """Run a free synthetic demo, or load cached trades without downloading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="Load this symbol's existing cached trades.")
    parser.add_argument(
        "--end", help="Exclusive timezone-aware data coverage endpoint."
    )
    parser.add_argument("--rule", default="5min", help="Fixed-duration bar frequency.")
    args = parser.parse_args()
    if args.symbol:
        if not args.end:
            parser.error("--end is required with --symbol; use the known coverage end.")
        trades = load_history(args.symbol)
        end = pd.Timestamp(args.end)
    else:
        count = 600
        rng = np.random.default_rng(37000)
        timestamps = pd.date_range(
            "2026-10-01",
            periods=count,
            freq="1min",
            tz="UTC",
            name="ts_event",
        )
        trades = pd.DataFrame(
            {
                "symbol": "SYNTHETIC",
                "price": 450 * np.exp(np.cumsum(rng.normal(0, 0.001, count))),
                "size": rng.integers(1, 10, count),
                "side": rng.choice(["B", "A", "N"], count),
            },
            index=timestamps,
        )
        end = (
            pd.Timestamp(args.end)
            if args.end
            else timestamps[-1] + pd.Timedelta("1min")
        )
    bars = build_analysis_bars(trades, args.rule, end=end)
    print(f"Source: {args.symbol or 'synthetic (not market performance)'}")
    print(f"Completed bars: {len(bars)}; cutoff: {end}")
    print(
        f"Volume: {bars['volume'].sum()}; unknown-side: {bars['unknown_volume'].sum()}"
    )
    print(bars.head().to_string())
    dataset = build_prediction_dataset(bars, window=5, horizon=2)
    train, test = chronological_split(dataset)
    print(f"Dataset: {len(dataset)}; train: {len(train)}; test: {len(test)}")
    print(f"Last training target: {train['target_end'].max()}")
    print(f"First test observation: {test.index[0]}")


if __name__ == "__main__":
    main()
