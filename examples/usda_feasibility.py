"""Run from the repository root: python examples/usda_feasibility.py.

Demonstrates the USDA surprise -> ZC price-reaction join on synthetic data.
The numbers below are invented to exercise the pipeline; see
docs/usda_feasibility.md for the real data-source assessment. No USDA or
Databento access is used.
"""

import numpy as np
import pandas as pd

from finm37000 import (
    SurpriseObservation,
    build_surprise_sample,
    join_reactions,
)

TZ = "America/Chicago"

# Two synthetic releases with an expectation known the day before each.
observations = [
    SurpriseObservation(
        report="crop_production",
        expected_time=pd.Timestamp("2026-08-11 11:00", tz=TZ),
        release_time=pd.Timestamp("2026-08-12 11:00", tz=TZ),
        expected=15000.0,
        actual=15250.0,  # bullish-for-supply surprise (more corn than expected)
        unit="million bushels",
    ),
    SurpriseObservation(
        report="grain_stocks",
        expected_time=pd.Timestamp("2026-09-29 11:00", tz=TZ),
        release_time=pd.Timestamp("2026-09-30 11:00", tz=TZ),
        expected=1800.0,
        actual=1740.0,  # tighter stocks than expected
        unit="million bushels",
    ),
]
sample = build_surprise_sample(observations)

# Synthetic 30-minute ZC closes (cents/bushel) spanning both releases.
index = pd.date_range("2026-08-12 09:00", "2026-09-30 13:00", freq="30min", tz=TZ)
rng = np.random.default_rng(37000)
prices = pd.Series(
    450 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(index)))), index=index
)

joined = join_reactions(sample, prices, pd.Timedelta(hours=2))
print("Validated joined feasibility sample (synthetic):")
print(joined[["report", "expected", "actual", "surprise", "pct_surprise", "reaction"]])
print("\nColumns a real study would need:")
print(" - actual (USDA, at release)  : NASS QuickStats API (free key) -- feasible")
print(" - precise release_time       : NASS release calendar          -- feasible")
print(" - expected (pre-report)      : trade-press surveys, paywalled  -- the gap")
print("\nSee docs/usda_feasibility.md for the recommendation.")
