"""Run from the repository root: python examples/prediction_dataset.py."""

import numpy as np
import pandas as pd

from finm37000 import (
    FEATURE_COLUMNS,
    build_prediction_dataset,
    chronological_split,
)

# Synthetic bars demonstrate the pipeline without downloading market data.
rng = np.random.default_rng(37000)
bars = pd.DataFrame(
    {
        "close": 450 * np.exp(np.cumsum(rng.normal(0, 0.001, 240))),
        "volume": rng.integers(1, 100, 240),
    },
    index=pd.date_range("2026-01-05 15:00", periods=240, freq="5min", tz="UTC"),
)
dataset = build_prediction_dataset(bars, window=20, horizon=3)
train, test = chronological_split(dataset, test_fraction=0.2)
print(f"Complete rows: {len(dataset)}; train: {len(train)}; test: {len(test)}")
print(f"Purged rows: {len(dataset) - len(train) - len(test)}")
print(f"Last training target: {train['target_end'].max()}")
print(f"First test observation: {test.index[0]}")
print(train[list(FEATURE_COLUMNS)].head())
# Fit a model/scaler only on train[FEATURE_COLUMNS], with train.target_return.
# Evaluate once on the held-out test data; these are synthetic, not real results.
