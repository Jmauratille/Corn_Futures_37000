"""Build causal bar features and purged chronological research datasets."""

import numpy as np
import pandas as pd

from .features import (
    bollinger_bands,
    realized_volatility,
    returns,
    simple_moving_average,
)

MIN_WINDOW = 2

FEATURE_COLUMNS = ("return_1", "sma_distance", "bandwidth", "volatility", "volume")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer."
        raise ValueError(msg)


def _check_index(frame: pd.DataFrame) -> None:
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.hasnans
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        msg = "Expected a unique, increasing DatetimeIndex without missing timestamps."
        raise ValueError(msg)


def build_prediction_dataset(
    bars: pd.DataFrame,
    window: int = 20,
    horizon: int = 1,
) -> pd.DataFrame:
    """Combine end-of-bar features with a future simple-return target.

    Args:
        bars: One contract's bars, indexed by bar END time, with numeric
            ``close`` and ``volume``. Must be sorted with unique timestamps.
            A symbol column, if present, must contain one nonmissing symbol.
        window: Observed bars in the rolling feature windows (at least two).
        horizon: Number of observed bars ahead for the target, not elapsed time.

    Returns:
        Complete feature rows plus ``target_return = close[t+h]/close[t]-1``
        and ``target_end`` (the timestamp of that future close). Warm-up and
        unknown-target rows are dropped. Inputs are never modified.

    Raises:
        ValueError: For invalid timestamps, prices, volumes, symbols or parameters.
        KeyError: If required columns are missing.

    Notes:
        Features use data available through the bar close. Executing at that
        same close is not guaranteed: later strategy work must model execution
        and costs. Never include target_return or target_end as model features.
        Gaps between observations are preserved; no prices are forward-filled.
    """
    _positive_integer("window", window)
    _positive_integer("horizon", horizon)
    if window < MIN_WINDOW:
        msg = "window must be at least two for sample volatility."
        raise ValueError(msg)
    _check_index(bars)
    missing = {"close", "volume"} - set(bars.columns)
    if missing:
        msg = f"Missing bar columns: {sorted(missing)}."
        raise KeyError(msg)
    if "symbol" in bars and (
        bars["symbol"].isna().any() or len(bars["symbol"].unique()) != 1
    ):
        msg = "bars must contain exactly one nonmissing symbol."
        raise ValueError(msg)
    values = bars[["close", "volume"]]
    if (
        not all(pd.api.types.is_numeric_dtype(values[c]) for c in values)
        or values.isna().any().any()
        or not np.isfinite(values.to_numpy(dtype=float)).all()
        or (bars["close"] <= 0).any()
        or (bars["volume"] < 0).any()
    ):
        msg = "close must be finite and positive; volume finite and nonnegative."
        raise ValueError(msg)
    close = bars["close"]
    bands = bollinger_bands(close, window=window)
    result = pd.DataFrame(
        {
            "return_1": returns(close, log=False),
            "sma_distance": close / simple_moving_average(close, window) - 1,
            "bandwidth": bands["bandwidth"],
            "volatility": realized_volatility(close, window),
            "volume": bars["volume"],
            "target_return": close.shift(-horizon) / close - 1,
            "target_end": bars.index.to_series().shift(-horizon),
        },
        index=bars.index,
    )
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def chronological_split(
    dataset: pd.DataFrame,
    test_fraction: float = 0.2,
    gap: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest rows for testing and purge overlapping training labels.

    Args:
        dataset: Output of build_prediction_dataset, including target_end.
        test_fraction: Fraction reserved for testing, rounded up to whole rows.
        gap: Additional dataset rows excluded immediately before the test start.

    Returns:
        Independent train and test copies. Every retained training target ends
        strictly before the first test timestamp. Features may use historical
        bars before the split; fit scalers and models only on training rows.

    Raises:
        ValueError: For invalid fractions, gaps, timestamps or empty partitions.
        KeyError: If target_end is missing.
    """
    _check_index(dataset)
    if not 0 < test_fraction < 1:
        msg = "test_fraction must be strictly between zero and one."
        raise ValueError(msg)
    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
        msg = "gap must be a nonnegative integer."
        raise ValueError(msg)
    if "target_end" not in dataset:
        msg = "dataset must contain target_end."
        raise KeyError(msg)
    index = pd.DatetimeIndex(dataset.index)
    target_end = dataset["target_end"]
    if (
        not pd.api.types.is_datetime64_any_dtype(target_end)
        or target_end.isna().any()
        or target_end.dt.tz != index.tz
        or not (target_end > dataset.index).all()
    ):
        msg = "target_end must be a later timestamp in the index timezone."
        raise ValueError(msg)
    test_count = int(np.ceil(len(dataset) * test_fraction))
    split = len(dataset) - test_count
    if test_count == 0 or split <= gap:
        msg = "Not enough rows for nonempty train and test partitions."
        raise ValueError(msg)
    test = dataset.iloc[split:].copy()
    candidates = dataset.iloc[: split - gap]
    train = candidates.loc[candidates["target_end"] < test.index[0]].copy()
    if train.empty:
        msg = "No training rows remain after purging overlapping targets."
        raise ValueError(msg)
    return train, test
