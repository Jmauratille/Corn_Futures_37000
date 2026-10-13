import numpy as np
import pandas as pd
import pytest

from finm37000.dataset import (
    FEATURE_COLUMNS,
    build_prediction_dataset,
    chronological_split,
)


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {"close": np.arange(100.0, 140.0), "volume": np.arange(40)},
        index=pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC"),
    )


def test_target_alignment_and_input_preservation() -> None:
    source = bars()
    before = source.copy(deep=True)
    data = build_prediction_dataset(source, window=3, horizon=2)
    assert data.index[0] == source.index[3]
    assert data.index[-1] == source.index[-3]
    assert list(data.columns) == [*FEATURE_COLUMNS, "target_return", "target_end"]
    assert data.iloc[0]["target_return"] == pytest.approx(105 / 103 - 1)
    assert data.iloc[0]["target_end"] == source.index[5]
    pd.testing.assert_frame_equal(source, before)


def test_features_do_not_depend_on_future_prices() -> None:
    source = bars()
    changed = source.copy()
    cutoff = source.index[20]
    changed.loc[changed.index > cutoff, "close"] *= 3
    original = build_prediction_dataset(source, window=3, horizon=2)
    altered = build_prediction_dataset(changed, window=3, horizon=2)
    pd.testing.assert_frame_equal(
        original.loc[:cutoff, list(FEATURE_COLUMNS)],
        altered.loc[:cutoff, list(FEATURE_COLUMNS)],
    )
    assert original.loc[cutoff, "target_return"] != altered.loc[cutoff, "target_return"]


def test_split_purges_by_target_timestamp_even_with_missing_bars() -> None:
    source = bars().drop(bars().index[[10, 11, 12]])
    data = build_prediction_dataset(source, window=3, horizon=4)
    train, test = chronological_split(data, test_fraction=0.25)
    assert (train["target_end"] < test.index[0]).all()
    assert len(test) == int(np.ceil(len(data) * 0.25))
    assert len(train) + len(test) < len(data)
    assert train.index.intersection(test.index).empty
    pd.testing.assert_frame_equal(test, data.iloc[-len(test) :])


def test_extra_gap_and_split_copies() -> None:
    data = build_prediction_dataset(bars(), window=3)
    train, test = chronological_split(data, gap=5)
    split = len(data) - len(test)
    assert train.index[-1] == data.index[split - 6]
    sentinel = -999
    train.iloc[0, 0] = sentinel
    assert data.iloc[0, 0] != sentinel


@pytest.mark.parametrize(
    "window,horizon", [(0, 1), (1, 1), (2, 0), (True, 1), (2, 1.5)]
)
def test_invalid_parameters(window: int, horizon: int) -> None:
    with pytest.raises(ValueError):
        build_prediction_dataset(bars(), window=window, horizon=horizon)


@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "missing", "not_datetime"])
def test_invalid_index(kind: str) -> None:
    source = bars()
    if kind == "unsorted":
        source = source.iloc[::-1]
    elif kind == "duplicate":
        source.index = pd.DatetimeIndex([source.index[0]] * len(source))
    elif kind == "missing":
        source.index = pd.DatetimeIndex([pd.NaT, *source.index[1:]])
    else:
        source = source.reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_prediction_dataset(source)


@pytest.mark.parametrize(
    "column,value", [("close", 0), ("close", np.inf), ("close", np.nan), ("volume", -1)]
)
def test_invalid_values(column: str, value: float) -> None:
    source = bars().astype(float)
    source.loc[source.index[0], column] = value
    with pytest.raises(ValueError, match="finite"):
        build_prediction_dataset(source)


def test_multiple_symbols_and_missing_columns() -> None:
    source = bars().assign(symbol="ZCH7")
    source.loc[source.index[0], "symbol"] = "ZCK7"
    with pytest.raises(ValueError, match="symbol"):
        build_prediction_dataset(source)
    with pytest.raises(KeyError, match="volume"):
        build_prediction_dataset(bars().drop(columns="volume"))


def test_constant_prices_have_finite_features() -> None:
    source = bars().assign(close=100.0)
    data = build_prediction_dataset(source, window=3)
    assert not data.empty
    assert (data.drop(columns="target_end").drop(columns="volume") == 0).all().all()


@pytest.mark.parametrize(
    "fraction,gap", [(0, 0), (1, 0), (0.2, -1), (0.2, True), (np.nan, 0)]
)
def test_invalid_split_parameters(fraction: float, gap: int) -> None:
    with pytest.raises(ValueError):
        chronological_split(build_prediction_dataset(bars()), fraction, gap)


def test_insufficient_data_and_invalid_target_end() -> None:
    empty = build_prediction_dataset(bars().iloc[:2], window=3)
    with pytest.raises(ValueError):
        chronological_split(empty)
    data = build_prediction_dataset(bars(), window=3)
    with pytest.raises(ValueError, match="Not enough"):
        chronological_split(data, gap=len(data))
    with pytest.raises(ValueError, match="purging"):
        chronological_split(data.iloc[:2])
    data["target_end"] = data.index
    with pytest.raises(ValueError, match="later timestamp"):
        chronological_split(data)


@pytest.mark.parametrize("kind", ["missing", "string", "timezone", "nat"])
def test_target_metadata_is_required_and_valid(kind: str) -> None:
    data = build_prediction_dataset(bars(), window=3)
    if kind == "missing":
        with pytest.raises(KeyError, match="target_end"):
            chronological_split(data.drop(columns="target_end"))
        return
    if kind == "string":
        data["target_end"] = "tomorrow"
    elif kind == "timezone":
        data["target_end"] = data["target_end"].dt.tz_localize(None)
    else:
        data.loc[data.index[0], "target_end"] = pd.NaT
    with pytest.raises(ValueError, match="target_end"):
        chronological_split(data)
