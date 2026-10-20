import numpy as np
import pandas as pd
import pytest

from finm37000 import build_analysis_bars, build_prediction_dataset

END = pd.Timestamp("2026-10-01 09:20", tz="UTC")


def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["ZCH7"] * 5,
            "price": [100.0, 102.0, 101.0, 99.0, 103.0],
            "size": [2, 3, 4, 5, 6],
            "side": ["B", "A", "N", "B", "A"],
        },
        index=pd.DatetimeIndex(
            [
                "2026-10-01 09:00",
                "2026-10-01 09:04:59",
                "2026-10-01 09:05",
                "2026-10-01 09:05",
                "2026-10-01 09:15",
            ],
            tz="UTC",
            name="ts_event",
        ),
    )


def test_boundary_alignment_and_hand_calculated_values() -> None:
    source = trades()
    before = source.copy(deep=True)
    bars = build_analysis_bars(source, end=END)
    assert bars.index.tolist() == list(
        pd.to_datetime(
            ["2026-10-01 09:05Z", "2026-10-01 09:10Z", "2026-10-01 09:20Z"],
        )
    )
    first = bars.iloc[0]
    assert first[["open", "high", "low", "close"]].tolist() == [100, 102, 100, 102]
    assert first[
        ["volume", "trade_count", "buy_volume", "sell_volume", "ofi"]
    ].tolist() == [5, 2, 2, 3, -1]
    second = bars.iloc[1]
    assert second[
        ["open", "close", "volume", "trade_count", "unknown_volume", "ofi"]
    ].tolist() == [101, 99, 9, 2, 4, 5]
    assert bars["volume"].sum() == source["size"].sum()
    assert (
        bars["volume"]
        == bars[["buy_volume", "sell_volume", "unknown_volume"]].sum(axis=1)
    ).all()
    pd.testing.assert_frame_equal(source, before)


def test_event_column_overrides_receive_index_and_preserves_ties() -> None:
    source = trades().iloc[[4, 2, 3, 0, 1]].copy()
    source["ts_event"] = source.index
    source.index = source.index + pd.Timedelta("1h")
    source.index.name = "ts_recv"
    result = build_analysis_bars(source, end=END)
    pd.testing.assert_frame_equal(result, build_analysis_bars(trades(), end=END))


def test_partial_bar_excluded_and_future_trades_do_not_change_completed_bars() -> None:
    cutoff = pd.Timestamp("2026-10-01 09:07", tz="UTC")
    bars = build_analysis_bars(trades(), end=cutoff)
    assert bars.index.tolist() == [pd.Timestamp("2026-10-01 09:05", tz="UTC")]
    changed = trades()
    changed.loc[changed.index >= cutoff, "price"] = 900.0
    pd.testing.assert_frame_equal(bars, build_analysis_bars(changed, end=cutoff))
    at_boundary = build_analysis_bars(trades(), end=bars.index[0])
    pd.testing.assert_frame_equal(bars, at_boundary)


def test_timezone_conversion_and_missing_intervals() -> None:
    source = trades()
    source.index = pd.DatetimeIndex(source.index).tz_convert("America/Chicago")
    pd.testing.assert_frame_equal(
        build_analysis_bars(source, end=END.tz_convert("America/Chicago")),
        build_analysis_bars(trades(), end=END),
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("price", 0),
        ("price", -1),
        ("price", np.inf),
        ("price", np.nan),
        ("size", 0),
        ("size", -1),
        ("size", np.nan),
        ("size", 1.5),
        ("side", "X"),
        ("side", None),
        ("symbol", None),
        ("symbol", "ZCK7"),
    ],
)
def test_invalid_records_fail_loudly(column: str, value: float | str | None) -> None:
    source = trades().astype({"size": "float64", "side": "object", "symbol": "object"})
    source.iloc[0, source.columns.tolist().index(column)] = value
    with pytest.raises(ValueError):
        build_analysis_bars(source, end=END)


@pytest.mark.parametrize("column", ["symbol", "price", "size", "side"])
def test_missing_columns(column: str) -> None:
    with pytest.raises(KeyError, match=column):
        build_analysis_bars(trades().drop(columns=column), end=END)


@pytest.mark.parametrize("rule", ["0min", "-5min", "ME", "nonsense"])
def test_invalid_frequencies(rule: str) -> None:
    with pytest.raises(ValueError):
        build_analysis_bars(trades(), rule, end=END)


@pytest.mark.parametrize("kind", ["naive", "nat", "received", "string_column"])
def test_invalid_event_timestamps(kind: str) -> None:
    source = trades()
    if kind == "naive":
        source.index = pd.DatetimeIndex(source.index).tz_localize(None)
    elif kind == "nat":
        source.index = pd.DatetimeIndex([pd.NaT, *source.index[1:]], name="ts_event")
    elif kind == "received":
        source.index.name = "ts_recv"
    else:
        source["ts_event"] = "2026-10-01"
    with pytest.raises(ValueError):
        build_analysis_bars(source, end=END)


@pytest.mark.parametrize("end", [pd.Timestamp("2026-10-01"), pd.NaT])
def test_invalid_cutoff(end: pd.Timestamp) -> None:
    with pytest.raises(ValueError, match="end"):
        build_analysis_bars(trades(), end=end)


def test_empty_input_no_pre_cutoff_trades_and_no_complete_bars() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_analysis_bars(trades().iloc[:0], end=END)
    with pytest.raises(ValueError, match="No trades"):
        build_analysis_bars(trades(), end=trades().index[0])
    bars = build_analysis_bars(trades(), end=trades().index[0] + pd.Timedelta("1min"))
    assert bars.empty
    assert bars.index.name == "bar_end"


def test_analysis_bars_integrate_with_prediction_dataset() -> None:
    source = pd.DataFrame(
        {"symbol": "ZCH7", "price": np.arange(100.0, 130.0), "size": 1, "side": "B"},
        index=pd.date_range(
            "2026-10-01", periods=30, freq="5min", tz="UTC", name="ts_event"
        ),
    )
    bars = build_analysis_bars(source, end=source.index[-1] + pd.Timedelta("5min"))
    dataset = build_prediction_dataset(bars, window=3, horizon=2)
    assert not dataset.empty
    assert dataset.iloc[0]["target_return"] == pytest.approx(105 / 103 - 1)
    assert dataset.index[0] == bars.index[3]
