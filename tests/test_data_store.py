from pathlib import Path
from typing import Any, cast

import databento as db
import pandas as pd
import pytest

from finm37000.data_store import (
    get_data_dir,
    history_path,
    load_history,
    monthly_windows,
    pull_history,
)

START = pd.Timestamp("2026-04-15", tz="UTC")
END = pd.Timestamp("2026-06-10", tz="UTC")
# Part of April, all of May, and part of June.
NUM_WINDOWS = 3


class _FakeStore:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df


class _FakeTimeseries:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.calls: list[dict[str, Any]] = []

    def get_range(self, **kwargs: Any) -> _FakeStore:  # noqa: ANN401
        self.calls.append(kwargs)
        # One trade per requested window, stamped at the window start.
        index = pd.DatetimeIndex([kwargs["start"]], name="ts_recv")
        df = pd.DataFrame(
            {
                "symbol": [self.symbol],
                "price": [450.0 + len(self.calls)],
                "size": [len(self.calls)],
                "side": ["B"],
            },
            index=index,
        )
        return _FakeStore(df)


class _FakeClient:
    def __init__(self, symbol: str = "ZCH7") -> None:
        self.timeseries = _FakeTimeseries(symbol)


def _client(symbol: str = "ZCH7") -> tuple[db.Historical, _FakeTimeseries]:
    fake = _FakeClient(symbol)
    return cast("db.Historical", fake), fake.timeseries


def test_monthly_windows_cover_range_without_gaps() -> None:
    windows = monthly_windows(START, END)
    assert windows[0][0] == START
    assert windows[-1][1] == END
    for (_, right), (left, _) in zip(windows, windows[1:], strict=False):
        assert right == left
    assert len(windows) == NUM_WINDOWS


def test_monthly_windows_empty_when_start_not_before_end() -> None:
    assert monthly_windows(END, START) == []


def test_get_data_dir_respects_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "custom"
    monkeypatch.setenv("FINM37000_DATA_DIR", str(target))
    assert get_data_dir() == target
    assert target.is_dir()


def test_pull_history_requests_each_month_and_stores(tmp_path: Path) -> None:
    client, timeseries = _client()
    path = pull_history(client, "ZCH7", START, END, data_dir=tmp_path)
    assert path == history_path("ZCH7", data_dir=tmp_path)
    assert len(timeseries.calls) == NUM_WINDOWS
    assert {call["symbols"] for call in timeseries.calls} == {"ZCH7"}
    stored = load_history("ZCH7", data_dir=tmp_path)
    assert len(stored) == NUM_WINDOWS
    assert stored.index.is_monotonic_increasing
    assert not list(tmp_path.glob("*.tmp"))


def test_pull_history_skips_existing_file(tmp_path: Path) -> None:
    client, timeseries = _client()
    pull_history(client, "ZCH7", START, END, data_dir=tmp_path)
    pull_history(client, "ZCH7", START, END, data_dir=tmp_path)
    assert len(timeseries.calls) == NUM_WINDOWS


def test_pull_history_overwrite_pulls_again(tmp_path: Path) -> None:
    client, timeseries = _client()
    pull_history(client, "ZCH7", START, END, data_dir=tmp_path)
    pull_history(client, "ZCH7", START, END, data_dir=tmp_path, overwrite=True)
    assert len(timeseries.calls) == 2 * NUM_WINDOWS


def test_pull_history_rejects_unexpected_symbol(tmp_path: Path) -> None:
    client, _ = _client(symbol="ZCZ6")
    with pytest.raises(ValueError, match="ZCZ6"):
        pull_history(client, "ZCH7", START, END, data_dir=tmp_path)
    assert not history_path("ZCH7", data_dir=tmp_path).exists()


def test_pull_history_rejects_reversed_range(tmp_path: Path) -> None:
    client, _ = _client()
    with pytest.raises(ValueError, match="start must be before end"):
        pull_history(client, "ZCH7", END, START, data_dir=tmp_path)


def test_load_history_missing_file_explains_how_to_pull(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m finm37000.pull_data"):
        load_history("ZCH7", data_dir=tmp_path)
