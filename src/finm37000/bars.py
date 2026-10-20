"""Aggregate single-contract trades into aligned, completed analysis bars."""

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from .features import signed_volume


def _event_index(trades: pd.DataFrame) -> pd.DatetimeIndex:
    if "ts_event" in trades:
        if not pd.api.types.is_datetime64_any_dtype(trades["ts_event"]):
            msg = "ts_event must contain timezone-aware datetime values."
            raise ValueError(msg)
        index = pd.DatetimeIndex(trades["ts_event"])
    elif isinstance(trades.index, pd.DatetimeIndex) and trades.index.name in {
        "ts_event",
        "event_time",
    }:
        index = trades.index
    else:
        msg = (
            "Supply a ts_event column or an event-time index named ts_event/event_time."
        )
        raise ValueError(msg)
    if index.hasnans or index.tz is None:
        msg = "Event timestamps must be nonmissing and timezone-aware."
        raise ValueError(msg)
    return index.tz_convert("UTC").rename("bar_end")


def _validate_trades(trades: pd.DataFrame) -> None:
    missing = {"symbol", "price", "size", "side"} - set(trades.columns)
    if missing:
        msg = f"Missing trade columns: {sorted(missing)}."
        raise KeyError(msg)
    if trades.empty:
        msg = "trades must not be empty."
        raise ValueError(msg)
    if trades["symbol"].isna().any() or len(trades["symbol"].unique()) != 1:
        msg = "Select exactly one nonmissing contract symbol before aggregation."
        raise ValueError(msg)
    for column in ("price", "size"):
        series = trades[column]
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
            or series.isna().any()
            or not np.isfinite(series.to_numpy(dtype=float)).all()
            or (series <= 0).any()
        ):
            msg = f"{column} must be finite, positive real numbers."
            raise ValueError(msg)
    if (trades["size"] % 1 != 0).any():
        msg = "size must count whole contracts."
        raise ValueError(msg)
    if not trades["side"].isin(["B", "A", "N"]).all():
        msg = "side must be B (buy), A (sell), or N (unknown)."
        raise ValueError(msg)


def build_analysis_bars(
    trades: pd.DataFrame,
    rule: str = "5min",
    *,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Build OHLCV and order flow with shared intervals and bar-end timestamps.

    Args:
        trades: One contract's Databento trades: symbol, price, size, side,
            and timezone-aware ts_event column or event-time DatetimeIndex
            named ts_event/event_time. Duplicate timestamps are retained;
            unsorted trades are stably sorted, preserving tie order.
        rule: Positive fixed-duration pandas frequency, e.g. "5min" or "1h".
            Buckets are anchored at UTC midnight (not exchange sessions).
        end: Exclusive, timezone-aware event-time cutoff known by the caller.
            Only trades before end and bars fully ending by end are returned.
            Supply the actual coverage endpoint, not a desired future date.

    Returns:
        Nonempty-trade intervals indexed by bar_end in UTC, with symbol,
        OHLC prices, volume, trade_count, buy_volume, sell_volume,
        unknown_volume and ofi (buy minus sell). Every interval is [start, end)
        and labeled at its end. A trade on a boundary belongs to the next bar.
        Empty intervals are omitted, never filled. A partial last bar is
        omitted; therefore the returned frame may be empty. Inputs are unchanged.

    Raises:
        KeyError: If required trade columns are absent.
        ValueError: For invalid records, timestamps, rule or cutoff, mixed
            contracts, empty input, or no trades before the cutoff.

    Notes:
        B/A/N follow Databento Trade aggressor conventions. Unknown-side volume
        contributes to OHLCV but not OFI, and is reported separately. The data
        must consist of trade records. Price units are unchanged.
        Event-time completion does not model delayed market-data receipt or
        guarantee execution at the closing price; those need separate handling.
    """
    _validate_trades(trades)
    offset = to_offset(rule)
    if not isinstance(offset, Tick) or offset.nanos <= 0:
        msg = "rule must be a positive fixed-duration frequency."
        raise ValueError(msg)
    if not isinstance(end, pd.Timestamp) or pd.isna(end) or end.tz is None:
        msg = "end must be a nonmissing, timezone-aware pandas Timestamp."
        raise ValueError(msg)
    cutoff = end.tz_convert("UTC")
    frame = trades.copy()
    frame.index = _event_index(trades)
    frame = frame.loc[frame.index < cutoff].sort_index(kind="stable")
    if frame.empty:
        msg = "No trades before end."
        raise ValueError(msg)
    frame["buy_volume"] = frame["size"].where(frame["side"] == "B", 0)
    frame["sell_volume"] = frame["size"].where(frame["side"] == "A", 0)
    frame["unknown_volume"] = frame["size"].where(frame["side"] == "N", 0)
    frame["ofi"] = signed_volume(frame)
    bars = frame.resample(rule, label="right", closed="left", origin="start_day").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        trade_count=("size", "count"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        unknown_volume=("unknown_volume", "sum"),
        ofi=("ofi", "sum"),
    )
    bars = bars.loc[(bars["trade_count"] > 0) & (bars.index <= cutoff)].copy()
    bars.insert(0, "symbol", frame["symbol"].iloc[0])
    return bars
