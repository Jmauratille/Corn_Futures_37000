"""Fetch OHLCV bars and trades from Databento."""

from typing import Sequence, Union

import databento as db
import pandas as pd

from finm37000.continuous import (
    additive_splice,
    multiplicative_splice,
    splice_unadjusted,
)


def fetch_ohlcv(  # noqa PLR0913
    client: db.Historical,
    symbols: Union[str, Sequence[str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    dataset: str = "GLBX.MDP3",
    schema: str = "ohlcv-1d",
    stype_in: str = "raw_symbol",
) -> pd.DataFrame:
    """Fetch OHLCV bars for one or more symbols.

    Args:
        client: Databento client to make data requests.
        symbols: One or more symbols, interpreted according to `stype_in`.
                 Use `stype_in="continuous"` for symbols like `"CL.v.0"`,
                 or `stype_in="instrument_id"` with numeric ids.
        start: Start of the requested range.
        end: End of the requested range.
        dataset: Databento dataset code.
        schema: Bar schema, e.g. `"ohlcv-1d"` or `"ohlcv-1m"`.
        stype_in: Symbology of `symbols`.

    Returns:
        A `pd.DataFrame` indexed by `ts_event`, sorted chronologically.

    """
    return (
        client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            symbols=symbols,
            stype_in=stype_in,
            start=start,
            end=end,
        )
        .to_df()
        .sort_index()
    )


def fetch_trades(
    client: db.Historical,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dataset: str = "GLBX.MDP3",
) -> pd.DataFrame:
    """Fetch raw trades for a single symbol.

    Args:
        client: Databento client to make data requests.
        symbol: Raw contract symbol, e.g. `"CLQ6"`.
        start: Start of the requested range.
        end: End of the requested range.
        dataset: Databento dataset code.

    Returns:
        A `pd.DataFrame` indexed by `ts_recv`, sorted chronologically.

    """
    return (
        client.timeseries.get_range(
            dataset=dataset,
            schema="trades",
            symbols=symbol,
            start=start,
            end=end,
        )
        .to_df()
        .sort_index()
    )


def fetch_continuous_ohlcv(  # noqa PLR0913
    client: db.Historical,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    adjustment: str = "none",
    dataset: str = "GLBX.MDP3",
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Fetch a roll-adjusted continuous contract.

    The adjustment may be:

    - ``"none"``: the unadjusted splice with a price
      jump wherever the active leg changes.
    - ``"additive"``: forward additive adjustment, see
      `finm37000.continuous.additive_splice`.
    - ``"multiplicative"``: forward multiplicative adjustment, see
      `finm37000.continuous.multiplicative_splice`.

    Args:
        client: Databento client to make data requests.
        symbol: Continuous symbol, e.g. `"CL.v.0"`.
        start: Start of the requested range.
        end: End of the requested range.
        adjustment: Roll-adjustment mode: `"none"`, `"additive"`, or
                    `"multiplicative"`.
        dataset: Databento dataset code.
        price_cols: Columns to adjust; must all be present in the `ohlcv-1d`
                    schema. Ignored when `adjustment="none"`.

    Returns:
        A `pd.DataFrame` indexed by `ts_event`, sorted chronologically.

    Raises:
        ValueError: If `adjustment` is not a recognized mode.

    """
    roll_spec = client.symbology.resolve(
        dataset=dataset,
        symbols=[symbol],
        stype_in="continuous",
        stype_out="instrument_id",
        start_date=start.date(),
        end_date=end.date(),
    )["result"][symbol]
    required_ids = [spec["s"] for spec in roll_spec]
    legs = fetch_ohlcv(
        client, required_ids, start, end, dataset=dataset, stype_in="instrument_id"
    ).reset_index()

    if adjustment == "none":
        spliced = splice_unadjusted(roll_spec, legs, date_col="ts_event")
    elif adjustment == "additive":
        spliced = additive_splice(
            roll_spec,
            legs,
            date_col="ts_event",
            adjust_by="close",
            adjustment_cols=list(price_cols),
        )
    elif adjustment == "multiplicative":
        spliced = multiplicative_splice(
            roll_spec,
            legs,
            date_col="ts_event",
            adjust_by="close",
            adjustment_cols=list(price_cols),
        )
    else:
        msg = (
            f"Unknown adjustment {adjustment!r}; "
            "expected 'none', 'additive', or 'multiplicative'."
        )
        raise ValueError(msg)

    return spliced.set_index("ts_event").sort_index()
