"""Pull historical market data once and store it locally for reuse.

Downloading months of trade data is slow and costs Databento credits, so
each teammate pulls a contract's history once into a git-ignored data
folder and every notebook or script afterwards loads it from disk::

    python -m finm37000.pull_data --symbol ZCH7 --months 6

then, anywhere in the project::

    >>> from finm37000.data_store import load_history  # doctest: +SKIP
    >>> trades = load_history("ZCH7")  # doctest: +SKIP

Requests are split into calendar-month windows, matching the approach in
``project_data.ipynb``, so no single request is too large.
"""

import logging
import os
from pathlib import Path
from typing import Any

import databento as db
import pandas as pd

from finm37000.cache import _find_dot_git_or_home
from finm37000.time import tz_chicago

DEFAULT_DATASET = "GLBX.MDP3"
TRADE_COLUMNS = ("price", "size", "side")

logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """The directory in which to store pulled raw data.

    Defaults to `data/raw/` at the root of the current repo, or under the
    user home directory when not in a repo. Override with the
    FINM37000_DATA_DIR environment variable.
    """
    env_dir = os.getenv("FINM37000_DATA_DIR")
    if env_dir is not None:
        data_dir = Path(env_dir)
    else:
        data_dir = _find_dot_git_or_home() / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def history_path(
    symbol: str,
    schema: str = "trades",
    data_dir: Path | None = None,
) -> Path:
    """Path of the stored history file for a symbol and schema.

    Args:
        symbol: Raw contract symbol, e.g. `"ZCH7"`.
        schema: Databento schema, e.g. `"trades"`.
        data_dir: Directory holding the data. Default is `get_data_dir()`.

    Returns:
        The parquet file path, which may not exist yet.

    Examples:
        >>> history_path("ZCH7", data_dir=Path("data")).as_posix()
        'data/ZCH7_trades.parquet'

    """
    directory = get_data_dir() if data_dir is None else data_dir
    return directory / f"{symbol}_{schema}.parquet"


def monthly_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split `[start, end)` into half-open windows at calendar-month boundaries.

    Args:
        start: Start of the range, inclusive.
        end: End of the range, exclusive.

    Returns:
        Consecutive, non-overlapping `(left, right)` windows covering the range.

    Examples:
        >>> windows = monthly_windows(
        ...     pd.Timestamp("2026-04-15"), pd.Timestamp("2026-06-10")
        ... )
        >>> for left, right in windows:
        ...     print(left.date(), right.date())
        2026-04-15 2026-05-01
        2026-05-01 2026-06-01
        2026-06-01 2026-06-10

    """
    windows = []
    cursor = start
    while cursor < end:
        next_month = cursor.normalize().replace(day=1) + pd.DateOffset(months=1)
        stop = min(next_month, end)
        windows.append((cursor, stop))
        cursor = stop
    return windows


def default_range(
    client: db.Historical,
    months: int = 6,
    dataset: str = DEFAULT_DATASET,
    schema: str = "trades",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The last `months` of complete days available from Databento.

    The range ends at the earlier of midnight today and the last midnight
    with available data, both in Chicago time.

    Args:
        client: Databento client to make metadata requests.
        months: Number of calendar months to cover.
        dataset: Databento dataset code.
        schema: Databento schema whose availability bounds the range.

    Returns:
        The `(start, end)` range in Chicago time, end exclusive.

    """
    availability: dict[str, Any] = client.metadata.get_dataset_range(dataset=dataset)
    schema_range = availability.get("schema", {}).get(schema, availability)
    available_end = pd.Timestamp(schema_range["end"]).tz_convert(tz_chicago)
    today = pd.Timestamp.now(tz=tz_chicago).normalize()
    end = min(today, available_end.normalize())
    start = end - pd.DateOffset(months=months)
    return start, end


def _validate(df: pd.DataFrame, symbol: str, schema: str) -> None:
    if "symbol" in df.columns:
        symbols = set(df["symbol"].dropna())
        if symbols != {symbol}:
            msg = f"Expected only {symbol!r} but received {sorted(symbols)}."
            raise ValueError(msg)
    if schema == "trades":
        missing = set(TRADE_COLUMNS) - set(df.columns)
        if missing:
            msg = f"Trade data is missing required columns: {sorted(missing)}."
            raise KeyError(msg)


def pull_history(  # noqa PLR0913
    client: db.Historical,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    schema: str = "trades",
    dataset: str = DEFAULT_DATASET,
    data_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a contract's history and store it as one parquet file.

    Does nothing if the file already exists unless `overwrite` is set, so
    re-running a notebook never pays for the same data twice. The file is
    written only after every monthly request succeeds and the data passes
    validation, so a failed pull never leaves a partial file behind.

    Args:
        client: Databento client to make data requests.
        symbol: Raw contract symbol, e.g. `"ZCH7"`.
        start: Start of the requested range, inclusive.
        end: End of the requested range, exclusive.
        schema: Databento schema, e.g. `"trades"`.
        dataset: Databento dataset code.
        data_dir: Directory to store the file in. Default is `get_data_dir()`.
        overwrite: Download again even if the file already exists.

    Returns:
        The path of the stored parquet file.

    Raises:
        ValueError: If `start` is not before `end`, no records are returned,
            or the records contain an unexpected symbol.
        KeyError: If trade data is missing price, size, or side.

    """
    if start >= end:
        msg = f"start must be before end, got start={start} and end={end}."
        raise ValueError(msg)
    path = history_path(symbol, schema, data_dir)
    if path.exists() and not overwrite:
        logger.info("Using existing %s; pass overwrite=True to re-pull.", path)
        return path

    chunks = []
    for left, right in monthly_windows(start, end):
        logger.info("Pulling %s %s: %s -> %s", symbol, schema, left, right)
        chunk = client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            symbols=symbol,
            stype_in="raw_symbol",
            start=left,
            end=right,
        ).to_df()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        msg = f"No {schema} records returned for {symbol} in [{start}, {end})."
        raise ValueError(msg)

    # Windows are half-open, so concatenating cannot duplicate records.
    df = pd.concat(chunks).sort_index(kind="stable")
    _validate(df, symbol, schema)

    tmp_path = path.with_suffix(".tmp")
    df.to_parquet(tmp_path)
    tmp_path.replace(path)
    logger.info("Saved %d records to %s", len(df), path)
    return path


def load_history(
    symbol: str,
    schema: str = "trades",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a stored contract history.

    Args:
        symbol: Raw contract symbol, e.g. `"ZCH7"`.
        schema: Databento schema, e.g. `"trades"`.
        data_dir: Directory holding the data. Default is `get_data_dir()`.

    Returns:
        The stored records, sorted chronologically.

    Raises:
        FileNotFoundError: If the history has not been pulled yet.

    """
    path = history_path(symbol, schema, data_dir)
    if not path.exists():
        msg = (
            f"No stored {schema} for {symbol} at {path}. Pull it once with: "
            f"python -m finm37000.pull_data --symbol {symbol} --schema {schema}"
        )
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)
