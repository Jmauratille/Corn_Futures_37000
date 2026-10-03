"""Save DataFrames to a cache directory for faster reloads."""

import functools
import hashlib
import inspect
import logging
import os
import re
from collections.abc import Callable, Sequence
from datetime import date, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd

type DataFetcher[**P] = Callable[P, pd.DataFrame]

logger = logging.getLogger(__name__)


def _find_dot_git_or_home() -> Path:
    current = Path.cwd().resolve()

    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent

    return Path.home()


def _compute_short_checksum(val: Any, n: int = 4) -> str:  # noqa: ANN401
    raw_str = f"{type(val).__name__}_{str(val)}"
    return hashlib.md5(raw_str.encode("utf-8"), usedforsecurity=False).hexdigest()[:n]


def _serialize_arg_value(val: Any) -> str:  # noqa: ANN401
    if isinstance(val, (datetime, date, pd.Timestamp)):
        return val.isoformat()
    if isinstance(val, (int, float, str, bool, type(None))):
        return str(val)

    num_explicit_items = 3
    if isinstance(val, (list, tuple, set)):
        some_items = (_serialize_arg_value(x) for x in islice(val, num_explicit_items))
        substring = f"[{','.join(some_items)}"
        suffix = (
            "]"
            if len(val) < num_explicit_items
            else f",{_compute_short_checksum(val)}]"
        )
        return substring + suffix

    class_name = type(val).__name__
    checksum = _compute_short_checksum(val)
    return f"{class_name}(hash={checksum})"


def _generate_readable_arg_str(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    args_to_ignore: Sequence[str],
) -> str:
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    parts = []
    for name, value in bound_args.arguments.items():
        if name in args_to_ignore:
            continue
        safe_val = _serialize_arg_value(value)
        parts.append(f"{name}={safe_val}")

    raw_str = "_".join(parts) if parts else "no_args"
    clean_str = re.sub(r'[\s\\/:*?"<>|]', "", raw_str)
    max_arg_str_len = 120
    hash_len = 10
    if len(clean_str) > max_arg_str_len:
        hash_str = _compute_short_checksum(bound_args, n=hash_len)
        clean_str = f"{clean_str[: (max_arg_str_len - hash_len)]}_{hash_str}"
    return clean_str


def generate_cache_filename(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    args_to_ignore: Sequence[str],
) -> str:
    """Create readable, legal file name from function and arguments.

    Complex arguments get hashed. Not guaranteed unique. Just something
    simple to support course notebooks.
    """
    arg_str = _generate_readable_arg_str(
        func,
        args,
        kwargs,
        args_to_ignore=args_to_ignore,
    )
    ext = "parquet"
    return f"{func.__name__}_{arg_str}.{ext}"


def get_cache_dir() -> Path:
    """The directory in which to cache data.

    Defaults to `.cache/finm37000/` in the current repo
    by default. When not in a repo, the default puts that in the
    user home directory. Override with FINM37000_CACHE_DIR environment
    variable.
    """
    env_dir = os.getenv("FINM37000_CACHE_DIR")
    if env_dir is not None:
        cache_dir = Path(env_dir)
    else:
        cache_dir = _find_dot_git_or_home() / ".cache" / "finm37000"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_dataframe[**P](
    cache_file: Path,
    fetch_fn: DataFetcher[P],
    *args: P.args,
    **kwargs: P.kwargs,
) -> pd.DataFrame:
    """Read a cached DataFrame or run and cache the fetch function.

    A failed save will log a warning and return the data.
    A failed read is treated as a missing cache and will proceed
    to call the `fetch_fn`.
    """
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            logger.info("Read cache_file %s.", cache_file)
            return df
        except Exception as e:
            logger.debug("Cache read failed for %s: %s", cache_file, e)

    df = fetch_fn(*args, **kwargs)

    try:
        df.to_parquet(cache_file)
        logger.info("Saved data to cache_file=%s", cache_file)
    except Exception as e:
        logger.warning("Failed to write cache_file=%s: %s", cache_file, e)
    return df


def disk_cache[**P](
    scope: str | None = None,
    args_to_ignore: Sequence[str] | None = None,
) -> Callable[[Callable[P, pd.DataFrame]], Callable[P, pd.DataFrame]]:
    """Cache a DataFrame producing function.

    Decorate pd.DataFrame producing functions to save to disk
    as parquet files. See get_cache_dir() for where the cache
    is.

    scope: Name of a subdirectory of get_cache_dir() where data files will go.
    args_to_ignore: Subset of arguments to use in the cache filename. Default None
        will ignore ``("self", "cls", "client", "dataset")``.
        A bare ``str`` is not supported even though it is a ``Sequence[str]`` to avoid
        confusion about what ``in args_to_ignore` means.
    """
    if isinstance(args_to_ignore, str):
        msg = f"{args_to_ignore=} does not support a bare str."
        raise ValueError(msg)

    def decorator(f: Callable[P, pd.DataFrame]) -> Callable[P, pd.DataFrame]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> pd.DataFrame:
            ignored = (
                ("self", "cls", "client", "dataset")
                if args_to_ignore is None
                else args_to_ignore
            )
            filename = generate_cache_filename(
                f,
                args,
                kwargs,
                args_to_ignore=ignored,
            )

            cache_dir = get_cache_dir()
            if scope is not None:
                cache_dir = cache_dir / scope
                cache_dir.mkdir(parents=True, exist_ok=True)

            df = get_cached_dataframe(
                cache_dir / filename,
                f,
                *args,
                **kwargs,
            )
            return df

        return wrapper

    return decorator
