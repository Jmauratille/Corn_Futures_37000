"""Feasibility tools for the USDA corn-report surprise study.

The project README proposes regressing ZC price moves on USDA *surprises* --
the actual Crop Production / Grain Stocks figure minus the pre-report market
expectation. Before committing to that, issue #13 asks whether the required
data can be obtained and joined without look-ahead.

This module provides the small, pure building blocks needed to answer that:
compute a surprise, assemble a *look-ahead-safe* sample of observations, and
measure the ZC price reaction in a window around each release. It deliberately
does not fetch anything over the network; the data-availability assessment and
recommendation live in ``docs/usda_feasibility.md``.

All functions are pure and operate on ``pandas`` objects, so the whole
pipeline can be exercised on synthetic data with no USDA or Databento access.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Report families in scope for the feasibility check.
REPORT_TYPES = ("crop_production", "grain_stocks")

SAMPLE_COLUMNS = (
    "report",
    "expected_time",
    "expected",
    "actual",
    "surprise",
    "pct_surprise",
    "unit",
)


@dataclass(frozen=True)
class SurpriseObservation:
    """One USDA release paired with the expectation known before it.

    Attributes:
        report: Report family, one of :data:`REPORT_TYPES`.
        expected_time: When the market expectation was known. Must be strictly
            before ``release_time`` so the expectation never uses the outcome.
        release_time: Scheduled, timezone-aware USDA release timestamp.
        expected: Pre-report market expectation (consensus), in ``unit``.
        actual: Value published *at release*, not a later revision, in ``unit``.
        unit: Unit of ``expected`` and ``actual``, e.g. ``"million bushels"``.
    """

    report: str
    expected_time: pd.Timestamp
    release_time: pd.Timestamp
    expected: float
    actual: float
    unit: str


def compute_surprise(expected: float, actual: float) -> tuple[float, float]:
    """Return the surprise and percentage surprise of a release.

    Args:
        expected: Pre-report market expectation.
        actual: Value published at release.

    Returns:
        A ``(surprise, pct_surprise)`` pair where ``surprise = actual -
        expected`` and ``pct_surprise = surprise / expected``.

    Raises:
        ValueError: If ``expected`` is zero or either input is not finite.

    Examples:
        >>> compute_surprise(400.0, 410.0)
        (10.0, 0.025)

    """
    if not (np.isfinite(expected) and np.isfinite(actual)):
        msg = "expected and actual must be finite."
        raise ValueError(msg)
    if expected == 0:
        msg = "expected must be non-zero to form a percentage surprise."
        raise ValueError(msg)
    surprise = actual - expected
    return surprise, surprise / expected


def _check_tz_aware(name: str, ts: pd.Timestamp) -> None:
    if not isinstance(ts, pd.Timestamp) or pd.isna(ts) or ts.tzinfo is None:
        msg = f"{name} must be a nonmissing, timezone-aware Timestamp."
        raise ValueError(msg)


def build_surprise_sample(
    observations: Sequence[SurpriseObservation],
) -> pd.DataFrame:
    """Validate observations and join them into a look-ahead-safe sample.

    Args:
        observations: Releases paired with their pre-report expectations.

    Returns:
        A ``DataFrame`` indexed by ``release_time`` (sorted, unique) with the
        columns in :data:`SAMPLE_COLUMNS`. ``surprise`` and ``pct_surprise``
        are computed with :func:`compute_surprise`.

    Raises:
        ValueError: If ``observations`` is empty, a report type is unknown,
            timestamps are naive/missing, an expectation is not known strictly
            before its release (look-ahead), or release times are not unique.

    """
    if len(observations) == 0:
        msg = "observations must not be empty."
        raise ValueError(msg)
    rows = []
    for obs in observations:
        if obs.report not in REPORT_TYPES:
            msg = f"Unknown report {obs.report!r}; expected one of {REPORT_TYPES}."
            raise ValueError(msg)
        _check_tz_aware("expected_time", obs.expected_time)
        _check_tz_aware("release_time", obs.release_time)
        if obs.expected_time >= obs.release_time:
            msg = (
                "expected_time must be strictly before release_time to avoid "
                f"look-ahead; got {obs.expected_time} >= {obs.release_time}."
            )
            raise ValueError(msg)
        surprise, pct = compute_surprise(obs.expected, obs.actual)
        rows.append(
            {
                "release_time": obs.release_time,
                "report": obs.report,
                "expected_time": obs.expected_time,
                "expected": obs.expected,
                "actual": obs.actual,
                "surprise": surprise,
                "pct_surprise": pct,
                "unit": obs.unit,
            }
        )
    sample = pd.DataFrame(rows).set_index("release_time").sort_index()
    if not sample.index.is_unique:
        msg = "release_time values must be unique."
        raise ValueError(msg)
    return sample[list(SAMPLE_COLUMNS)]


def price_reaction(
    prices: pd.Series,
    release_time: pd.Timestamp,
    window: pd.Timedelta,
) -> float:
    """Measure the ZC price move in a window after a release.

    The reaction is the simple return from the last price strictly *before*
    the release to the last price in ``[release_time, release_time + window]``.
    Only prices at or after the release contribute to the post value, so the
    measurement never looks past the window.

    Args:
        prices: Price series indexed by a sorted, timezone-aware
            ``DatetimeIndex``.
        release_time: Timezone-aware release timestamp.
        window: Positive horizon after the release over which to measure.

    Returns:
        The simple return ``post / pre - 1`` across the release.

    Raises:
        ValueError: If the index is not sorted/timezone-aware, ``window`` is
            not positive, or there is no price on one side of the release.

    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        msg = "prices must have a timezone-aware DatetimeIndex."
        raise ValueError(msg)
    if not prices.index.is_monotonic_increasing:
        msg = "prices index must be sorted."
        raise ValueError(msg)
    _check_tz_aware("release_time", release_time)
    if window <= pd.Timedelta(0):
        msg = "window must be a positive Timedelta."
        raise ValueError(msg)
    pre = prices.loc[prices.index < release_time]
    post = prices.loc[
        (prices.index >= release_time) & (prices.index <= release_time + window)
    ]
    if pre.empty or post.empty:
        msg = "Need a price before the release and one within the window after it."
        raise ValueError(msg)
    return float(post.iloc[-1]) / float(pre.iloc[-1]) - 1.0


def join_reactions(
    sample: pd.DataFrame,
    prices: pd.Series,
    window: pd.Timedelta,
) -> pd.DataFrame:
    """Attach a price reaction to each release in a surprise sample.

    Args:
        sample: Output of :func:`build_surprise_sample`.
        prices: ZC price series indexed by a timezone-aware ``DatetimeIndex``.
        window: Horizon after each release passed to :func:`price_reaction`.

    Returns:
        A copy of ``sample`` with an added ``reaction`` column. This is the
        small validated joined sample the feasibility check calls for.

    Raises:
        KeyError: If ``sample`` is missing required columns.
        ValueError: Propagated from :func:`price_reaction` for any release.

    """
    missing = set(SAMPLE_COLUMNS) - set(sample.columns)
    if missing:
        msg = f"sample is missing required columns: {sorted(missing)}."
        raise KeyError(msg)
    joined = sample.copy()
    joined["reaction"] = [
        price_reaction(prices, release_time, window)
        for release_time in sample.index
    ]
    return joined
