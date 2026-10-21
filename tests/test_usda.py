import numpy as np
import pandas as pd
import pytest

from finm37000.usda import (
    SurpriseObservation,
    build_surprise_sample,
    compute_surprise,
    join_reactions,
    price_reaction,
)

TZ = "America/Chicago"


def _obs(
    report: str = "crop_production",
    expected: float = 15000.0,
    actual: float = 15200.0,
    release: str = "2026-08-12 11:00",
    expected_lead_hours: int = 24,
) -> SurpriseObservation:
    release_time = pd.Timestamp(release, tz=TZ)
    return SurpriseObservation(
        report=report,
        expected_time=release_time - pd.Timedelta(hours=expected_lead_hours),
        release_time=release_time,
        expected=expected,
        actual=actual,
        unit="million bushels",
    )


def test_compute_surprise_values() -> None:
    surprise, pct = compute_surprise(400.0, 410.0)
    assert surprise == pytest.approx(10.0)
    assert pct == pytest.approx(0.025)


def test_compute_surprise_rejects_zero_expected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        compute_surprise(0.0, 5.0)


def test_build_surprise_sample_sorts_and_computes() -> None:
    later = _obs(release="2026-09-30 11:00", expected=1800.0, actual=1750.0,
                 report="grain_stocks")
    earlier = _obs(release="2026-08-12 11:00", expected=15000.0, actual=15200.0)
    sample = build_surprise_sample([later, earlier])
    assert list(sample.index) == [earlier.release_time, later.release_time]
    assert sample.loc[earlier.release_time, "surprise"] == pytest.approx(200.0)
    later_pct = sample.loc[later.release_time, "pct_surprise"]
    assert later_pct == pytest.approx(-50.0 / 1800.0)
    assert sample.index.is_monotonic_increasing


def test_build_surprise_sample_rejects_lookahead() -> None:
    bad = SurpriseObservation(
        report="crop_production",
        expected_time=pd.Timestamp("2026-08-12 11:00", tz=TZ),
        release_time=pd.Timestamp("2026-08-12 11:00", tz=TZ),
        expected=15000.0,
        actual=15200.0,
        unit="million bushels",
    )
    with pytest.raises(ValueError, match="look-ahead"):
        build_surprise_sample([bad])


def test_build_surprise_sample_rejects_naive_timestamp() -> None:
    bad = SurpriseObservation(
        report="crop_production",
        expected_time=pd.Timestamp("2026-08-11 11:00"),
        release_time=pd.Timestamp("2026-08-12 11:00", tz=TZ),
        expected=15000.0,
        actual=15200.0,
        unit="million bushels",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        build_surprise_sample([bad])


def test_build_surprise_sample_rejects_unknown_report() -> None:
    with pytest.raises(ValueError, match="Unknown report"):
        build_surprise_sample([_obs(report="wasde")])


def test_build_surprise_sample_rejects_empty() -> None:
    with pytest.raises(ValueError, match="not be empty"):
        build_surprise_sample([])


def _price_series() -> pd.Series:
    index = pd.date_range("2026-08-12 10:00", periods=7, freq="30min", tz=TZ)
    # Prices step up right after the 11:00 release.
    values = [100.0, 100.0, 100.0, 104.0, 105.0, 105.0, 106.0]
    return pd.Series(values, index=index)


def test_price_reaction_measures_move_after_release() -> None:
    prices = _price_series()
    release = pd.Timestamp("2026-08-12 11:00", tz=TZ)
    # pre = last price before 11:00 (100.0); post = last price <= 12:00 (105.0).
    reaction = price_reaction(prices, release, pd.Timedelta(hours=1))
    assert reaction == pytest.approx(105.0 / 100.0 - 1.0)


def test_price_reaction_requires_post_price_in_window() -> None:
    prices = _price_series()
    # Release falls in the 10:30 -> 11:00 gap; a 1-minute window has no price.
    release = pd.Timestamp("2026-08-12 10:45", tz=TZ)
    with pytest.raises(ValueError, match="within the window"):
        price_reaction(prices, release, pd.Timedelta(minutes=1))


def test_price_reaction_rejects_nonpositive_window() -> None:
    prices = _price_series()
    release = pd.Timestamp("2026-08-12 11:00", tz=TZ)
    with pytest.raises(ValueError, match="positive"):
        price_reaction(prices, release, pd.Timedelta(0))


def test_price_reaction_rejects_naive_index() -> None:
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2026-08-12", periods=2))
    release = pd.Timestamp("2026-08-12 11:00", tz=TZ)
    with pytest.raises(ValueError, match="timezone-aware"):
        price_reaction(naive, release, pd.Timedelta(hours=1))


def test_join_reactions_adds_reaction_column() -> None:
    sample = build_surprise_sample([_obs()])
    prices = _price_series()
    joined = join_reactions(sample, prices, pd.Timedelta(hours=1))
    assert "reaction" in joined.columns
    assert joined["reaction"].notna().all()
    assert np.isfinite(joined["reaction"].to_numpy()).all()
    # Original sample is not mutated.
    assert "reaction" not in sample.columns


def test_join_reactions_requires_sample_columns() -> None:
    prices = _price_series()
    with pytest.raises(KeyError, match="missing required columns"):
        join_reactions(pd.DataFrame({"foo": [1]}), prices, pd.Timedelta(hours=1))
