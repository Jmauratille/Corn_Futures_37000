"""Extract futures data from databento objects."""

import datetime
from typing import Callable

import databento as db
import numpy as np
import pandas as pd

from finm37000.time import us_business_day


def get_official_stats(raw_stats: pd.DataFrame, def_df: pd.DataFrame) -> pd.DataFrame:
    """Filter official daily statistics with instrument expiration.

    Args:
        raw_stats: raw daily statistics including columns `instrument_id`,
                   `raw_symbol`, `ts_ref`, `stat_type`, `stat_flags`, `price`,
                    and `quantity` as returned by `databento` clients for
                    futures `statistics` schemas.
        def_df: instrument definitions including columns `instrument_id`,
            `expiration`.

    Returns:
        pd.DataFrame indexed on `Trade date` and `Symbol` with columns
        `Settlement price`, `Cleared volume`, `Open interest`, and `expiration`.

    """
    def_df = def_df[["instrument_id", "expiration", "raw_symbol"]]
    stats_df = raw_stats.merge(def_df, on="instrument_id")
    stats_df = stats_df.rename(columns={"raw_symbol": "Symbol"})
    stats_df["Trade date"] = stats_df["ts_ref"].dt.date
    final_actual_flag = 3
    # CME MDP3 tag 715 SettlPriceType flag: bit 0 = 1 (final) bit 1 = 1 (actual)
    # https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457414586/Settlement+Prices#SettlementPrices-SettlementatTradingTick/SettlementatClearingTick
    # https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457226917/MDP+3.0+-+Settlement+Price
    stats_df["Settlement price"] = stats_df[
        (stats_df["stat_type"] == db.StatType.SETTLEMENT_PRICE)
        & (stats_df["stat_flags"] == final_actual_flag)
    ]["price"]
    stats_df["Cleared volume"] = stats_df[
        stats_df["stat_type"] == db.StatType.CLEARED_VOLUME
    ]["quantity"]
    stats_df["Open interest"] = stats_df[
        stats_df["stat_type"] == db.StatType.OPEN_INTEREST
    ]["quantity"]
    last_df = (
        stats_df.groupby(["Trade date", "Symbol"])
        .agg("last")
        .sort_values(["Trade date", "expiration"])
    )
    return last_df[
        ["Settlement price", "Cleared volume", "Open interest", "expiration"]
    ]


def filter_legs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the futures legs from the data.

    :param df: `pd.DataFrame` with an "instrument_class" and "expiration" column.
    :return: Rows of `df` matching `db.InstrumentClass.FUTURE` indexed and sorted by
        "expiration".
    """
    df = df[df["instrument_class"] == db.InstrumentClass.FUTURE]
    df = df.set_index("expiration").sort_index()
    return df


def get_stats_by_date(
    client: db.Historical,
    date: datetime.date,
    defs: pd.DataFrame,
) -> pd.DataFrame:
    """Attempt to find all futures leg stats for a date.

    Look 2 business days ahead to include a wide enough
    window to cover most delays.

    :param client: Databento client to make data requests.
    :param date: Date on which to get the futures legs.
    :param defs: `pd.DataFrame` of definitions to be filtered for futures legs.
    :return: The `pd.DataFrame` of stats.
    """
    leg_defs = filter_legs(defs)
    legs = leg_defs["raw_symbol"].unique()
    next_bday = (date + 2 * us_business_day).to_pydatetime().date()
    raw_stats = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="statistics",
        symbols=legs,
        start=date,
        end=next_bday,
    )
    raw_df = raw_stats.to_df()
    raw_same_day = raw_df[raw_df["ts_ref"].dt.date == date]
    stats = get_official_stats(raw_same_day, leg_defs.reset_index())
    return stats


type RateCurve = Callable[[float], float]


def build_short_rate_curve(
    stats: pd.DataFrame,
    start: pd.Timestamp,
    days_per_year: float = 360.0,
) -> RateCurve:
    """Bootstrap a zero-rate curve from one date's STIR futures settlement prices.

    Takes settlement prices from daily `stats` data frame.
    Bootstrap a discount curve by compounding the legs
    sequentially in expiration order, treating each leg's quoted rate as
    applying over the incremental period since the prior leg's expiration
    (the front leg's own rate covers the stub from `start` to its
    expiration). The resulting discount factors are converted to
    continuously-compounded zero rates and returned as a function mapping a
    time to expiration (in years) to an interpolated (flat-extrapolated)
    rate.

    Strips out NaN data which sometimes comes from the data chain for
    an expired contract that should not be part of the curve.

    Parameters
    ----------
    stats : pd.DataFrame
        One row per leg for the target date, with `expiration` and
        `Settlement price` columns. See `get_official_stats` and
        `get_stats_by_date`.
    start : pd.Timestamp
        The date to build the curve as of; also the years-to-expiration anchor.
    days_per_year : float
        The number of days to use to normalize day counts.
        Defaults to 360 to follow SR3 conventions, but otherwise is not
        completely faithful to those conventions.

    Returns:
    -------
    Callable[[float], float]
        Maps a time to expiration in years to an interpolated zero rate.

    """
    stats = stats.sort_values("expiration").assign(
        years_to_expiration=lambda df: (
            (df["expiration"] - start).dt.total_seconds() / days_per_year / 24 / 60 / 60
        ),
    )
    stats = stats[stats["years_to_expiration"] > 0]
    stats = stats.dropna(subset=["Settlement price"])
    stats = stats[stats["Settlement price"] > 0]
    forward_rate = (100 - stats["Settlement price"]) / 100

    curve_t = []
    curve_r = []
    discount_factor = 1.0
    prior_t = 0.0
    for t, rate in zip(stats["years_to_expiration"], forward_rate, strict=True):
        discount_factor /= 1 + rate * (t - prior_t)
        curve_t.append(t)
        curve_r.append(-np.log(discount_factor) / t)
        prior_t = t

    curve_t_arr = np.array(curve_t)
    curve_r_arr = np.array(curve_r)

    def rate_curve(years_to_expiration: float) -> float:
        return float(np.interp(years_to_expiration, curve_t_arr, curve_r_arr))

    return rate_curve


def get_short_rate_curve(
    client: db.Historical,
    start: pd.Timestamp,
    parent: str = "SR3.FUT",
    days_per_year: float = 365.0,
) -> RateCurve:
    """Build a zero-rate curve from a STIR futures strip's settlement prices.

    Pulls every leg of `parent` (default SR3)
    and its official settlement price as of `start`'s date, converts each
    price to a rate using the `100(1-r)` quoting convention, and bootstraps
    a curve from them. See `build_short_rate_curve` for the bootstrap
    itself. This wrapper does one date fetch. Call
    `build_short_rate_curve` directly to reuse already fetched data.

    Parameters
    ----------
    client : db.Historical
        The Historical client to retrieve databento data.
    start : pd.Timestamp
        The date to build the curve as of; also the years-to-expiration anchor.
    parent : str
        The STIR futures parent symbol, in Databento's `[ROOT].FUT` format.
    days_per_year : float
        The number of days to use to normalize day counts.

    Returns:
    -------
    Callable[[float], float]
        Maps a time to expiration in years to an interpolated zero rate.

    """
    defs = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="definition",
        symbols=parent,
        stype_in="parent",
        start=start.date(),
    ).to_df()
    stats = get_stats_by_date(client, start.date(), defs=defs)
    return build_short_rate_curve(
        stats.reset_index(), start, days_per_year=days_per_year
    )
