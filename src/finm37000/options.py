"""Simple option pricing implementations."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, cast

import databento as db
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares, root_scalar
from scipy.stats import norm

from .lr_tree import calc_american_price


class OptionType(Enum):
    """Enumeration of option types."""

    CALL = "Call"
    PUT = "Put"


def calc_black_scholes(  # noqa: PLR0913
    *,
    S: float | npt.NDArray[np.float64],  # noqa: N803
    K: float | npt.NDArray[np.float64],  # noqa: N803
    T: float | npt.NDArray[np.float64],  # noqa: N803
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    option_type: OptionType,
) -> float | npt.NDArray[np.float64]:
    """Compute Black-Scholes model for European options on equities."""
    f = S * np.exp((r - q) * T)
    d1 = (np.log(f / K) + (vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if option_type == OptionType.CALL else -1
    return discount_factor * cp * (f * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def calc_black_scholes_numerical_rho(  # noqa: PLR0913
    *,
    S: float | npt.NDArray[np.float64],  # noqa: N803
    K: float | npt.NDArray[np.float64],  # noqa: N803
    T: float | npt.NDArray[np.float64],  # noqa: N803
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    q: float | npt.NDArray[np.float64],
    option_type: OptionType,
    dr: float = 0.0001,
) -> float | npt.NDArray[np.float64]:
    """Compute Black-Scholes rho for European options on equities."""
    up_price = calc_black_scholes(
        S=S, K=K, r=r + dr, T=T, q=q, vol=vol, option_type=option_type
    )
    down_price = calc_black_scholes(
        S=S, K=K, r=r - dr, T=T, q=q, vol=vol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dr)


def calc_black(  # noqa: PLR0913
    *,
    F: float | npt.NDArray[np.float64],  # noqa: N803
    K: float | npt.NDArray[np.float64],  # noqa: N803
    T: float | npt.NDArray[np.float64],  # noqa: N803
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    option_type: OptionType,
) -> float | npt.NDArray[np.float64]:
    """Compute Black 76 model for European options on forwards."""
    d1 = (np.log(F / K) + (vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    discount_factor = np.exp(-r * T)

    cp = 1 if option_type == OptionType.CALL else -1
    return discount_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def calc_black_one_day_theta(  # noqa: PLR0913
    *,
    F: float | npt.NDArray[np.float64],  # noqa: N803
    K: float | npt.NDArray[np.float64],  # noqa: N803
    T: float | npt.NDArray[np.float64],  # noqa: N803
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    option_type: OptionType,
    dt: float,
) -> float | npt.NDArray[np.float64]:
    """Compute Black 76 model for European options on forwards."""
    price_dt = calc_black(
        F=F,
        K=K,
        r=r,
        T=T - dt,
        vol=vol,
        option_type=option_type,
    )
    price = calc_black(
        F=F,
        K=K,
        r=r,
        T=T,
        vol=vol,
        option_type=option_type,
    )
    return price - price_dt


def calc_black_numerical_theta(  # noqa: PLR0913
    *,
    F: float | npt.NDArray[np.float64],  # noqa: N803
    K: float | npt.NDArray[np.float64],  # noqa: N803
    T: float | npt.NDArray[np.float64],  # noqa: N803
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    option_type: OptionType,
    dt: float,
) -> float | npt.NDArray[np.float64]:
    """Use Black model to numerically compute theta."""
    return (
        calc_black_one_day_theta(
            F=F,
            K=K,
            T=T,
            vol=vol,
            r=r,
            option_type=option_type,
            dt=dt,
        )
        / dt
    )


def calc_numerical_delta(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dfuture: float,
) -> npt.NDArray[np.float64]:
    """Compute non-skew adjusted American option deltas."""
    up_price = calc_american_price(
        future=future + dfuture,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    down_price = calc_american_price(
        future=future - dfuture,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    return (up_price - down_price) / (2.0 * dfuture)


def calc_one_day_theta(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dt: float = 1 / 365,
) -> npt.NDArray[np.float64]:
    """Compute American option one-day time decay."""
    t_1 = np.maximum(t - dt, 0.0)
    price_dt = calc_american_price(
        future=future,
        strike=strike,
        r=r,
        t=t_1,
        vol=vol,
        option_type=option_type,
    )
    price = calc_american_price(
        future=future,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
    )
    return price - price_dt


def calc_numerical_theta(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dt: float = 1 / 365,
) -> npt.NDArray[np.float64]:
    """Compute American option thetas."""
    return (
        calc_one_day_theta(
            future=future,
            strike=strike,
            r=r,
            t=t,
            vol=vol,
            option_type=option_type,
            dt=dt,
        )
        / dt
    )


def calc_numerical_vega(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dvol: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Compute American option vegas."""
    up_price = calc_american_price(
        future=future, strike=strike, r=r, t=t, vol=vol + dvol, option_type=option_type
    )
    down_price = calc_american_price(
        future=future, strike=strike, r=r, t=t, vol=vol - dvol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dvol)


def calc_numerical_rho(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dr: float = 0.0001,
) -> npt.NDArray[np.float64]:
    """Compute American option rho."""
    up_price = calc_american_price(
        future=future, strike=strike, r=r + dr, t=t, vol=vol, option_type=option_type
    )
    down_price = calc_american_price(
        future=future, strike=strike, r=r - dr, t=t, vol=vol, option_type=option_type
    )
    return (up_price - down_price) / (2.0 * dr)


def calc_american_greeks(  # noqa: PLR0913
    *,
    future: float | npt.NDArray[np.float64],
    strike: float | npt.NDArray[np.float64],
    t: float | npt.NDArray[np.float64],
    vol: float | npt.NDArray[np.float64],
    r: float | npt.NDArray[np.float64],
    option_type: npt.ArrayLike,
    dfuture: float = 0.01,
    dvol: float = 0.01,
    dr: float = 0.0001,
    dt: float = 1 / 365,
) -> npt.NDArray[np.float64]:
    """Compute American price and greeks on futures via the Leisen-Reimer tree.

    Bundles `calc_american_price` with the `calc_numerical_delta`/
    `calc_numerical_vega`/`calc_numerical_theta`/`calc_numerical_rho`
    finite-difference greeks into one `(NPV, delta, vega, theta, rho)`
    array. Roughly 9 tree
    evaluations per option (1 for NPV, 2 for each of the 4 greeks), so this
    is fine for a handful of options but gets slow on a large chain.

    Returns:
    -------
    numpy.ndarray of shape (n_options, 5): columns are
    (NPV, delta, vega, theta, rho).

    """
    npv = calc_american_price(
        future=future, strike=strike, t=t, vol=vol, r=r, option_type=option_type
    )
    delta = calc_numerical_delta(
        future=future,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
        dfuture=dfuture,
    )
    vega = calc_numerical_vega(
        future=future,
        strike=strike,
        r=r,
        t=t,
        vol=vol,
        option_type=option_type,
        dvol=dvol,
    )
    theta = calc_numerical_theta(
        future=future, strike=strike, r=r, t=t, vol=vol, option_type=option_type, dt=dt
    )
    rho = calc_numerical_rho(
        future=future, strike=strike, r=r, t=t, vol=vol, option_type=option_type, dr=dr
    )
    return np.column_stack(
        [np.ravel(npv), np.ravel(delta), np.ravel(vega), np.ravel(theta), np.ravel(rho)]
    )


def get_options_chain(
    parent: str,
    start: pd.Timestamp,
    client: db.Historical,
    underlying: Optional[str],
    days_per_year: float = 365.0,
    instrument_class: Optional[Sequence[str]] = ("C", "P"),
) -> pd.DataFrame:
    """Retrieve the definitions of shared-parent options contracts.

    Parameters
    ----------
    parent : str
        The parent symbol, such as ES.
    start : pd.Timestamp
        The date to obtain the definitions for.
    client: db.Historical
        The Historical client to retrieve databento data.
    underlying : str
        The underlying contract for the option. Default of None
        will return options with any underlying. Note that spreads
        do not come with the `"underlying"` column populated, so
        spreads require this argument to be None.
    days_per_year: float
        The number of days to use to normalize day counts.
    instrument_class: Sequence[str]
        Filter for which instrument type to return. Calls (`"C"`),
        puts (`"P"`), and others such as spreads (`"T"`). Default
        of None will return all definitions.

    Returns:
    -------
    pd.DataFrame

    """
    options_def = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="definition",
        symbols=f"{parent}.OPT",
        stype_in="parent",
        start=start.date(),
    )

    df = options_def.to_df()
    if underlying is not None:
        df = df[df["underlying"] == underlying]
    if instrument_class is not None:
        df = df[df["instrument_class"].isin(instrument_class)]
    df["years_to_expiration"] = (
        (df["expiration"] - start).dt.total_seconds() / days_per_year / 24 / 60 / 60
    )
    return df.sort_values("strike_price")


def get_top_of_book(
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    client: db.Historical,
) -> pd.DataFrame:
    """Get the last top-of-book and calculate midprice.

    Parameters
    ----------
    symbols : Iterable[str]
        A collection of symbols to retrieve the midprices for.
    start : pd.Timestamp
        The start time.
    end : pd.Timestamp
        The end time (exclusive).
    client: db.Historical
        The Historical client to retrieve databento data.

    Returns:
    -------
    pd.DataFrame

    """
    price_df = client.timeseries.get_range(
        dataset=db.Dataset.GLBX_MDP3,
        schema="mbp-1",
        symbols=symbols,
        start=start,
        end=end,
    ).to_df()

    price_df = price_df.groupby("symbol").last()
    price_df["bid"] = price_df["bid_px_00"]
    price_df["ask"] = price_df["ask_px_00"]
    price_df["bidq"] = price_df["bid_sz_00"]
    price_df["askq"] = price_df["ask_sz_00"]
    price_df["midprice"] = np.mean(price_df[["bid", "ask"]], axis=1)
    wt = price_df["bidq"] / (price_df["bidq"] + price_df["askq"])
    price_df["weighted_midprice"] = price_df["bid"] * (1.0 - wt) + price_df["ask"] * wt

    cols = ["bid", "ask", "midprice", "bidq", "askq", "weighted_midprice"]
    return price_df[cols]


def imply_european_vol(
    row: pd.Series,
    price_col: str = "midprice",
) -> float:
    """Find the roots of the Black-76 model by varying sigma, implied volatility.

    This function is for use with `pandas.Dataframe.apply`. Each row should contain
    a column for "strike_price", "years_to_expiration", "instrument_class", "midprice",
    and "underlying_price",

    If the optimization fails, `numpy.nan` is returned.

    Parameters
    ----------
    row : pd.Series
        A series of data to process.
    price_col : str, optional
        The name of the column in `row` that contains the price.

    Returns:
    -------
    float | numpy.nan

    """
    target = float(row[price_col])
    option_type = OptionType.CALL if row["instrument_class"] == "C" else OptionType.PUT

    def model_price(vol: float) -> float:
        return float(
            calc_black(
                F=row["underlying_price"],
                K=row["strike_price"],
                T=row["years_to_expiration"],
                vol=vol,
                r=row["interest_rate"],
                option_type=option_type,
            )
        )

    def f(vol: float) -> float:
        return target - model_price(vol)

    lb = 0.00001
    ub = 4
    f_lb = f(lb)
    f_ub = f(ub)
    if f_ub * f_lb >= 0:
        lower_vol = model_price(lb)
        upper_vol = model_price(ub)
        print(
            (
                f"Cannot find {option_type} vol between "
                f"{lb=} and {ub=} "
                f"at strike {row['strike_price']}: "
                f"{lower_vol=} {target=} {upper_vol=}"
            )
        )
        print(
            (
                f"  F={row['underlying_price']} "
                f"T={row['years_to_expiration']} "
                f"r={row['interest_rate']} "
                f"mid={row['midprice']}"
            )
        )
        return np.nan
    result = root_scalar(f, bracket=[lb, ub], method="brentq")
    if result.converged:
        return result.root
    print(
        f"Could not find sigma for {row['raw_symbol']} with midprice {row['midprice']}",
    )
    return np.nan


@dataclass
class RateFitInputs:
    """Per-expiration strikes/prices to fit a shared rate via `imply_american_rate`."""

    strikes: npt.NDArray[np.float64]
    call_prices: npt.NDArray[np.float64]
    put_prices: npt.NDArray[np.float64]
    futures_price: float
    years_to_expiration: float


def imply_american_rate(  # noqa: PLR0913
    *,
    strikes: npt.NDArray[np.float64],
    call_prices: npt.NDArray[np.float64],
    put_prices: npt.NDArray[np.float64],
    futures_price: float,
    years_to_expiration: float,
    initial_rate: float = 0.04,
    initial_vol: float = 0.2,
    min_rate: float = 0.0,
    max_rate: float = 0.25,
    min_vol: float = 1e-4,
    max_vol: float = 4.0,
) -> tuple[float, npt.NDArray[np.float64]]:
    """Infer one shared rate and per-strike vols from American call/put prices.

    Joint least-squares fit over a single expiration to infer one
    `vol` per strike and one shared `rate`.

    Pass strikes you consider a reasonable sample for the fit.

    The default bounds do not search below `min_rate=0.0`, since negative
    short-term rates are unusual enough that they're worth an explicit,
    deliberate choice by the caller rather than a silent default.

    Parameters
    ----------
    strikes : array of strike prices.
    call_prices : market call prices, aligned with `strikes`.
    put_prices : market put prices, aligned with `strikes`.
    futures_price : the underlying futures price.
    years_to_expiration : the time to expiration.
    initial_rate, initial_vol : starting guesses for the optimizer.
    min_rate, max_rate, min_vol, max_vol : bounds for the optimizer.

    Raises:
    ------
    ValueError
        If `strikes` is empty. With no strikes there is nothing to fit against,
        so the optimizer has zero residuals and trivially "succeeds" at
        whatever `initial_rate`/`initial_vol` it started from -- silently
        returning the starting guess back as if it were a real fit. A single
        strike is fine (exactly determined: one rate, one vol, a call and a
        put price to fit them to).

    Returns:
    -------
    tuple[float, np.ndarray]
        The fitted rate, and the fitted per-strike vols (aligned with `strikes`).

    """
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    put_prices = np.asarray(put_prices, dtype=float)
    n = len(strikes)
    if n == 0:
        msg = "Need at least 1 strike with a call and a put price to fit a rate."
        raise ValueError(msg)

    def residuals(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        rate = params[0]
        vols = params[1:]
        call_model = calc_american_price(
            future=futures_price,
            strike=strikes,
            t=years_to_expiration,
            vol=vols,
            r=rate,
            option_type="C",
        )
        put_model = calc_american_price(
            future=futures_price,
            strike=strikes,
            t=years_to_expiration,
            vol=vols,
            r=rate,
            option_type="P",
        )
        return cast(
            "npt.NDArray[np.float64]",
            np.concatenate([call_model - call_prices, put_model - put_prices]),
        )

    x0 = np.concatenate([[initial_rate], np.full(n, initial_vol)])
    lower = np.concatenate([[min_rate], np.full(n, min_vol)])
    upper = np.concatenate([[max_rate], np.full(n, max_vol)])

    result = least_squares(residuals, x0, bounds=(lower, upper))
    return float(result.x[0]), result.x[1:]
