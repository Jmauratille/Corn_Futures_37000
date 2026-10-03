"""Functions to fit basic skews."""

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from .lr_tree import imply_american_vol
from .options import RateFitInputs, imply_european_vol

logger = logging.getLogger(__name__)


def filter_valid(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Keep pairs that are neither NaN."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def calculate_option_vols(
    top_df: pd.DataFrame,
    underlying_symbol: str,
    option_chain: pd.DataFrame,
    interest_rate: float,
) -> tuple[pd.DataFrame, float]:
    """Convenience function doing multiple implied vol calculations."""
    underlying_price = top_df["midprice"].loc[underlying_symbol]
    with_top_prices = option_chain.join(top_df, on="symbol").dropna(subset="midprice")
    with_top_prices["underlying_price"] = underlying_price
    with_top_prices["interest_rate"] = interest_rate
    with_vols = with_top_prices.assign(
        **{
            f"iv_{col}": imply_american_vol(
                future=underlying_price,
                strike=with_top_prices["strike_price"],
                t=with_top_prices["years_to_expiration"],
                price=with_top_prices[col],
                r=interest_rate,
                option_type=with_top_prices["instrument_class"],
            )
            for col in ("bid", "midprice", "ask", "weighted_midprice")
        }
    )
    with_vols["european_vol"] = with_vols.apply(
        imply_european_vol,
        axis=1,
    )
    return with_vols, underlying_price


def select_rate_inputs(
    top_df: pd.DataFrame,
    underlying_symbol: str,
    option_chain: pd.DataFrame,
    price_col: str = "midprice",
    n_strikes: int = 5,
) -> RateFitInputs:
    """Select near-the-money strikes/prices for a single-expiration rate fit.

    Usage:

        inputs = select_rate_inputs(top_df, underlying_symbol, option_chain)
        rate, vols = imply_american_rate(
            strikes=inputs.strikes,
            call_prices=inputs.call_prices,
            put_prices=inputs.put_prices,
            futures_price=inputs.futures_price,
            years_to_expiration=inputs.years_to_expiration,
        )

    Joins `option_chain` to `top_df` the same way `calculate_option_vols` does,
    keeps only strikes with a valid price on both the call and the put side,
    and returns the `n_strikes` closest to the futures price by absolute
    moneyness.

    The `option_chain` should already be filtered down to one expiration, since
    the returned inputs share a single time to expiration.

    Raises:
    ------
    ValueError
        If `n_strikes` is less than 1, or if fewer than `n_strikes` strikes
        have a valid price on both the call and put side.
    """
    if n_strikes < 1:
        msg = f"n_strikes must be at least 1; got {n_strikes}."
        raise ValueError(msg)

    underlying_price = top_df["midprice"].loc[underlying_symbol]
    with_top_prices = option_chain.join(top_df, on="symbol").dropna(subset=price_col)
    call, put = split_call_put(with_top_prices)
    paired = call.merge(put, on="strike_price", suffixes=("_call", "_put"))
    min_strikes = max(1, n_strikes)
    if len(paired) < min_strikes:
        msg = (
            f"Need at least {min_strikes} strikes with a valid price on both "
            f"the call and put side to fit a rate; found {len(paired)}."
        )
        raise ValueError(msg)

    moneyness = (paired["strike_price"] - underlying_price).abs()
    nearest = paired.loc[moneyness.sort_values().index[:n_strikes]]

    return RateFitInputs(
        strikes=nearest["strike_price"].to_numpy(),
        call_prices=nearest[f"{price_col}_call"].to_numpy(),
        put_prices=nearest[f"{price_col}_put"].to_numpy(),
        futures_price=underlying_price,
        years_to_expiration=nearest["years_to_expiration_call"].iloc[0],
    )


def split_call_put(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into the call and put rows."""
    call = df[df["instrument_class"] == "C"]
    put = df[df["instrument_class"] == "P"]
    return call, put


def filter_otm(df: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
    """Extract OTM rows from the options dataframe."""
    call, put = split_call_put(df)
    otm = pd.concat(
        [
            call[call["strike_price"] >= underlying_price],
            put[put["strike_price"] <= underlying_price],
        ]
    ).sort_values(by="strike_price")
    return otm


def fit_polynomial_skew(k: np.ndarray, sigma: np.ndarray, degree: int) -> np.poly1d:
    """Fit a polynomial skew curve to volatility."""
    k, sigma = filter_valid(k, sigma)
    coefficients = np.polyfit(k, sigma, degree)
    return np.poly1d(coefficients)


def fit_weighted_piecewise_polynomial_skew(
    k: np.ndarray, sigma: np.ndarray, atm: float, degree: int, atm_weight: float = 1e6
) -> Callable[[pd.Series | np.ndarray], np.ndarray]:
    """Fit a piecewise polynomial skew-like function.

    Join pieces at-the-money by using extra weight in the regression.
    """
    k, sigma = filter_valid(k, sigma)
    put_k = k[k < atm]
    put_wt = np.ones_like(put_k, dtype=float)
    put_wt[-1] = atm_weight
    put_skew_coef = np.polyfit(put_k, sigma[k < atm], degree, w=put_wt)
    call_k = k[k >= atm]
    call_wt = np.ones_like(call_k, dtype=float)
    call_wt[0] = atm_weight
    call_skew_coef = np.polyfit(call_k, sigma[k >= atm], degree, w=call_wt)
    put_skew = np.poly1d(put_skew_coef)
    call_skew = np.poly1d(call_skew_coef)

    def piecewise_polynomial_skew(x: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        return np.piecewise(x, [x < atm, x >= atm], [put_skew, call_skew])

    return piecewise_polynomial_skew


def fit_spline_skew(
    k: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    percent: float = 0.1,
    bc_type: str = "clamped",
    extrapolate: bool = False,
) -> CubicSpline:
    """Fit a basic cubic spline.

    Fit a cubic spline to the implied volatilities ``sigma``
    at strikes ``k``. Among the non-NA values, downsample roughly ``percent``
    of the strikes to pass through, equally spaced by the index
    not by strike. ``percent=1.0`` selects all
    the strikes; the default selects ten percent of strikes.

    The ``bc_type`` and ``extrapolate`` arguments are passed through
    to ``scipy.interpolate.CubicSpline`` constructor.
    """
    if not (0.0 < percent <= 1.0):
        msg = f"downsampling percentage {percent} must be between 0 and 1"
        raise ValueError(msg)
    fifty_percent = 0.5
    if fifty_percent < percent < 1.0:
        msg = (
            "Downsampling percentages between 0.5 and 1.0 select half or all points,"
            f" received {percent}."
        )
        warnings.warn(msg, stacklevel=2)
    k, sigma = filter_valid(k, sigma)
    step_size = max(1, int(1 / percent))
    spline = CubicSpline(
        k[::step_size],
        sigma[::step_size],
        bc_type=bc_type,
        extrapolate=extrapolate,
    )  # type: ignore[call-overload]
    return cast("CubicSpline", spline)


def calc_raw_svi(  # noqa: PLR0913
    k: npt.NDArray[np.float64],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """The raw SVI curve.

    Positive variance guaranteed when a + b * sigma * sqrt(1-rho^2) >= 0.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


@dataclass
class RawSVIParams:
    """Parameters for raw SVI curve."""

    a: float = 0.1  # real:  vertical level of the skew
    b: float = 0.1  # nonnegative: higher b raises both wings
    rho: float = 0  # |rho| < 1: Lower rho raises the puts relative to the calls.
    m: float = 0  # real: higher m moves to the right
    sigma: float = 0.1  # positive: Higher sigma reduces ATM curvature.

    def __array__(self) -> np.ndarray:
        """Cast RawSVIParams to a numpy array."""
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray) -> Self:
        """Convert numpy array to RawSVIParams."""
        return cls(
            a=float(x[0]),
            b=float(x[1]),
            rho=float(x[2]),
            m=float(x[3]),
            sigma=float(x[4]),
        )

    def calc(self, k: np.ndarray) -> float | np.ndarray:
        """Evaluate the raw svi curve at the given log-strikes."""
        return calc_raw_svi(k, self.a, self.b, self.rho, self.m, self.sigma)


def fit_raw_svi(k: np.ndarray, w: np.ndarray) -> RawSVIParams:
    """Fit raw SVI curve to given log-strikes."""
    k, w = filter_valid(k, w)
    tol = 1e-6  # prevent rho from reaching +/-1
    lower: npt.NDArray[np.float64] = np.array(
        [-np.inf, 0, -1 + tol, -np.inf, 0], dtype=float
    )
    upper: npt.NDArray[np.float64] = np.array(
        [np.inf, np.inf, 1 - tol, np.inf, np.inf], dtype=float
    )
    bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (lower, upper)
    p0 = np.asarray(RawSVIParams())
    fit, _ = curve_fit(
        f=calc_raw_svi,
        xdata=k,
        ydata=w,
        p0=p0,
        bounds=bounds,
    )
    return RawSVIParams.from_array(fit)


def calc_call_price_implied_density(K: np.ndarray, C: np.ndarray) -> np.ndarray:  # noqa: N803
    """Calculate implied density.

    If assumptions are violated, it won't be a density.
    Need no butterfly arb, no calendar arb.
    """
    q_bar = -np.diff(C, prepend=-1 + C[0]) / np.diff(K, prepend=1 + K[0])
    if np.any(q_bar < 0) or np.any(q_bar > 1):
        n = ((q_bar < 0) | (q_bar > 1)).sum()
        logger.warning("%d call spreads are outside of [0, 1]", n)
    q = -np.diff(q_bar, append=np.nan)
    return q
