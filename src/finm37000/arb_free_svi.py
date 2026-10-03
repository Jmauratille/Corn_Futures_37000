"""Arbitrage-constrained raw SVI fit, with analytic gradients for speed."""

from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from .skew import RawSVIParams, calc_raw_svi, filter_valid


def svi_butterfly(p: np.ndarray) -> np.ndarray:
    """SVI butterfly arbitrage constraints (all must be >= 0)."""
    a, b, rho, m, sigma = p
    c1 = (a - m * b * (rho + 1)) * (4 - a + m * b * (rho + 1)) - (b**2) * (rho + 1) ** 2
    c2 = (a - m * b * (rho - 1)) * (4 - a + m * b * (rho - 1)) - (b**2) * (rho - 1) ** 2
    c3 = 4 - (b**2) * (rho + 1) ** 2
    c4 = 4 - (b**2) * (rho - 1) ** 2
    return np.array([c1, c2, c3, c4])


def svi_butterfly_jac(p: np.ndarray) -> np.ndarray:
    """Jacobian of `svi_butterfly`, shape (4, 5)."""
    a, b, rho, m, sigma = p  # noqa: F841
    dc1da = -2 * a + 2 * b * m * (rho + 1) + 4
    dc1db = (
        -2 * b * (rho + 1) ** 2
        + m * (a - b * m * (rho + 1)) * (rho + 1)
        - m * (rho + 1) * (-a + b * m * (rho + 1) + 4)
    )
    dc1drho = (
        -(b**2) * (2 * rho + 2)
        + b * m * (a - b * m * (rho + 1))
        - b * m * (-a + b * m * (rho + 1) + 4)
    )
    dc1dm = b * (a - b * m * (rho + 1)) * (rho + 1) - b * (rho + 1) * (
        -a + b * m * (rho + 1) + 4
    )
    dc2da = -2 * a + 2 * b * m * (rho - 1) + 4
    dc2db = (
        -2 * b * (rho - 1) ** 2
        + m * (a - b * m * (rho - 1)) * (rho - 1)
        - m * (rho - 1) * (-a + b * m * (rho - 1) + 4)
    )
    dc2drho = (
        -(b**2) * (2 * rho - 2)
        + b * m * (a - b * m * (rho - 1))
        - b * m * (-a + b * m * (rho - 1) + 4)
    )
    dc2dm = b * (a - b * m * (rho - 1)) * (rho - 1) - b * (rho - 1) * (
        -a + b * m * (rho - 1) + 4
    )
    dc3db = -2 * b * (rho + 1) ** 2
    dc3drho = -(b**2) * (2 * rho + 2)
    dc4db = -2 * b * (rho - 1) ** 2
    dc4drho = -(b**2) * (2 * rho - 2)
    z = 0.0
    return np.array(
        [
            [dc1da, dc1db, dc1drho, dc1dm, z],
            [dc2da, dc2db, dc2drho, dc2dm, z],
            [z, dc3db, dc3drho, z, z],
            [z, dc4db, dc4drho, z, z],
        ]
    )


def calc_raw_svi_jac(  # noqa: PLR0913
    k: npt.NDArray[np.float64],
    a: float,  # noqa: ARG001
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> npt.NDArray[np.float64]:
    """Jacobian of `calc_raw_svi` w.r.t. (a, b, rho, m, sigma), shape (len(k), 5)."""
    dsda = np.ones_like(k)
    dsdb = rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2)
    dsdrho = b * (k - m)
    dsdm = b * (-rho + (m - k) / np.sqrt(sigma**2 + (k - m) ** 2))
    dsdsigma = b * sigma / np.sqrt(sigma**2 + (k - m) ** 2)
    return np.array([dsda, dsdb, dsdrho, dsdm, dsdsigma]).T


def fit_arb_free_svi(k: np.ndarray, w: np.ndarray) -> RawSVIParams:
    """Fit raw SVI curve under no-butterfly-arbitrage constraints."""
    k, w = filter_valid(k, w)

    def svi_mse(params: np.ndarray, k: np.ndarray, w: np.ndarray) -> float:
        return float(((calc_raw_svi(k, *params) - w) ** 2).sum())

    def svi_mse_jac(params: np.ndarray, k: np.ndarray, w: np.ndarray) -> np.ndarray:
        resid = calc_raw_svi(k, *params) - w
        jac = calc_raw_svi_jac(k, *params)
        return cast("np.ndarray", 2 * (resid @ jac))

    tol = 1e-6  # prevent rho from reaching +/-1
    lower = [-np.inf, 0, -1 + tol, -np.inf, 0]
    upper = [np.inf, np.inf, 1 - tol, np.inf, np.inf]
    bounds = Bounds(lower, upper)
    inequalities = [
        {"type": "ineq", "fun": svi_butterfly, "jac": svi_butterfly_jac},
    ]
    p0 = np.asarray(RawSVIParams())
    res = minimize(
        svi_mse,
        p0,
        args=(k, w),
        jac=svi_mse_jac,
        constraints=inequalities,  # type: ignore[arg-type]
        bounds=bounds,
    )
    return RawSVIParams.from_array(res.x)
