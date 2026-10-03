"""A Leisen-Reimer binomial tree pricer for American options on futures."""

import numpy as np
import numpy.typing as npt


def _peizer_pratt_inversion(
    z: npt.NDArray[np.float64], n: int
) -> npt.NDArray[np.float64]:
    """Peizer-Pratt method-2 inversion used to build the Leisen-Reimer tree.

    Maps a Black-Scholes d1/d2-style z-score to a risk-neutral probability
    such that the resulting binomial tree's skewness/kurtosis matches the
    lognormal distribution to a higher order than Cox-Ross-Rubinstein.
    """
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    exponent = -((z / denom) ** 2) * (n + 1.0 / 6.0)
    return 0.5 + np.sign(z) * np.sqrt(0.25 - 0.25 * np.exp(exponent))


def calc_american_price(  # noqa: PLR0913, PLR0915
    *,
    future: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    vol: npt.ArrayLike,
    r: npt.ArrayLike,
    option_type: npt.ArrayLike,
    n_steps: int = 201,
) -> npt.NDArray[np.float64]:
    """Price American options on futures with a Leisen-Reimer binomial tree.

    If t is exactly 0.0, exercise value will be returned.

    Parameters
    ----------
    future, strike, t, vol, r : float or array-like, broadcastable together.
    option_type : "C" or "P", broadcastable with the above.
    n_steps : int
        Number of tree steps. Leisen-Reimer requires an odd step count;
        an even value is bumped up by one.

    Returns:
    -------
    numpy.ndarray matching the broadcast shape of the inputs (0-d for
    fully scalar inputs).

    """
    n = n_steps if n_steps % 2 == 1 else n_steps + 1

    future_, strike_, t_, vol_, r_, opt_ = np.broadcast_arrays(
        future, strike, t, vol, r, option_type
    )
    shape = future_.shape
    future_ = future_.ravel().astype(float)
    strike_ = strike_.ravel().astype(float)
    t_ = t_.ravel().astype(float)
    vol_ = vol_.ravel().astype(float)
    r_ = r_.ravel().astype(float)
    opt_ = opt_.ravel()

    is_call = opt_ == "C"
    is_put = opt_ == "P"
    if not np.all(is_call | is_put):
        bad = np.unique(opt_[~(is_call | is_put)])
        msg = f"Invalid option type(s): {bad.tolist()}"
        raise ValueError(msg)
    cp = np.where(is_call, 1.0, -1.0)

    out = np.full(future_.shape[0], np.nan)
    valid = ~np.isnan(vol_)
    if not valid.any():
        return out.reshape(shape)

    f = future_[valid]
    k = strike_[valid]
    tt = t_[valid]
    v = vol_[valid]
    rr = r_[valid]
    c = cp[valid]

    vals = np.maximum(c * (f - k), 0.0)
    expired = tt == 0.0
    if expired.all():
        out[valid] = vals
        return out.reshape(shape)
    if expired.any():
        f, k, tt, v, rr, c = (a[~expired] for a in (f, k, tt, v, rr, c))

    dt = tt / n
    sqrt_t = np.sqrt(tt)
    d1 = (np.log(f / k) + (v**2 / 2.0) * tt) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    p = _peizer_pratt_inversion(d2, n)
    p_prime = _peizer_pratt_inversion(d1, n)

    # At extreme |d1|/|d2| (very low vol relative to moneyness/maturity/step
    # count), p and p_prime saturate to exactly 0.0 or 1.0 in float64, which
    # makes u or d a 0/0.
    degenerate = (p <= 0.0) | (p >= 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(degenerate, 1.0, p_prime / p)
        d = np.where(degenerate, 1.0, (1.0 - p_prime) / (1.0 - p))
    disc = np.exp(-rr * dt)

    j = np.arange(n + 1)
    spot = f[:, None] * (u[:, None] ** j[None, :]) * (d[:, None] ** (n - j)[None, :])
    values = np.maximum(c[:, None] * (spot - k[:, None]), 0.0)

    for i in range(n - 1, -1, -1):
        up = values[:, 1 : i + 2]
        down = values[:, 0 : i + 1]
        continuation = disc[:, None] * (p[:, None] * up + (1.0 - p[:, None]) * down)
        spot = spot[:, : i + 1] / d[:, None]
        exercise = np.maximum(c[:, None] * (spot - k[:, None]), 0.0)
        values = np.maximum(continuation, exercise)

    if expired.any():
        vals[~expired] = values[:, 0]
    else:
        vals = values[:, 0]
    out[valid] = vals
    return out.reshape(shape)


def imply_american_vol(  # noqa: PLR0913
    *,
    future: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    price: npt.ArrayLike,
    r: npt.ArrayLike,
    option_type: npt.ArrayLike,
    min_vol: float = 1e-4,
    max_vol: float = 4.0,
    tol: float = 1e-6,
    max_iter: int = 60,
    n_steps: int = 201,
) -> npt.NDArray[np.float64]:
    """Solve implied vol from American prices via bisection on the LR tree.

    Broadcasts `future`/`strike`/`t`/`price`/`r`/`option_type` together and
    solves every element in the same pass. Each bisection iteration prices
    through `calc_american_price` in one vectorized call.

    Returns:
    -------
    numpy.ndarray of implied vols, matching the broadcast shape (0-d for
    fully scalar inputs).

    """
    future_, strike_, t_, price_, r_, opt_ = np.broadcast_arrays(
        future, strike, t, price, r, option_type
    )
    shape = future_.shape
    future_ = future_.ravel().astype(float)
    strike_ = strike_.ravel().astype(float)
    t_ = t_.ravel().astype(float)
    price_ = price_.ravel().astype(float)
    r_ = r_.ravel().astype(float)
    opt_ = opt_.ravel()

    lo = np.full(future_.shape[0], min_vol)
    hi = np.full(future_.shape[0], max_vol)
    price_lo = calc_american_price(
        future=future_,
        strike=strike_,
        t=t_,
        vol=lo,
        r=r_,
        option_type=opt_,
        n_steps=n_steps,
    )
    price_hi = calc_american_price(
        future=future_,
        strike=strike_,
        t=t_,
        vol=hi,
        r=r_,
        option_type=opt_,
        n_steps=n_steps,
    )
    bracketed = (price_ >= price_lo) & (price_ <= price_hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = calc_american_price(
            future=future_,
            strike=strike_,
            t=t_,
            vol=mid,
            r=r_,
            option_type=opt_,
            n_steps=n_steps,
        )
        too_high = price_mid > price_
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.all(hi - lo < tol):
            break

    result = np.where(bracketed, 0.5 * (lo + hi), np.nan)
    return result.reshape(shape)
