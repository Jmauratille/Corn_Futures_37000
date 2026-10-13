"""Trading-signal features for futures bar and trade data.

These utilities turn OHLCV-style bar data and raw trade data into the
engineered features used downstream for return prediction and strategy
research: moving averages, Bollinger bands, multi-horizon returns,
realized volatility, and order-flow imbalance.

All functions are pure: they accept ``pandas`` objects and return new
``pandas`` objects without mutating their inputs, so they compose freely
and can be unit tested on synthetic data without any Databento access.
"""

import numpy as np
import pandas as pd


def _check_positive(name: str, value: int) -> None:
    """Raise ``ValueError`` unless ``value`` is a positive integer."""
    if value < 1:
        msg = f"{name} must be a positive integer, got {value}."
        raise ValueError(msg)


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Rolling simple moving average of a price series.

    Args:
        prices: Price series indexed by time.
        window: Number of observations in the rolling window.

    Returns:
        The rolling mean aligned to ``prices``. The first ``window - 1``
        entries are ``NaN``.

    Raises:
        ValueError: If ``window`` is not a positive integer.

    Examples:
        >>> import pandas as pd
        >>> simple_moving_average(pd.Series([1.0, 2.0, 3.0]), 2).tolist()
        [nan, 1.5, 2.5]

    """
    _check_positive("window", window)
    result: pd.Series = prices.rolling(window).mean()
    return result


def exponential_moving_average(prices: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted moving average of a price series.

    Uses ``adjust=False`` so the average is a simple recursion seeded with
    the first observation, which is the usual convention for a streaming
    trading indicator.

    Args:
        prices: Price series indexed by time.
        span: Span of the exponential weighting; larger is smoother.

    Returns:
        The exponentially weighted mean aligned to ``prices``.

    Raises:
        ValueError: If ``span`` is not a positive integer.

    Examples:
        >>> import pandas as pd
        >>> exponential_moving_average(pd.Series([1.0, 2.0]), 1).tolist()
        [1.0, 2.0]

    """
    _check_positive("span", span)
    result: pd.Series = prices.ewm(span=span, adjust=False).mean()
    return result


def bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger bands around a simple moving average.

    Args:
        prices: Price series indexed by time.
        window: Rolling window for the moving average and standard deviation.
        num_std: Number of standard deviations for the upper and lower bands.

    Returns:
        A ``DataFrame`` indexed like ``prices`` with columns:

        - ``middle``: the ``window`` simple moving average.
        - ``upper``: ``middle + num_std * rolling_std``.
        - ``lower``: ``middle - num_std * rolling_std``.
        - ``bandwidth``: ``(upper - lower) / middle``.
        - ``pct_b``: price position within the bands,
          ``(price - lower) / (upper - lower)``.

    Raises:
        ValueError: If ``window`` is not positive or ``num_std`` is negative.

    """
    _check_positive("window", window)
    if num_std < 0:
        msg = f"num_std must be non-negative, got {num_std}."
        raise ValueError(msg)
    middle = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    width = upper - lower
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": width / middle,
            "pct_b": (prices - lower) / width,
        }
    )


def returns(prices: pd.Series, horizon: int = 1, *, log: bool = True) -> pd.Series:
    """Returns over a fixed horizon.

    Args:
        prices: Price series indexed by time.
        horizon: Number of periods between the two prices compared.
        log: If ``True`` return log returns ``log(p_t / p_{t-h})``;
            otherwise simple returns ``p_t / p_{t-h} - 1``.

    Returns:
        The horizon returns aligned to ``prices``. The first ``horizon``
        entries are ``NaN``.

    Raises:
        ValueError: If ``horizon`` is not a positive integer.

    """
    _check_positive("horizon", horizon)
    if log:
        log_ret: pd.Series = np.log(prices).diff(horizon)
        return log_ret
    simple_ret: pd.Series = prices.pct_change(horizon)
    return simple_ret


def realized_volatility(
    prices: pd.Series,
    window: int,
    periods_per_year: int | None = None,
) -> pd.Series:
    """Rolling realized volatility from one-period log returns.

    Computes the rolling standard deviation of one-period log returns. When
    ``periods_per_year`` is given, the result is annualized by multiplying
    by ``sqrt(periods_per_year)`` (e.g. ``252`` for daily bars).

    Args:
        prices: Price series indexed by time.
        window: Rolling window, in periods, over which to measure volatility.
        periods_per_year: Periods per year used to annualize. When ``None``
            the raw per-period volatility is returned.

    Returns:
        The rolling (optionally annualized) volatility aligned to ``prices``.

    Raises:
        ValueError: If ``window`` or ``periods_per_year`` is not positive.

    """
    _check_positive("window", window)
    log_ret = returns(prices, horizon=1, log=True)
    vol: pd.Series = log_ret.rolling(window).std()
    if periods_per_year is not None:
        _check_positive("periods_per_year", periods_per_year)
        vol = vol * np.sqrt(periods_per_year)
    return vol


def signed_volume(
    trades: pd.DataFrame,
    side_col: str = "side",
    size_col: str = "size",
    buy_side: str = "B",
    sell_side: str = "A",
) -> pd.Series:
    """Per-trade signed volume derived from the aggressor side.

    Databento records the aggressor side of each trade: ``"B"`` (Bid) for
    buyer-initiated trades, ``"A"`` (Ask) for seller-initiated trades, and
    ``"N"`` where no side is recorded. Buyer-initiated size is counted as
    positive, seller-initiated size as negative, and anything else as zero.

    Args:
        trades: Trade records containing ``side_col`` and ``size_col``.
        side_col: Name of the aggressor-side column.
        size_col: Name of the trade-size column.
        buy_side: Side label treated as buyer-initiated.
        sell_side: Side label treated as seller-initiated.

    Returns:
        Signed trade sizes aligned to ``trades``.

    Raises:
        KeyError: If ``side_col`` or ``size_col`` is not in ``trades``.

    """
    missing = {side_col, size_col} - set(trades.columns)
    if missing:
        msg = f"trades is missing required columns: {sorted(missing)}."
        raise KeyError(msg)
    sign = pd.Series(0, index=trades.index, dtype="int64")
    sign = sign.mask(trades[side_col] == buy_side, 1)
    sign = sign.mask(trades[side_col] == sell_side, -1)
    signed: pd.Series = sign * trades[size_col]
    return signed


def order_flow_imbalance(
    trades: pd.DataFrame,
    rule: str,
    side_col: str = "side",
    size_col: str = "size",
    buy_side: str = "B",
    sell_side: str = "A",
) -> pd.Series:
    """Order-flow imbalance aggregated into time bars.

    Order-flow imbalance (OFI) is buyer-initiated volume minus
    seller-initiated volume within each bar. A positive value means buyers
    were the more aggressive side over the bar.

    Args:
        trades: Trade records with a ``DatetimeIndex`` and the columns
            required by :func:`signed_volume`.
        rule: Resample rule for the bars, e.g. ``"5min"`` or ``"1D"``.
        side_col: Name of the aggressor-side column.
        size_col: Name of the trade-size column.
        buy_side: Side label treated as buyer-initiated.
        sell_side: Side label treated as seller-initiated.

    Returns:
        The per-bar order-flow imbalance, named ``"ofi"``.

    Raises:
        KeyError: If ``side_col`` or ``size_col`` is not in ``trades``.

    """
    signed = signed_volume(trades, side_col, size_col, buy_side, sell_side)
    ofi: pd.Series = signed.resample(rule).sum()
    ofi.name = "ofi"
    return ofi
