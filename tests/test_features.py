import numpy as np
import pandas as pd
import pytest

from finm37000.features import (
    bollinger_bands,
    exponential_moving_average,
    order_flow_imbalance,
    realized_volatility,
    returns,
    signed_volume,
    simple_moving_average,
)


def test_simple_moving_average_values() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 4.0])
    sma = simple_moving_average(prices, 2)
    assert pd.isna(sma.iloc[0])
    assert sma.iloc[1:].tolist() == [1.5, 2.5, 3.5]


def test_simple_moving_average_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        simple_moving_average(pd.Series([1.0, 2.0]), 0)


def test_exponential_moving_average_span_one_is_identity() -> None:
    prices = pd.Series([1.0, 2.0, 3.0])
    ema = exponential_moving_average(prices, 1)
    assert ema.tolist() == prices.tolist()


def test_exponential_moving_average_seeds_on_first_value() -> None:
    prices = pd.Series([10.0, 20.0, 30.0])
    ema = exponential_moving_average(prices, 3)
    assert ema.iloc[0] == pytest.approx(10.0)


def test_bollinger_bands_structure_and_symmetry() -> None:
    prices = pd.Series(np.arange(1.0, 11.0))
    bands = bollinger_bands(prices, window=3, num_std=2.0)
    assert list(bands.columns) == ["middle", "upper", "lower", "bandwidth", "pct_b"]
    # middle is exactly the simple moving average.
    pd.testing.assert_series_equal(
        bands["middle"], simple_moving_average(prices, 3), check_names=False
    )
    valid = bands.dropna()
    # Bands straddle the middle and are symmetric about it.
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["lower"] <= valid["middle"]).all()
    mid_gap = valid["upper"] - valid["middle"]
    low_gap = valid["middle"] - valid["lower"]
    np.testing.assert_allclose(mid_gap.to_numpy(), low_gap.to_numpy())


def test_bollinger_bands_rejects_negative_num_std() -> None:
    with pytest.raises(ValueError, match="num_std"):
        bollinger_bands(pd.Series([1.0, 2.0, 3.0]), window=2, num_std=-1.0)


def test_returns_log_and_simple() -> None:
    prices = pd.Series([1.0, 2.0, 4.0])
    log_ret = returns(prices, horizon=1, log=True)
    assert pd.isna(log_ret.iloc[0])
    np.testing.assert_allclose(log_ret.iloc[1:].to_numpy(), [np.log(2.0), np.log(2.0)])
    simple_ret = returns(prices, horizon=1, log=False)
    np.testing.assert_allclose(simple_ret.iloc[1:].to_numpy(), [1.0, 1.0])


def test_returns_horizon_two() -> None:
    prices = pd.Series([1.0, 2.0, 3.0, 4.0])
    simple_ret = returns(prices, horizon=2, log=False)
    assert pd.isna(simple_ret.iloc[1])
    assert simple_ret.iloc[2] == pytest.approx(2.0)


def test_returns_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        returns(pd.Series([1.0, 2.0]), horizon=0)


def test_realized_volatility_annualization_scales_by_sqrt() -> None:
    rng = pd.Series([1.0, 1.1, 1.0, 1.2, 1.15, 1.3])
    raw = realized_volatility(rng, window=3)
    annual = realized_volatility(rng, window=3, periods_per_year=252)
    ratio = (annual.dropna() / raw.dropna()).to_numpy()
    np.testing.assert_allclose(ratio, np.sqrt(252.0))


def test_signed_volume_maps_sides() -> None:
    trades = pd.DataFrame(
        {"side": ["B", "A", "N", "B"], "size": [10, 5, 7, 3]},
    )
    signed = signed_volume(trades)
    assert signed.tolist() == [10, -5, 0, 3]


def test_signed_volume_requires_columns() -> None:
    trades = pd.DataFrame({"side": ["B"]})
    with pytest.raises(KeyError, match="size"):
        signed_volume(trades)


def test_order_flow_imbalance_sums_per_bar() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01 09:00",
            "2026-01-01 09:30",
            "2026-01-01 10:00",
            "2026-01-02 09:00",
        ]
    )
    trades = pd.DataFrame(
        {"side": ["B", "A", "B", "A"], "size": [10, 5, 3, 8]}, index=index
    )
    ofi = order_flow_imbalance(trades, "1D")
    expected_day_one = 10 - 5 + 3
    expected_day_two = -8
    assert ofi.name == "ofi"
    assert ofi.index.tolist() == pd.to_datetime(["2026-01-01", "2026-01-02"]).tolist()
    assert ofi.iloc[0] == expected_day_one
    assert ofi.iloc[1] == expected_day_two
