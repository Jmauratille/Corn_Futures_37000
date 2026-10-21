import json

import numpy as np
import pandas as pd
import pytest

from finm37000.baseline import (
    evaluate_predictions,
    fit_linear_baseline,
    run_baseline,
)
from finm37000.dataset import FEATURE_COLUMNS, build_prediction_dataset

HORIZON = 3
MAX_RANDOM_WALK_R2 = 0.02
HIT_RATE_RANGE = (0.4, 0.6)


def random_walk_bars(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 450 + 0.25 * np.cumsum(rng.choice([-1, 0, 1], n))
    return pd.DataFrame(
        {"symbol": "ZCH7", "close": close, "volume": rng.integers(1, 100, n)},
        index=pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC"),
    )


def linear_rows(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))),
        columns=list(FEATURE_COLUMNS),
    )
    frame["target_return"] = 0.5 + 2 * frame["return_1"] - frame["volume"]
    return frame


def test_ols_recovers_known_linear_relationship() -> None:
    rows = linear_rows()
    model = fit_linear_baseline(rows)
    np.testing.assert_allclose(model.predict(rows), rows["target_return"], atol=1e-10)
    coef = dict(zip(model.feature_columns, model.coef, strict=True))
    assert coef["return_1"] == pytest.approx(2 * model.scales[0])
    assert coef["sma_distance"] == pytest.approx(0, abs=1e-10)


def test_scaling_uses_training_rows_only() -> None:
    rows = linear_rows()
    train, test = rows.iloc[:100], rows.iloc[100:].copy()
    model = fit_linear_baseline(train)
    np.testing.assert_allclose(model.means, train[list(FEATURE_COLUMNS)].mean())
    test[list(FEATURE_COLUMNS)] *= 1000
    refit = fit_linear_baseline(train)
    np.testing.assert_array_equal(model.coef, refit.coef)
    assert model.predict(test).index.equals(test.index)


def test_ridge_shrinks_and_constant_feature_is_harmless() -> None:
    rows = linear_rows().assign(bandwidth=3.0)
    ols = fit_linear_baseline(rows)
    ridge = fit_linear_baseline(rows, ridge=10.0)
    assert np.linalg.norm(ridge.coef) < np.linalg.norm(ols.coef)
    assert np.isfinite(ols.predict(rows)).all()
    assert ridge.intercept == pytest.approx(rows["target_return"].mean())


@pytest.mark.parametrize("ridge", [-1.0, np.nan, np.inf])
def test_invalid_ridge(ridge: float) -> None:
    with pytest.raises(ValueError, match="ridge"):
        fit_linear_baseline(linear_rows(), ridge=ridge)


def test_target_columns_cannot_be_features() -> None:
    with pytest.raises(ValueError, match="Target columns"):
        fit_linear_baseline(linear_rows(), ("return_1", "target_return"))


def test_fit_input_validation() -> None:
    rows = linear_rows()
    with pytest.raises(KeyError, match="target_return"):
        fit_linear_baseline(rows.drop(columns="target_return"))
    with pytest.raises(KeyError, match="volume"):
        fit_linear_baseline(rows.drop(columns="volume"))
    with pytest.raises(ValueError, match="Not enough"):
        fit_linear_baseline(rows.iloc[:3])
    rows.loc[0, "return_1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_linear_baseline(rows)


def test_metrics_and_zero_return_treatment() -> None:
    index = pd.RangeIndex(5)
    actual = pd.Series([0.02, -0.01, 0.0, 0.03, -0.02], index=index)
    predicted = pd.Series([0.01, 0.01, 0.05, 0.0, -0.01], index=index)
    metrics = evaluate_predictions(actual, predicted)
    errors = np.array([0.01, 0.02, 0.05, 0.03, 0.01])
    assert metrics["mae"] == pytest.approx(errors.mean())
    assert metrics["rmse"] == pytest.approx(np.sqrt((errors**2).mean()))
    assert metrics["zero_actual_share"] == pytest.approx(0.2)
    # Scored rows: 0, 1, 4 (row 2 did not move; row 3 predicted zero).
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == pytest.approx(3 / 4)


def test_zero_benchmark_has_no_directional_score() -> None:
    actual = pd.Series([0.01, -0.02, 0.0])
    metrics = evaluate_predictions(actual, pd.Series(0.0, index=actual.index))
    assert np.isnan(metrics["hit_rate"])
    assert metrics["coverage"] == 0
    assert metrics["mae"] == pytest.approx(0.01)


def test_metrics_reject_misaligned_or_bad_inputs() -> None:
    actual = pd.Series([0.01, 0.02])
    with pytest.raises(ValueError, match="identical"):
        evaluate_predictions(actual, pd.Series([0.01, 0.02], index=[1, 2]))
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_predictions(actual.iloc[:0], actual.iloc[:0])
    with pytest.raises(ValueError, match="finite"):
        evaluate_predictions(actual, pd.Series([np.nan, 0.0]))


def test_report_records_settings_and_respects_split() -> None:
    bars = random_walk_bars()
    report = run_baseline(bars, window=10, horizon=HORIZON, test_fraction=0.25)
    json.dumps(report, allow_nan=False)  # Strict JSON, no NaN.
    assert report["settings"]["symbol"] == "ZCH7"
    assert report["settings"]["horizon"] == HORIZON
    train, test = report["sample"]["train"], report["sample"]["test"]
    assert pd.Timestamp(train["last_target_end"]) < pd.Timestamp(
        test["first_observation"]
    )
    dataset = build_prediction_dataset(bars, window=10, horizon=HORIZON)
    assert test["rows"] == int(np.ceil(len(dataset) * 0.25))
    assert report["sample"]["purged_rows"] == HORIZON


def test_holdout_changes_do_not_change_the_fit() -> None:
    bars = random_walk_bars()
    changed = bars.copy()
    changed.iloc[-50:, changed.columns.get_loc("close")] *= 1.1
    original = run_baseline(bars, window=10)
    altered = run_baseline(changed, window=10)
    assert original["model"] == altered["model"]
    assert original["holdout"] != altered["holdout"]


def test_random_walk_has_no_reliable_edge() -> None:
    report = run_baseline(random_walk_bars(n=3000, seed=7), window=20)
    holdout = report["holdout"]
    assert holdout["oos_r2_vs_zero"] < MAX_RANDOM_WALK_R2
    low, high = HIT_RATE_RANGE
    assert low < holdout["baseline"]["hit_rate"] < high
