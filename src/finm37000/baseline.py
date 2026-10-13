"""Fit a baseline future-return predictor and evaluate it on a holdout.

The baseline is a standardized linear (optionally ridge) regression of
``target_return`` on the causal features from
:func:`finm37000.dataset.build_prediction_dataset`. Standardization and the
coefficients are estimated from training rows only, and the held-out
predictions are compared with a benchmark that always predicts a zero return.

Corn futures trade in quarter-cent ticks, so many short-horizon returns are
exactly zero. The directional metric therefore states how zeros are handled:
rows whose realized return is zero are excluded, and a zero prediction is an
abstention rather than a guess (see :func:`evaluate_predictions`).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dataset import FEATURE_COLUMNS, build_prediction_dataset, chronological_split

TARGET_COLUMN = "target_return"


@dataclass(frozen=True)
class LinearBaseline:
    """A fitted standardized linear model.

    Attributes:
        feature_columns: Feature names, in coefficient order.
        means: Training means used to center each feature.
        scales: Training standard deviations used to scale each feature.
            Constant features get a scale of one, so they contribute nothing.
        coef: Coefficients on the standardized features.
        intercept: Intercept, equal to the training mean target.
        ridge: The ridge penalty used in the fit.
    """

    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    intercept: float
    ridge: float

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict target returns for rows of ``frame``.

        Args:
            frame: Rows containing every column in ``feature_columns``.

        Returns:
            Predicted returns aligned to ``frame``, named ``"prediction"``.

        Raises:
            KeyError: If a feature column is missing.
            ValueError: If a feature value is missing or not finite.

        """
        x = _feature_matrix(frame, self.feature_columns)
        z = (x - self.means) / self.scales
        return pd.Series(
            z @ self.coef + self.intercept, index=frame.index, name="prediction"
        )


def _feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = set(columns) - set(frame.columns)
    if missing:
        msg = f"Missing feature columns: {sorted(missing)}."
        raise KeyError(msg)
    x: np.ndarray = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        msg = "Feature values must be finite."
        raise ValueError(msg)
    return x


def fit_linear_baseline(
    train: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    ridge: float = 0.0,
) -> LinearBaseline:
    """Fit a standardized linear regression using training rows only.

    Each feature is centered and scaled by its training mean and standard
    deviation; the target is centered by its training mean, which becomes
    the intercept. Coefficients solve
    ``(Z'Z + ridge * n * I) b = Z'y`` so the penalty is per observation and
    comparable across sample sizes. ``ridge=0`` is ordinary least squares
    (solved with ``lstsq``, so collinear features do not fail).

    Args:
        train: Training rows with the feature columns and ``target_return``.
        feature_columns: Features to use. Never include ``target_return``
            or ``target_end``.
        ridge: Nonnegative ridge penalty.

    Returns:
        The fitted model.

    Raises:
        ValueError: For an invalid penalty, leaked target columns, too few
            rows, or nonfinite values.
        KeyError: If a feature or the target column is missing.

    """
    if not np.isfinite(ridge) or ridge < 0:
        msg = "ridge must be a finite, nonnegative number."
        raise ValueError(msg)
    columns = tuple(feature_columns)
    leaked = {TARGET_COLUMN, "target_end"} & set(columns)
    if leaked:
        msg = f"Target columns cannot be features: {sorted(leaked)}."
        raise ValueError(msg)
    if TARGET_COLUMN not in train:
        msg = f"train must contain {TARGET_COLUMN}."
        raise KeyError(msg)
    x = _feature_matrix(train, columns)
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        msg = "Target values must be finite."
        raise ValueError(msg)
    n = len(train)
    if n < len(columns) + 2:
        msg = "Not enough training rows to fit the baseline."
        raise ValueError(msg)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    z = (x - means) / scales
    intercept = float(y.mean())
    yc = y - intercept
    if ridge == 0:
        coef = np.linalg.lstsq(z, yc, rcond=None)[0]
    else:
        coef = np.linalg.solve(z.T @ z + ridge * n * np.eye(len(columns)), z.T @ yc)
    return LinearBaseline(columns, means, scales, coef, intercept, float(ridge))


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Error and directional metrics for predicted returns.

    Directional metric: ``hit_rate`` is the share of *scored* rows where the
    predicted and realized returns have the same sign. A row is scored only
    when the realized return is nonzero (a zero move has no direction) and
    the prediction is nonzero (a zero prediction makes no directional call).
    ``coverage`` is the share of nonzero-return rows that were scored, so a
    model that always predicts zero has coverage 0 and an undefined (NaN)
    hit rate rather than a misleading score.

    Args:
        actual: Realized returns.
        predicted: Predicted returns aligned to ``actual``.

    Returns:
        ``n``, ``mae``, ``rmse``, ``zero_actual_share``, ``hit_rate`` and
        ``coverage``.

    Raises:
        ValueError: If the inputs are misaligned, empty or not finite.

    """
    if not actual.index.equals(predicted.index) or actual.empty:
        msg = "actual and predicted must be nonempty with identical indexes."
        raise ValueError(msg)
    y = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        msg = "actual and predicted must be finite."
        raise ValueError(msg)
    error = p - y
    moved = y != 0
    scored = moved & (p != 0)
    hits = np.sign(p[scored]) == np.sign(y[scored])
    return {
        "n": float(len(y)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "zero_actual_share": float((~moved).mean()),
        "hit_rate": float(hits.mean()) if scored.any() else float("nan"),
        "coverage": float(scored.sum() / moved.sum()) if moved.any() else 0.0,
    }


def _json_safe(metrics: dict[str, float]) -> dict[str, float | None]:
    return {k: None if np.isnan(v) else v for k, v in metrics.items()}


def _span(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_observation": str(frame.index[0]),
        "last_observation": str(frame.index[-1]),
        "last_target_end": str(frame["target_end"].max()),
    }


def run_baseline(
    bars: pd.DataFrame,
    window: int = 20,
    horizon: int = 1,
    test_fraction: float = 0.2,
    gap: int = 0,
    ridge: float = 0.0,
) -> dict[str, Any]:
    """Build the dataset, fit on the training split and score the holdout once.

    All settings are fixed before the call; nothing is tuned on the holdout.
    If settings need to be chosen, compare them with a further chronological
    split inside the training rows.

    Args:
        bars: End-labeled bars accepted by ``build_prediction_dataset``.
        window: Rolling feature window, in bars.
        horizon: Target horizon, in observed bars.
        test_fraction: Fraction of the latest rows held out.
        gap: Extra rows excluded before the holdout.
        ridge: Ridge penalty passed to :func:`fit_linear_baseline`.

    Returns:
        A strict-JSON report (undefined metrics are ``None``) with the
        settings, the train and test sample dates, the fitted coefficients,
        and holdout metrics for the model and the zero-return benchmark.
        ``oos_r2_vs_zero`` is ``1 - SSE_model / SSE_zero``; positive values
        beat the benchmark.

    """
    dataset = build_prediction_dataset(bars, window=window, horizon=horizon)
    train, test = chronological_split(dataset, test_fraction=test_fraction, gap=gap)
    model = fit_linear_baseline(train, ridge=ridge)
    actual = test[TARGET_COLUMN]
    predicted = model.predict(test)
    zero = pd.Series(0.0, index=test.index)
    sse_model = float(((predicted - actual) ** 2).sum())
    sse_zero = float((actual**2).sum())
    symbol = str(bars["symbol"].iloc[0]) if "symbol" in bars else None
    return {
        "settings": {
            "symbol": symbol,
            "window": window,
            "horizon": horizon,
            "test_fraction": test_fraction,
            "gap": gap,
            "ridge": ridge,
            "features": list(model.feature_columns),
        },
        "sample": {
            "bars": len(bars),
            "dataset_rows": len(dataset),
            "purged_rows": len(dataset) - len(train) - len(test),
            "train": _span(train),
            "test": _span(test),
        },
        "model": {
            "intercept": model.intercept,
            "coef": dict(zip(model.feature_columns, model.coef.tolist(), strict=True)),
        },
        "holdout": {
            "baseline": _json_safe(evaluate_predictions(actual, predicted)),
            "zero_benchmark": _json_safe(evaluate_predictions(actual, zero)),
            "oos_r2_vs_zero": 1 - sse_model / sse_zero if sse_zero > 0 else None,
        },
    }
