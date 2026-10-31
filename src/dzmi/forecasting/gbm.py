"""Gradient boosting forecaster (scikit-learn HistGradientBoostingRegressor with Poisson loss).

One global model per forecast origin, trained only on examples whose features AND target lie inside the history up to
that origin (features/build.py: training_frame). scikit-learn is imported lazily so the rest of dzmi (Milestones 1-2 and
the baseline models) works without it.
"""
from __future__ import annotations

import numpy as np

from ..errors import DZMIError
from ..features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, features_at, training_frame
from ..features.panel import Future, Panel
from .config import ForecastSettings

INSTALL_HINT = "scikit-learn is required for the gbm model. Install it with:  python -m pip install \"scikit-learn>=1.7,<2\""


def _regressor(prm: dict, seed: int):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:                                     # pragma: no cover - depends on the machine
        raise DZMIError(INSTALL_HINT) from exc
    cat = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=int(prm.get("max_iter", 150)), learning_rate=float(prm.get("learning_rate", 0.08)),
        max_leaf_nodes=int(prm.get("max_leaf_nodes", 31)), min_samples_leaf=int(prm.get("min_samples_leaf", 40)),
        l2_regularization=float(prm.get("l2_regularization", 1.0)), categorical_features=cat,
        early_stopping=False, random_state=seed)


def forecast_gbm(hist: Panel, fut: Future, s: ForecastSettings) -> np.ndarray:
    prm = s.model_params.get("gbm", {})
    horizon = len(fut.dates)
    x, y = training_frame(hist, horizon, s.censor_policy, int(prm.get("train_origin_step", 7)),
                          int(prm.get("max_train_origins", 52)), int(prm.get("min_history_days", 28)))
    n = hist.n_series
    if len(y) < 100 or float(y.sum()) <= 0:
        return np.zeros((n, horizon))                              # not enough signal to fit: predict no demand
    model = _regressor(prm, s.seed).fit(x, y)
    xo = features_at(hist, hist.n_days - 1, fut.calendar, fut.promo_discount, s.censor_policy)
    return np.maximum(model.predict(xo).reshape(n, horizon), 0.0)
