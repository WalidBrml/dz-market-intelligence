"""Baseline and statistical forecasters, vectorised over series. All return an (n_series, horizon) array of >= 0.

Every function receives only a history (Panel sliced at the origin) and the future calendar: leakage is impossible
by construction. NaN in the demand matrix means "ignore this day": missing days and, under the "exclude" censoring
policy, days whose end-of-day stock was 0 (sales may have been truncated by the stockout).

Models
  naive            last usable observation, repeated
  seasonal_naive   the value one week earlier (same weekday); if that day is unusable, 2, 3, 4 weeks earlier;
                   if none is usable, the 28-day mean
  moving_average   mean of the last `window` usable days, repeated
  ses_weekday      simple exponential smoothing of the weekday-adjusted series x weekday index (alpha chosen per series)
  croston_sba      Croston's intermittent-demand method with the Syntetos-Boylan bias correction
The gradient boosting model lives in forecasting/gbm.py.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..features.panel import Future, Panel
from ..features.stats import nanmean_last, nanmean_rows
from .config import ForecastSettings


def usable_demand(hist: Panel, policy: str) -> np.ndarray:
    """(n, T) float demand with NaN on days that must be ignored."""
    y = hist.units.astype(float).copy()
    if policy == "exclude":
        y[hist.censored] = np.nan
    return y


def _finish(pred: np.ndarray) -> np.ndarray:
    return np.maximum(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def naive(y: np.ndarray, horizon: int) -> np.ndarray:
    n, t = y.shape
    valid = ~np.isnan(y)
    last = t - 1 - np.argmax(valid[:, ::-1], axis=1)
    out = np.where(valid.any(axis=1), y[np.arange(n), last], 0.0)
    return _finish(np.repeat(out[:, None], horizon, axis=1))


def seasonal_naive(y: np.ndarray, horizon: int, period: int = 7, max_back: int = 4) -> np.ndarray:
    n, t = y.shape
    out = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        base = t - 1 + k - period * math.ceil(k / period)      # last observed day with the same weekday as target k
        for i in range(max_back):
            j = base - period * i
            if j < 0:
                break
            v = y[:, j]
            fill = np.isnan(out[:, k - 1]) & ~np.isnan(v)
            out[fill, k - 1] = v[fill]
    fallback = nanmean_last(y, 28)
    return _finish(np.where(np.isnan(out), fallback[:, None], out))


def moving_average(y: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    return _finish(np.repeat(nanmean_last(y, window)[:, None], horizon, axis=1))


def _ses_level(z: np.ndarray, alpha: float, fit_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Final SES level per series and the squared one-step error over the last `fit_window` days (NaN days skipped)."""
    n, t = z.shape
    level = np.nan_to_num(nanmean_rows(z[:, :14]), nan=0.0)
    sse = np.zeros(n)
    for j in range(t):
        zt = z[:, j]
        valid = ~np.isnan(zt)
        err = np.where(valid, zt - level, 0.0)
        if j >= t - fit_window:
            sse += err ** 2
        level = level + alpha * err
    return level, sse


def ses_weekday(y: np.ndarray, hist_dow: np.ndarray, fut_dow: np.ndarray, alphas=(0.05, 0.1, 0.2, 0.3),
                fit_window: int = 56, index_window: int = 84) -> np.ndarray:
    n, t = y.shape
    recent = slice(max(0, t - index_window), t)
    ys, ds = y[:, recent], hist_dow[recent]
    overall = nanmean_rows(ys)
    idx = np.ones((n, 7))
    for d in range(7):
        m = ds == d
        if m.any():
            wd = nanmean_rows(ys[:, m])
            idx[:, d] = np.where(overall > 0, wd / np.where(overall > 0, overall, 1.0), 1.0)
    idx = np.nan_to_num(idx, nan=1.0)
    adj = idx[:, hist_dow]
    z = np.where(adj > 0, y / np.where(adj > 0, adj, 1.0), np.nan)
    best_sse = np.full(n, np.inf)
    best_level = np.zeros(n)
    for a in alphas:
        level, sse = _ses_level(z, float(a), fit_window)
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        best_level = np.where(better, level, best_level)
    return _finish(best_level[:, None] * idx[:, fut_dow])


def croston_sba(y: np.ndarray, horizon: int, alpha: float = 0.1) -> np.ndarray:
    """Croston with SBA correction: forecast = (1 - alpha/2) * z / p, z = demand size, p = interval between demands.

    Unusable (NaN) days are skipped entirely: they do not lengthen the interval.
    """
    n, t = y.shape
    z = np.zeros(n)
    p = np.zeros(n)
    q = np.zeros(n)
    started = np.zeros(n, dtype=bool)
    for j in range(t):
        d = y[:, j]
        valid = ~np.isnan(d)
        q = q + valid
        hit = valid & (np.nan_to_num(d) > 0)
        first = hit & ~started
        upd = hit & started
        z = np.where(first, d, np.where(upd, z + alpha * (d - z), z))
        p = np.where(first, q, np.where(upd, p + alpha * (q - p), p))
        started |= hit
        q = np.where(hit, 0.0, q)
    out = np.where(started, (1 - alpha / 2) * z / np.where(p > 0, p, 1.0), 0.0)
    return _finish(np.repeat(out[:, None], horizon, axis=1))


def forecast_baseline(name: str, hist: Panel, fut: Future, s: ForecastSettings) -> np.ndarray:
    """Dispatch for the non-ML models."""
    y = usable_demand(hist, s.censor_policy)
    h = len(fut.dates)
    prm = s.model_params.get(name, {})
    if name == "naive":
        return naive(y, h)
    if name == "seasonal_naive":
        return seasonal_naive(y, h, s.seasonal_period, int(prm.get("max_back_weeks", 4)))
    if name == "moving_average":
        return moving_average(y, h, int(prm.get("window", 28)))
    if name == "ses_weekday":
        return ses_weekday(y, hist.dates.dayofweek.to_numpy(), fut.calendar["dow"].to_numpy(dtype=int),
                           tuple(prm.get("alphas", (0.05, 0.1, 0.2, 0.3))), int(prm.get("fit_window", 56)),
                           int(prm.get("index_window", 84)))
    if name == "croston_sba":
        return croston_sba(y, h, float(prm.get("alpha", 0.1)))
    raise KeyError(name)


def forecast_model(name: str, hist: Panel, fut: Future, s: ForecastSettings) -> np.ndarray:
    """Forecast of any configured model for the days in `fut`, using only `hist` and the known future."""
    if name == "gbm":
        from .gbm import forecast_gbm
        return forecast_gbm(hist, fut, s)
    return forecast_baseline(name, hist, fut, s)


MODEL_DESCRIPTIONS: dict[str, str] = {
    "naive": "last usable observation repeated",
    "seasonal_naive": "same weekday one week earlier (older weeks if that day is unusable)",
    "moving_average": "mean of the last 28 usable days",
    "ses_weekday": "exponential smoothing level x weekday index (alpha chosen per series)",
    "croston_sba": "Croston with Syntetos-Boylan correction (intermittent demand)",
    "gbm": "gradient boosting, Poisson loss, direct multi-horizon (scikit-learn HistGradientBoostingRegressor)",
}
