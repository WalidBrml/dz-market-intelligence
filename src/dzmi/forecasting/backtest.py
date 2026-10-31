"""Rolling-origin backtest: train on the past, forecast the next `horizon` days, move the origin forward, repeat.

For origin day o the model sees `panel.slice_to(o)` and `panel.future_known(o, H)` only; the actuals for days o+1..o+H
are used afterwards to score the forecast. Accuracy is measured on OBSERVED sales (what a retailer can see), on the
subset of days without a stockout, and optionally (simulation only) against latent demand.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..errors import DZMIError
from ..features.panel import Panel
from . import metrics as M
from .config import ForecastSettings
from .models import forecast_model

log = logging.getLogger("dzmi.forecasting")


def make_origins(n_days: int, horizon: int, min_train_days: int, n_origins: int, step: int) -> list[int]:
    """Origin day indices (last training day), oldest first. The newest origin leaves exactly `horizon` days of actuals."""
    last = n_days - 1 - horizon
    if last < min_train_days - 1:
        raise DZMIError(f"Not enough history for a backtest: {n_days} days loaded, but {min_train_days} training days "
                        f"plus a {horizon}-day horizon are needed. Lower --min-train/--horizon or load more days.")
    origins = [last - k * step for k in range(n_origins) if last - k * step >= min_train_days - 1]
    return sorted(origins)


def group_matrix(panel: Panel) -> np.ndarray:
    """(G, n_series) 0/1 matrix: membership of each series in a store x category group."""
    codes, _ = pd.factorize(pd.Series(list(zip(panel.series["store_key"], panel.series["category_code"]))))
    m = np.zeros((codes.max() + 1, panel.n_series))
    m[codes, np.arange(panel.n_series)] = 1.0
    return m


@dataclass
class BacktestResult:
    settings: ForecastSettings
    origins: list[pd.Timestamp]
    n_series: int
    metrics: pd.DataFrame
    seconds: dict[str, float] = field(default_factory=dict)
    latent_used: bool = False


def evaluate_fold(model: str, origin_date: pd.Timestamp, pred: np.ndarray, actual: np.ndarray, censored: np.ndarray,
                  groups: np.ndarray, latent: np.ndarray | None = None) -> list[dict]:
    """All metric rows of one (model, origin)."""
    base = {"model_name": model, "origin_date": origin_date}
    rows = []
    obs = ~np.isnan(actual)
    rows += M.series_day_rows({**base, "evaluation_basis": "observed"}, actual, pred, obs)
    rows += M.weekly_rows({**base, "evaluation_basis": "observed"}, actual, pred, groups)
    rows += M.series_day_rows({**base, "evaluation_basis": "observed_uncensored"}, actual, pred, obs & ~censored)
    if latent is not None:
        rows += M.series_day_rows({**base, "evaluation_basis": "latent_simulation_only"}, latent, pred, ~np.isnan(latent))
        rows += M.weekly_rows({**base, "evaluation_basis": "latent_simulation_only"}, latent, pred, groups)
    return rows


def run_backtest(panel: Panel, s: ForecastSettings, latent: np.ndarray | None = None,
                 progress: Callable[[str], None] | None = None) -> BacktestResult:
    """Run every configured model on every origin. `latent` (n, T), if given, adds the simulation-only diagnostic."""
    say = progress or (lambda msg: log.info(msg))
    h = s.horizon_days
    idxs = make_origins(panel.n_days, h, s.min_train_days, s.n_origins, s.origin_step_days)
    if len(idxs) < s.n_origins:
        say(f"Only {len(idxs)} of {s.n_origins} origins fit in the {panel.n_days} loaded days.")
    groups = group_matrix(panel)
    rows: list[dict] = []
    seconds = {m: 0.0 for m in s.models}
    for o in idxs:
        hist, fut = panel.slice_to(o), panel.future_known(o, h)
        sl = slice(o + 1, o + 1 + h)
        actual, cens = panel.units[:, sl], panel.censored[:, sl]
        lat = None if latent is None else latent[:, sl]
        for m in s.models:
            t0 = time.perf_counter()
            pred = forecast_model(m, hist, fut, s)
            seconds[m] += time.perf_counter() - t0
            rows += evaluate_fold(m, panel.dates[o], pred, actual, cens, groups, lat)
        say(f"origin {panel.dates[o].date()} done ({len(rows):,} metric rows so far)")
    df = pd.DataFrame(rows)
    df = df[M.KEY_COLUMNS + M.SUM_COLUMNS]
    return BacktestResult(s, [panel.dates[o] for o in idxs], panel.n_series, df, seconds, latent is not None)


def single_forecast(panel: Panel, s: ForecastSettings, model: str, origin_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """One model at one origin: (forecast, actual, censored, target dates). Uses only history up to origin_idx."""
    fut = panel.future_known(origin_idx, s.horizon_days)
    pred = forecast_model(model, panel.slice_to(origin_idx), fut, s)
    sl = slice(origin_idx + 1, origin_idx + 1 + s.horizon_days)
    return pred, panel.units[:, sl], panel.censored[:, sl], fut.dates
