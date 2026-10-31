"""Persist forecast runs and backtest metrics. Nothing is ever overwritten or deleted: an existing run id is an error."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..errors import DZMIError
from ..features.build import FEATURE_DEFINITION
from ..features.panel import Panel
from ..ingestion.db import Database
from .backtest import BacktestResult
from .config import ForecastSettings
from .metrics import KEY_COLUMNS, SUM_COLUMNS
from .models import MODEL_DESCRIPTIONS

MODEL_VERSION = "dzmi-m3-1"
_RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0))


def check_run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id or ""):
        raise DZMIError("A run id may contain only letters, digits, '_' and '-' (maximum 40 characters).")
    return run_id


def default_backtest_id(profile: str, now: pd.Timestamp | None = None) -> str:
    now = now or utc_now()
    return check_run_id(f"bt_{re.sub(r'[^A-Za-z0-9]', '', profile)[:18]}_{now:%Y%m%d_%H%M%S}")


def default_forecast_id(model: str, origin: pd.Timestamp, now: pd.Timestamp | None = None) -> str:
    now = now or utc_now()
    return check_run_id(f"fc_{model}_{origin:%Y%m%d}_{now:%H%M%S}")


def ensure_free(db: Database, table: str, id_column: str, run_id: str) -> None:
    """Fail BEFORE any heavy work if the id is already used (a re-run never overwrites a previous run)."""
    check_run_id(run_id)
    if db.scalar(f"SELECT COUNT(*) FROM dwh.{table} WHERE {id_column} = '{run_id}'"):
        raise DZMIError(f"Run id '{run_id}' already exists in dwh.{table}. Nothing was changed. "
                        "Choose another --run-id (previous runs are never overwritten).")


def save_backtest(db: Database, run_id: str, result: BacktestResult, notes: str = "") -> int:
    """Write dwh.backtest_run and dwh.forecast_backtest_metric. Returns the number of metric rows."""
    ensure_free(db, "backtest_run", "backtest_run_id", run_id)
    s = result.settings
    head = pd.DataFrame([{
        "backtest_run_id": run_id, "created_at": utc_now(), "profile": s.profile, "horizon_days": s.horizon_days,
        "n_origins": len(result.origins), "origin_step_days": s.origin_step_days, "min_train_days": s.min_train_days,
        "n_series": result.n_series, "censor_policy": s.censor_policy, "seed": s.seed, "models": ",".join(s.models),
        "settings_json": json.dumps(s.to_dict(), sort_keys=True), "notes": notes or None}])
    db.write_df(head, "dwh", "backtest_run")
    m = result.metrics[KEY_COLUMNS + SUM_COLUMNS].copy()
    for c in ("sum_actual", "sum_pred", "sum_abs_error", "sum_sq_error"):
        m[c] = m[c].round(3)
    m.insert(0, "backtest_run_id", run_id)
    try:
        return db.write_df(m, "dwh", "forecast_backtest_metric")
    except Exception as exc:
        raise DZMIError(f"The metrics of run '{run_id}' could not be saved ({exc}). The run header was written; "
                        "validate-db reports it as a run without metrics.") from exc


def save_forecast(db: Database, run_id: str, model: str, panel: Panel, origin_idx: int, pred: np.ndarray,
                  s: ForecastSettings, notes: str = "") -> int:
    """Write dwh.model_run and dwh.fact_demand_forecast for one model at one origin. Returns forecast rows."""
    ensure_free(db, "model_run", "forecast_run_id", run_id)
    horizon = pred.shape[1]
    definition = f"{model}: {MODEL_DESCRIPTIONS[model]}. Censoring policy: {s.censor_policy}."
    if model == "gbm":
        definition += " " + FEATURE_DEFINITION
    head = pd.DataFrame([{
        "forecast_run_id": run_id, "model_name": model, "model_version": MODEL_VERSION, "trained_at": utc_now(),
        "train_end_date": panel.dates[origin_idx].normalize(), "horizon_days": horizon,
        "feature_definition": definition, "notes": notes or None}])
    db.write_df(head, "dwh", "model_run")
    days = panel.dates[origin_idx + 1: origin_idx + 1 + horizon]
    date_keys = (days.year * 10000 + days.month * 100 + days.day).to_numpy(dtype=np.int64)
    n = panel.n_series
    df = pd.DataFrame({
        "forecast_run_id": run_id,
        "store_key": np.repeat(panel.series["store_key"].to_numpy(dtype=np.int64), horizon),
        "product_key": np.repeat(panel.series["product_key"].to_numpy(dtype=np.int64), horizon),
        "date_key": np.tile(date_keys, n),
        "horizon_step": np.tile(np.arange(1, horizon + 1), n),
        "predicted_units": np.round(pred.reshape(-1), 3)})
    try:
        return db.write_df(df, "dwh", "fact_demand_forecast")
    except Exception as exc:
        raise DZMIError(f"The forecasts of run '{run_id}' could not be saved ({exc}). The run header was written; "
                        "validate-db reports it as a run without forecasts.") from exc
