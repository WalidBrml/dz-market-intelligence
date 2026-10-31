"""Forecast accuracy metrics, stored as ADDITIVE SUMS so any roll-up is exact (never average ratios).

For a set of (actual a, forecast f) pairs:  n_obs, sum_actual, sum_pred, sum_abs_error = sum|f-a|,
sum_sq_error = sum (f-a)^2.   From them:  MAE = sum_abs_error / n_obs,  MSE = sum_sq_error / n_obs,
RMSE = sqrt(MSE),  WAPE = sum_abs_error / sum_actual,  bias = (sum_pred - sum_actual) / sum_actual
(positive = over-forecast).  Ratios are NaN when their denominator is 0.  MAPE is deliberately not used: demand has many zeros.
Skill vs the seasonal-naive baseline = 1 - MAE(model) / MAE(seasonal_naive); positive = better than the baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUM_COLUMNS = ["n_obs", "sum_actual", "sum_pred", "sum_abs_error", "sum_sq_error"]
KEY_COLUMNS = ["model_name", "evaluation_basis", "level", "origin_date", "horizon_bucket"]
LEVELS = ("series_day", "store_category_week")
BASES = ("observed", "observed_uncensored", "latent_simulation_only")
BASELINE = "seasonal_naive"


def error_sums(actual: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Additive error sums over the entries where `valid` is True."""
    a, f = actual[valid], pred[valid]
    e = f - a
    return {"n_obs": int(valid.sum()), "sum_actual": float(a.sum()), "sum_pred": float(f.sum()),
            "sum_abs_error": float(np.abs(e).sum()), "sum_sq_error": float((e ** 2).sum())}


def series_day_rows(base: dict, actual: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> list[dict]:
    """One row per horizon step 1..H (series x day level)."""
    rows = []
    for h in range(actual.shape[1]):
        v = valid[:, h]
        rows.append({**base, "level": "series_day", "horizon_bucket": h + 1,
                     **error_sums(np.nan_to_num(actual[:, h]), pred[:, h], v)})
    return rows


def weekly_rows(base: dict, actual: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> list[dict]:
    """One row per forecast week (steps 1-7 = week 1, ...). A cell = store x category summed over the week's days.

    `groups` is the (G, n_series) 0/1 membership matrix. A cell is evaluated only if every series-day in it has an actual.
    """
    rows = []
    horizon = actual.shape[1]
    for w in range((horizon + 6) // 7):
        cols = slice(7 * w, min(7 * w + 7, horizon))
        missing = np.isnan(actual[:, cols]).sum(axis=1).astype(float)
        a_cell = groups @ np.nan_to_num(actual[:, cols]).sum(axis=1)
        f_cell = groups @ pred[:, cols].sum(axis=1)
        ok = (groups @ missing == 0) & (groups.sum(axis=1) > 0)
        rows.append({**base, "level": "store_category_week", "horizon_bucket": w + 1,
                     **error_sums(a_cell, f_cell, ok)})
    return rows


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add mae, mse, rmse, wape, bias columns computed from the additive sums."""
    out = df.copy()
    n = out["n_obs"].replace(0, np.nan)
    sa = out["sum_actual"].replace(0, np.nan)
    out["mae"] = out["sum_abs_error"] / n
    out["mse"] = out["sum_sq_error"] / n
    out["rmse"] = np.sqrt(out["mse"])
    out["wape"] = out["sum_abs_error"] / sa
    out["bias"] = (out["sum_pred"] - out["sum_actual"]) / sa
    return out


def summarise(metrics: pd.DataFrame, by_horizon: bool = False) -> pd.DataFrame:
    """Aggregate fold rows over origins (and horizon buckets unless by_horizon) and add accuracy + skill columns."""
    keys = ["model_name", "evaluation_basis", "level"] + (["horizon_bucket"] if by_horizon else [])
    g = metrics.groupby(keys, as_index=False)[SUM_COLUMNS].sum()
    g["n_origins"] = metrics.groupby(keys)["origin_date"].nunique().to_numpy()
    g = add_derived(g)
    base_keys = [k for k in keys if k != "model_name"]
    base = g.loc[g["model_name"] == BASELINE, base_keys + ["mae"]].rename(columns={"mae": "mae_baseline"})
    g = g.merge(base, on=base_keys, how="left")
    g["skill_vs_seasonal_naive"] = 1.0 - g["mae"] / g["mae_baseline"].replace(0, np.nan)
    return g.drop(columns="mae_baseline").sort_values(keys).reset_index(drop=True)
