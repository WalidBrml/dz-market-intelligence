"""Independent oracle generator for Forecasting Performance and Forecast Accuracy by Horizon
(Milestones B4/B5): plain SQL sums on an in-memory SQLite copy of forecast_backtest_metric.csv,
with the DAX rules of _Measures.tmdl re-implemented here (no dzmi_app code). Not a test;
prints the oracle dictionary:  python tests/oracle_generator_forecast.py

Rules: MAE = sum_abs_error/n_obs; RMSE = sqrt(sum_sq_error/n_obs); WAPE = sum_abs_error/sum_actual;
Bias = (sum_pred - sum_actual)/sum_actual; scope = the single selected level / basis, else
series_day / observed; Benchmark MAE = seasonal_naive MAE in the same scope (model filter ignored,
approved spec); Skill = 1 - MAE/Benchmark; MAE Rank = DENSE ascending over the selected models;
Origins Won = origins (of the origin slicer) where the model's MAE equals the best selected model's.
"""

import csv
import math
import pprint
import sqlite3
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "bi_export" / "smoke_m3_20261025"
LABEL = {"naive": "Naive", "seasonal_naive": "Seasonal naive (benchmark)", "moving_average": "Moving average",
         "ses_weekday": "Exp. smoothing + weekday", "croston_sba": "Croston-SBA", "gbm": "Gradient boosting"}
ORDER = ["naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba", "gbm"]
BASES, LEVELS = ["observed", "observed_uncensored"], ["series_day", "store_category_week"]

SCENARIOS = {
    "default": {},
    "weekly": {"levels": ["store_category_week"]},
    "uncensored": {"bases": ["observed_uncensored"]},
    "uncensored_weekly": {"bases": ["observed_uncensored"], "levels": ["store_category_week"]},
    "two_models": {"models": ["croston_sba", "naive"]},
    "one_origin": {"origins": ["2025-04-10"]},
    "two_origins": {"origins": ["2025-03-27", "2025-04-17"]},
    "benchmark_only": {"models": ["seasonal_naive"]},
    "both_bases": {"bases": ["observed", "observed_uncensored"]},
    "models_no_benchmark_weekly": {"models": ["gbm", "ses_weekday", "croston_sba"], "levels": ["store_category_week"]},
}


def load(con):
    cols = ["backtest_run_id", "model_name", "evaluation_basis", "level", "origin_date", "horizon_bucket",
            "n_obs", "sum_actual", "sum_pred", "sum_abs_error", "sum_sq_error"]
    con.execute(f"CREATE TABLE m ({', '.join(cols)})")
    with open(DATA / "forecast_backtest_metric.csv", newline="", encoding="utf-8") as fh:
        con.executemany(f"INSERT INTO m VALUES ({', '.join('?' * len(cols))})", ([r[c] for c in cols] for r in csv.DictReader(fh)))


def sums(con, model, basis, level, origins, bucket=None):
    q, p = "SELECT SUM(CAST(n_obs AS INTEGER)), SUM(CAST(sum_actual AS REAL)), SUM(CAST(sum_pred AS REAL)), SUM(CAST(sum_abs_error AS REAL)), SUM(CAST(sum_sq_error AS REAL)) FROM m WHERE evaluation_basis=? AND level=?", [basis, level]
    if model is not None:
        q += (" AND model_name IN (%s)" % ",".join("?" * len(model))); p += list(model)
    if origins:
        q += " AND origin_date IN (%s)" % ",".join("?" * len(origins)); p += list(origins)
    if bucket is not None:
        q += " AND CAST(horizon_bucket AS INTEGER)=?"; p.append(bucket)
    r = con.execute(q, p).fetchone()
    return None if r[0] is None else dict(n=r[0], actual=r[1], pred=r[2], abs=r[3], sq=r[4])


def metrics(s):
    if s is None:
        return dict(n=None, mae=None, rmse=None, wape=None, bias=None)
    return dict(n=s["n"], mae=s["abs"] / s["n"], rmse=math.sqrt(s["sq"] / s["n"]),
                wape=(s["abs"] / s["actual"]) if s["actual"] else None,
                bias=((s["pred"] - s["actual"]) / s["actual"]) if s["actual"] else None)


def oracle(con, sc):
    sel_models = [m for m in ORDER if not sc.get("models") or m in sc["models"]]
    origins_all = [r[0] for r in con.execute("SELECT DISTINCT origin_date FROM m ORDER BY origin_date")]
    sel_origins = sc.get("origins") or origins_all
    level = sc["levels"][0] if len(sc.get("levels", [])) == 1 else "series_day"
    basis = sc["bases"][0] if len(sc.get("bases", [])) == 1 else "observed"
    sel_bases, sel_levels = sc.get("bases") or BASES, sc.get("levels") or LEVELS
    out = {"level": level, "basis": basis, "origins_count": len(sel_origins)}
    # scope message labels
    out["scope_message"] = ("Scope: %s  |  %s  |  pooled over all selected origins" % (
        {"observed": "Observed sales", "observed_uncensored": "Observed, no-stockout days"}[basis] if len(sc.get("bases", [])) == 1 else "Observed sales (default)",
        {"series_day": "Series x day", "store_category_week": "Store x category x week"}[level] if len(sc.get("levels", [])) == 1 else "Series x day (default)"))

    bench = metrics(sums(con, ["seasonal_naive"], basis, level, sel_origins))["mae"]
    rows = []
    maes = {m: metrics(sums(con, [m], basis, level, sel_origins))["mae"] for m in sel_models}
    distinct = sorted({v for v in maes.values() if v is not None})
    for m in sel_models:
        mt = metrics(sums(con, [m], basis, level, sel_origins))
        if mt["n"] is None:
            continue
        won = 0
        for o in sel_origins:
            mine = metrics(sums(con, [m], basis, level, [o]))["mae"]
            best = min([v for v in (metrics(sums(con, [x], basis, level, [o]))["mae"] for x in sel_models) if v is not None], default=None)
            if mine is not None and best is not None and mine == best:
                won += 1
        rows.append(dict(model=m, label=LABEL[m], **mt, benchmark_mae=bench,
                         skill=(1 - mt["mae"] / bench) if bench else None,
                         rank=distinct.index(mt["mae"]) + 1, origins_won=won))
    out["models"] = sorted(rows, key=lambda r: (r["mae"], r["label"]))
    out["empty"] = not rows
    out["by_origin"] = {m: {o: metrics(sums(con, [m], basis, level, [o]))["mae"] for o in sel_origins} for m in sel_models}

    best_rows = []
    for b in sel_bases:
        for lv in sel_levels:
            ms = {m: metrics(sums(con, [m], b, lv, sel_origins))["mae"] for m in sel_models}
            vals = {m: v for m, v in ms.items() if v is not None}
            if not vals:
                continue
            best = min(vals.values())
            bm = metrics(sums(con, ["seasonal_naive"], b, lv, sel_origins))["mae"]
            beating = sum(1 for v in vals.values() if bm and (1 - v / bm) > 0)
            best_rows.append(dict(basis=b, level=lv, best_model=min(LABEL[m] for m, v in vals.items() if v == best),
                                  best_mae=best, benchmark_mae=bm, best_vs_benchmark=(1 - best / bm) if bm else None,
                                  models_beating=beating))
    out["best_table"] = best_rows

    buckets = [r[0] for r in con.execute("SELECT DISTINCT CAST(horizon_bucket AS INTEGER) FROM m WHERE level=? ORDER BY 1", (level,))]
    hz = []
    for bk in buckets:
        per_model = {m: metrics(sums(con, [m], basis, level, sel_origins, bk)) for m in sel_models}
        pooled = sums(con, sel_models, basis, level, sel_origins, bk)
        if pooled is None:
            continue
        hz.append(dict(bucket=bk, observations=pooled["n"], per_model={m: v for m, v in per_model.items() if v["n"] is not None}))
    out["horizon"] = hz
    return out


if __name__ == "__main__":
    con = sqlite3.connect(":memory:")
    load(con)
    pprint.pprint({k: oracle(con, v) for k, v in SCENARIOS.items()}, width=120, sort_dicts=False)
