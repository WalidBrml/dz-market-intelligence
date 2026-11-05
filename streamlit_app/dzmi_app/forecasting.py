"""Forecasting Performance (Page 4): context, filters and results. No Streamlit here.

Everything on this page is BACKTEST accuracy: forecasts made at past origins and scored against what
was observed afterwards. fact_demand_forecast and model_run are empty in this dataset, so no
forecast of the future exists and none is produced.

Filter propagation (relationships.tmdl): the four slicers (evaluation basis, aggregation level, model,
forecast origin) filter forecast_backtest_metric through dim_eval_basis, dim_eval_level, dim_model and
dim_origin. Measures then pin the level and basis to the single selected value, else to the defaults.
Benchmark MAE ignores the model slicer (approved specification; see the B4 notes on the literal DAX).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import measures_forecast as mf
from .data_loader import Dataset, _resolve_dir, load_dataset
from .dims_forecast import ForecastDims

EMPTY_NOTE = ("No data for this basis / level combination. The no-stockout basis exists only at "
              "series x day level.")
MODEL_COLUMNS = ["model_name", "model_label", "observations", "mae", "rmse", "wape", "bias",
                 "benchmark_mae", "skill", "rank", "origins_won"]


@dataclass(frozen=True)
class ForecastFilters:
    """Selected KEYS: basis (evaluation_basis), level, model (model_name), origin (ISO date label).
    An empty tuple means All."""

    bases: tuple[str, ...] = ()
    levels: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastContext:
    metric: pd.DataFrame            # forecast_backtest_metric + origin_label
    dims: ForecastDims
    run: dict                       # the backtest_run row(s) summary
    n_runs: int
    basis_labels: dict[str, str]    # key -> label
    level_labels: dict[str, str]
    model_labels: dict[str, str]
    model_order: tuple[str, ...]
    origin_labels: tuple[str, ...]
    forecast_rows: int              # fact_demand_forecast rows (0 here)
    model_run_rows: int             # model_run rows (0 here)
    coverage_text: str


def build_forecast_context(ds: Dataset) -> ForecastContext:
    dims, run = ds.forecast_dims, ds["backtest_run"]
    metric = ds["forecast_backtest_metric"].copy()
    metric["origin_label"] = metric["origin_date"].dt.strftime("%Y-%m-%d")
    dd = ds["dim_date"]
    lo, hi = dd["full_date"].min().date(), dd["full_date"].max().date()
    r = run.iloc[0]
    return ForecastContext(
        metric=metric, dims=dims, n_runs=len(run),
        run={"backtest_run_id": r["backtest_run_id"] if len(run) == 1 else None,
             "n_series": int(run["n_series"].max()), "horizon_days": int(run["horizon_days"].max()),
             "censor_policy": r["censor_policy"] if run["censor_policy"].nunique() == 1 else None},
        basis_labels=dict(zip(dims.basis["evaluation_basis"], dims.basis["basis_label"])),
        level_labels=dict(zip(dims.level["level"], dims.level["level_label"])),
        model_labels=dict(zip(dims.model["model_name"], dims.model["model_label"])),
        model_order=tuple(dims.model["model_name"]),
        origin_labels=tuple(dims.origin["origin_label"]),
        forecast_rows=len(ds["fact_demand_forecast"]), model_run_rows=len(ds["model_run"]),
        coverage_text=f"Synthetic data loaded: {lo:%Y-%m-%d} to {hi:%Y-%m-%d}")


@lru_cache(maxsize=4)
def _context_for(resolved_dir: str) -> ForecastContext:
    return build_forecast_context(load_dataset(Path(resolved_dir)))


def load_forecast_context(data_dir: Path | str | None = None) -> ForecastContext:
    return _context_for(str(_resolve_dir(data_dir)))


def filters_from_labels(ctx: ForecastContext, bases=(), levels=(), models=(), origins=()) -> ForecastFilters:
    """Translate the labels shown in the slicers into keys."""
    inv = lambda d: {v: k for k, v in d.items()}
    return ForecastFilters(
        bases=tuple(inv(ctx.basis_labels)[x] for x in bases), levels=tuple(inv(ctx.level_labels)[x] for x in levels),
        models=tuple(inv(ctx.model_labels)[x] for x in models), origins=tuple(origins))


# --------------------------------------------------------------------------- scope and rows


def scope_of(f: ForecastFilters) -> tuple[str, str]:
    return mf.resolve_scope(f.bases, f.levels)


def scope_message(ctx: ForecastContext, f: ForecastFilters) -> str:
    """Measure 'Forecast Scope Message' (labels of the single selected values, else the defaults)."""
    level = ctx.level_labels[f.levels[0]] if len(f.levels) == 1 else "Series x day (default)"
    basis = ctx.basis_labels[f.bases[0]] if len(f.bases) == 1 else "Observed sales (default)"
    return f"Scope: {basis}  |  {level}  |  pooled over all selected origins"


def selected_models(ctx: ForecastContext, f: ForecastFilters) -> list[str]:
    return [m for m in ctx.model_order if not f.models or m in f.models]


def selected_origins(ctx: ForecastContext, f: ForecastFilters) -> list[str]:
    return [o for o in ctx.origin_labels if not f.origins or o in f.origins]


def rows_for(ctx: ForecastContext, f: ForecastFilters, basis: str, level: str,
             models: list[str] | None = None, origins: list[str] | None = None) -> pd.DataFrame:
    """Metric rows of one basis / level for the given models and origins."""
    m = ctx.metric
    mask = (m["evaluation_basis"] == basis) & (m["level"] == level)
    if models is not None:
        mask &= m["model_name"].isin(models)
    if origins is not None:
        mask &= m["origin_label"].isin(origins)
    return m[mask]


def benchmark_mae(ctx: ForecastContext, f: ForecastFilters, basis: str, level: str):
    return mf.mae(mf.sums_of(rows_for(ctx, f, basis, level, [mf.BENCHMARK_MODEL], selected_origins(ctx, f))))


# --------------------------------------------------------------------------- results


def model_comparison(ctx: ForecastContext, f: ForecastFilters) -> pd.DataFrame:
    """'Model comparison' table: one row per selected model with data, sorted by MAE ascending."""
    basis, level = scope_of(f)
    models, origins = selected_models(ctx, f), selected_origins(ctx, f)
    bench = benchmark_mae(ctx, f, basis, level)
    per_model = {m: mf.sums_of(rows_for(ctx, f, basis, level, [m], origins)) for m in models}
    maes = {m: mf.mae(s) for m, s in per_model.items()}
    rows = []
    for m in models:
        s = per_model[m]
        if s is None:
            continue
        won = 0
        for o in origins:
            mine = mf.mae(mf.sums_of(rows_for(ctx, f, basis, level, [m], [o])))
            best = [v for v in (mf.mae(mf.sums_of(rows_for(ctx, f, basis, level, [x], [o]))) for x in models)
                    if v is not None]
            if mine is not None and best and mine == min(best):
                won += 1
        rows.append({"model_name": m, "model_label": ctx.model_labels[m], "observations": s.n_obs,
                     "mae": maes[m], "rmse": mf.rmse(s), "wape": mf.wape(s), "bias": mf.bias(s),
                     "benchmark_mae": bench, "skill": mf.skill_vs_benchmark(maes[m], bench),
                     "rank": mf.dense_rank_ascending(maes[m], maes.values()), "origins_won": won})
    out = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    return out.sort_values(["mae", "model_label"], kind="stable", na_position="last").reset_index(drop=True)


def mae_by_origin(ctx: ForecastContext, f: ForecastFilters) -> pd.DataFrame:
    """'Is the ranking stable?': MAE of every selected model at every selected origin (blank skipped)."""
    basis, level = scope_of(f)
    rows = []
    for m in selected_models(ctx, f):
        for o in selected_origins(ctx, f):
            v = mf.mae(mf.sums_of(rows_for(ctx, f, basis, level, [m], [o])))
            if v is not None:
                rows.append({"model_name": m, "model_label": ctx.model_labels[m], "origin_label": o, "mae": v})
    return pd.DataFrame(rows, columns=["model_name", "model_label", "origin_label", "mae"])


def best_by_basis_level(ctx: ForecastContext, f: ForecastFilters) -> pd.DataFrame:
    """'Best model by evaluation basis and level': one row per basis x level that has data.

    The slicers on basis / level limit the rows; each row is evaluated in its own basis and level.
    """
    models, origins = selected_models(ctx, f), selected_origins(ctx, f)
    rows = []
    for b in ctx.dims.basis["evaluation_basis"]:
        for lv in ctx.dims.level["level"]:
            if (f.bases and b not in f.bases) or (f.levels and lv not in f.levels):
                continue
            maes = {m: mf.mae(mf.sums_of(rows_for(ctx, f, b, lv, [m], origins))) for m in models}
            present = {m: v for m, v in maes.items() if v is not None}
            if not present:
                continue
            best = min(present.values())
            bench = benchmark_mae(ctx, f, b, lv)
            skills = [mf.skill_vs_benchmark(v, bench) for v in present.values()]
            rows.append({
                "basis_label": ctx.basis_labels[b], "level_label": ctx.level_labels[lv],
                "best_model": min(ctx.model_labels[m] for m, v in present.items() if v == best),
                "best_mae": best, "benchmark_mae": bench,
                "best_vs_benchmark": mf.skill_vs_benchmark(best, bench),
                "models_beating": sum(1 for s in skills if s is not None and s > 0)})
    return pd.DataFrame(rows, columns=["basis_label", "level_label", "best_model", "best_mae",
                                       "benchmark_mae", "best_vs_benchmark", "models_beating"])


def cards(ctx: ForecastContext, f: ForecastFilters) -> dict:
    basis, level = scope_of(f)
    pooled = mf.sums_of(rows_for(ctx, f, basis, level, selected_models(ctx, f), selected_origins(ctx, f)))
    return {
        "run": ctx.run["backtest_run_id"] if ctx.n_runs == 1 else "multiple runs",
        "series_scored": ctx.run["n_series"], "rolling_origins": len(selected_origins(ctx, f)),
        "horizon_days": ctx.run["horizon_days"], "stockout_policy": ctx.run["censor_policy"] or "n/a",
        "scope_message": scope_message(ctx, f),
        "empty_note": EMPTY_NOTE if mf.observations(pooled) is None else None}
