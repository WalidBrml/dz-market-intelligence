"""Forecast Accuracy by Horizon (Page 5): results by forecast horizon. No Streamlit here.

Same slicers, scope rules and measures as Forecasting Performance (forecasting.py), evaluated per
dim_horizon row (horizon_label, ordered by horizon_sort) and per model. Day buckets (series x day) and
week buckets (store x category x week) are never mixed: the scope fixes the level.

'Observations behind each point' is the literal measure: Forecast Observations summed over ALL
selected models at each horizon (it is not per model).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import measures_forecast as mf
from .forecasting import ForecastContext, ForecastFilters, rows_for, scope_message, scope_of, selected_models, selected_origins

AXIS_WEEK = "Forecast week after the origin (store x category x week level)"
AXIS_DAY = "Forecast day after the origin (series x day level)"


@dataclass(frozen=True)
class HorizonResult:
    axis_title: str
    scope_message: str
    by_model: pd.DataFrame      # horizon_label, horizon_sort, model_name, model_label, observations, mae, wape, bias
    observations: pd.DataFrame  # horizon_label, horizon_sort, observations (pooled over the selected models)
    mae_pivot: pd.DataFrame     # rows = horizon_label (horizon order), columns = model_label (model order)


def horizon_axis_title(f: ForecastFilters) -> str:
    return AXIS_WEEK if mf.resolve_scope(f.bases, f.levels)[1] == "store_category_week" else AXIS_DAY


def compute_horizon(ctx: ForecastContext, f: ForecastFilters) -> HorizonResult:
    basis, level = scope_of(f)
    models, origins = selected_models(ctx, f), selected_origins(ctx, f)
    rows = rows_for(ctx, f, basis, level, models, origins)
    horizon = ctx.dims.horizon[ctx.dims.horizon["level"] == level].set_index("horizon_bucket")
    records, pooled = [], []
    for bucket, label, sort in zip(horizon.index, horizon["horizon_label"], horizon["horizon_sort"]):
        at = rows[rows["horizon_bucket"] == bucket]
        s_all = mf.sums_of(at)
        if s_all is not None:
            pooled.append({"horizon_label": label, "horizon_sort": int(sort), "observations": s_all.n_obs})
        for m in models:
            s = mf.sums_of(at[at["model_name"] == m])
            if s is not None:
                records.append({"horizon_label": label, "horizon_sort": int(sort), "model_name": m,
                                "model_label": ctx.model_labels[m], "observations": s.n_obs,
                                "mae": mf.mae(s), "wape": mf.wape(s), "bias": mf.bias(s)})
    by_model = pd.DataFrame(records, columns=["horizon_label", "horizon_sort", "model_name", "model_label",
                                              "observations", "mae", "wape", "bias"])
    observations = pd.DataFrame(pooled, columns=["horizon_label", "horizon_sort", "observations"])
    if len(by_model):
        pivot = by_model.pivot(index="horizon_label", columns="model_label", values="mae")
        pivot = pivot.reindex(index=list(observations["horizon_label"]),
                              columns=[ctx.model_labels[m] for m in models if ctx.model_labels[m] in pivot.columns])
    else:
        pivot = pd.DataFrame()
    return HorizonResult(horizon_axis_title(f), scope_message(ctx, f), by_model, observations, pivot)
