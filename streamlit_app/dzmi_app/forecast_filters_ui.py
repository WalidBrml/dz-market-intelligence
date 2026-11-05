"""The four slicers shared by the forecast pages (each page keeps its own state through its prefix)."""

from __future__ import annotations

import streamlit as st

from . import state
from .forecasting import ForecastContext, ForecastFilters, filters_from_labels

FILTER_NAMES = ("basis", "level", "model", "origin")


def forecast_filter_bar(page: str, ctx: ForecastContext) -> ForecastFilters:
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.3, 1.1, 0.6], gap="small", vertical_alignment="bottom")
        with c1:
            bases = state.multiselect(page, "basis", "Evaluation basis (pick one)", list(ctx.basis_labels.values()))
        with c2:
            levels = state.multiselect(page, "level", "Aggregation level (pick one)", list(ctx.level_labels.values()))
        with c3:
            models = state.multiselect(page, "model", "Model", [ctx.model_labels[m] for m in ctx.model_order])
        with c4:
            origins = state.multiselect(page, "origin", "Forecast origin", list(ctx.origin_labels))
        with c5:
            st.button("Reset filters", key=f"{page}_reset", on_click=state.reset, args=(page, FILTER_NAMES))
    return filters_from_labels(ctx, bases, levels, models, origins)
