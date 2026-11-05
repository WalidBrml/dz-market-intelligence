"""Page 4 - Forecasting Performance (backtest accuracy). Mirrors the Power BI page of the same name.

Everything here is historical BACKTEST accuracy. No future forecast exists in the dataset and none is shown.
"""

from __future__ import annotations

import streamlit as st

from dzmi_app import charts_forecast, components, theme, ui_tables
from dzmi_app.forecast_filters_ui import forecast_filter_bar
from dzmi_app.forecasting import (
    best_by_basis_level, cards, load_forecast_context, mae_by_origin, model_comparison,
)
from dzmi_app.formatting import fmt_number, fmt_percent
from dzmi_app.formatting_extra import fmt_decimal, fmt_signed_percent

PAGE = "p4"
TITLE = "Forecasting Performance"
READ_WITH_CARE = ("The best model differs by level and origin. Only a few origins were scored on a small synthetic "
                  "sample, so small gaps are not evidence that one method is better in general. Results describe "
                  "the simulator, not Algerian retail.")

MODEL_TABLE_COLUMNS = [
    ("model_label", "Model", str),
    ("observations", "Forecast Observations", ui_tables.fmt_int),
    ("mae", "Forecast MAE", fmt_decimal),
    ("rmse", "Forecast RMSE", fmt_decimal),
    ("wape", "Forecast WAPE", ui_tables.fmt_pct1),
    ("bias", "Forecast Bias", fmt_signed_percent),
    ("skill", "Skill vs Benchmark", fmt_signed_percent),
    ("rank", "MAE Rank", ui_tables.fmt_int),
    ("origins_won", "Origins Won (lowest MAE)", ui_tables.fmt_int),
]
BEST_TABLE_COLUMNS = [
    ("basis_label", "Evaluation basis", str), ("level_label", "Aggregation level", str),
    ("best_model", "Best Model (MAE)", str), ("best_mae", "Best MAE", fmt_decimal),
    ("benchmark_mae", "Benchmark MAE", fmt_decimal), ("best_vs_benchmark", "Best vs Benchmark", fmt_signed_percent),
    ("models_beating", "Models Beating Benchmark", ui_tables.fmt_int),
]


def _cards(c: dict) -> None:
    cols = st.columns(5, gap="small")
    values = [("Run", str(c["run"])), ("Series scored", fmt_number(c["series_scored"], 0)),
              ("Rolling origins", fmt_number(c["rolling_origins"], 0)),
              ("Horizon (days)", fmt_number(c["horizon_days"], 0)), ("Stockout days", str(c["stockout_policy"]))]
    for col, (label, value) in zip(cols, values):
        with col:
            components.kpi(label, value)
    components.section("Scope in use", c["scope_message"])
    if c["empty_note"]:
        st.warning(c["empty_note"])


def _model_row(ctx, f):
    left, right = st.columns([1.45, 1.0], gap="small")
    table = model_comparison(ctx, f)
    with left:
        st.markdown(ui_tables.matrix_html("Model comparison (lower MAE / RMSE / WAPE is better)",
                                          MODEL_TABLE_COLUMNS, table.to_dict("records"), None),
                    unsafe_allow_html=True)
    with right:
        labels = list(table["model_label"])
        fig = charts_forecast.grouped_bar_chart(
            labels, [("Forecast MAE", list(table["mae"]), theme.PRIMARY),
                     ("Benchmark MAE", list(table["benchmark_mae"]), theme.NAVY)],
            "MAE by model vs the seasonal-naive benchmark (lower is better)", height=380)
        components.show_chart(fig, "p4_chart_mae_benchmark")


def _best_and_origin_row(ctx, f):
    left, right = st.columns([1.2, 1.0], gap="small")
    with left:
        best = best_by_basis_level(ctx, f)
        st.markdown(ui_tables.matrix_html(
            "Best model by evaluation basis and level (clear the slicers to see all rows)",
            BEST_TABLE_COLUMNS, best.to_dict("records"), None), unsafe_allow_html=True)
    with right:
        by_origin = mae_by_origin(ctx, f)
        series = []
        for name in ctx.model_order:
            part = by_origin[by_origin["model_name"] == name]
            if len(part):
                values = {o: v for o, v in zip(part["origin_label"], part["mae"])}
                series.append((ctx.model_labels[name], [values.get(o) for o in ctx.origin_labels
                               if o in set(by_origin["origin_label"])], charts_forecast.MODEL_COLORS[name]))
        x = [o for o in ctx.origin_labels if o in set(by_origin["origin_label"])]
        components.show_chart(charts_forecast.multi_line_chart(
            x, series, "Is the ranking stable? MAE by forecast origin", tickformat=".1f"), "p4_chart_origin")


def render() -> None:
    ctx = load_forecast_context()
    components.inject_css()
    st.markdown(f"<style>{ui_tables.TABLE_CSS}</style>", unsafe_allow_html=True)
    components.page_header(TITLE, ctx.coverage_text)
    components.show_banner()
    f = forecast_filter_bar(PAGE, ctx)
    _cards(cards(ctx, f))
    _model_row(ctx, f)
    _best_and_origin_row(ctx, f)
    st.info("**Read with care** - " + READ_WITH_CARE)
