"""Page 5 - Forecast Accuracy by Horizon. Mirrors the Power BI page of the same name (backtest only)."""

from __future__ import annotations

import streamlit as st

from dzmi_app import charts, charts_forecast, components, theme, ui_tables
from dzmi_app.forecast_filters_ui import forecast_filter_bar
from dzmi_app.forecast_horizon import compute_horizon
from dzmi_app.forecasting import load_forecast_context
from dzmi_app.formatting import fmt_number
from dzmi_app.formatting_extra import fmt_decimal, fmt_signed_percent

PAGE = "p5"
TITLE = "Forecast Accuracy by Horizon"


def _series(ctx, r, column):
    """One line per model over the horizon labels (blank = gap)."""
    labels = list(r.observations["horizon_label"])
    out = []
    for name in ctx.model_order:
        part = r.by_model[r.by_model["model_name"] == name]
        if len(part):
            values = dict(zip(part["horizon_label"], part[column]))
            out.append((ctx.model_labels[name], [values.get(h) for h in labels],
                        charts_forecast.MODEL_COLORS[name]))
    return labels, out


def render() -> None:
    ctx = load_forecast_context()
    components.inject_css()
    st.markdown(f"<style>{ui_tables.TABLE_CSS}</style>", unsafe_allow_html=True)
    components.page_header(TITLE, ctx.coverage_text)
    components.show_banner()
    f = forecast_filter_bar(PAGE, ctx)
    r = compute_horizon(ctx, f)
    components.section("Scope in use", r.scope_message)
    components.section("Horizon axis (read before comparing)", r.axis_title)
    if len(r.observations) == 0:
        st.warning("No data for this basis / level combination. The no-stockout basis exists only at "
                   "series x day level.")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        x, s = _series(ctx, r, "mae")
        components.show_chart(charts_forecast.multi_line_chart(
            x, s, "MAE by forecast horizon (lower is better)", tickformat=".1f"), "p5_chart_mae")
    with c2:
        x, s = _series(ctx, r, "wape")
        components.show_chart(charts_forecast.multi_line_chart(
            x, s, "WAPE by forecast horizon (lower is better)", tickformat=".0%", hover=".1%"), "p5_chart_wape")
    c3, c4 = st.columns(2, gap="small")
    with c3:
        x, s = _series(ctx, r, "bias")
        components.show_chart(charts_forecast.multi_line_chart(
            x, s, "Bias by horizon (0 = unbiased)", tickformat="+.0%", hover="+.1%", zero_line=True),
            "p5_chart_bias")
    with c4:
        o = r.observations
        components.show_chart(charts.column_chart(
            o["horizon_label"], o["observations"], [fmt_number(v, 0) for v in o["observations"]],
            [f"{fmt_number(v, 0)} observations (all selected models)" for v in o["observations"]],
            "Observations behind each point", height=340, color=theme.BLUE), "p5_chart_observations")

    pivot = r.mae_pivot
    columns = [("horizon_label", "Horizon", str)] + [(m, m, fmt_decimal) for m in pivot.columns]
    rows = [{"horizon_label": h, **{m: pivot.loc[h, m] for m in pivot.columns}} for h in pivot.index]
    st.markdown(ui_tables.matrix_html("MAE by horizon x model", columns, rows, None), unsafe_allow_html=True)
