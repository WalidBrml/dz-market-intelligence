"""Page 6 - Data & Method Notes. Facts come from the loaded dataset (manifest and tables)."""

from __future__ import annotations

import streamlit as st

from dzmi_app import components, notes, ui_tables
from dzmi_app.data_loader import load_dataset
from dzmi_app.formatting import fmt_count, fmt_number
from dzmi_app.formatting_extra import fmt_decimal, fmt_text_or_blank
from dzmi_app.integrity import run_integrity_checks

TITLE = "Data & Method Notes"


@st.cache_resource(show_spinner="Reading the dataset...")
def _page_data():
    ds = load_dataset()
    results = run_integrity_checks(ds)
    return {"coverage": notes.coverage(ds), "files": notes.dataset_files(ds), "recon": notes.reconciliation(ds),
            "checks": notes.integrity_table(results), "coverage_text":
            f"Synthetic data loaded: {ds['dim_date']['full_date'].min():%Y-%m-%d} to "
            f"{ds['dim_date']['full_date'].max():%Y-%m-%d}"}


def _table(title, columns, frame):
    st.markdown(ui_tables.matrix_html(title, columns, frame.to_dict("records"), None), unsafe_allow_html=True)


def render() -> None:
    d = _page_data()
    c = d["coverage"]
    components.inject_css()
    st.markdown(f"<style>{ui_tables.TABLE_CSS}</style>", unsafe_allow_html=True)
    components.page_header(TITLE, d["coverage_text"])
    components.show_banner()

    for heading, text in notes.WHAT_THIS_REPORT_IS:
        components.section(heading)
        st.markdown(text)

    components.section("Dataset coverage", "Counts are read from the loaded tables.")
    cols = st.columns(6, gap="small")
    facts = [("Period", f"{c['first_date']:%Y-%m-%d} to {c['last_date']:%Y-%m-%d}", f"{c['days']} days"),
             ("Stores / products", f"{c['stores']} / {c['products']}", f"{c['categories']} categories"),
             ("Sales lines", fmt_count(c["sales_lines"]), f"{fmt_count(c['transactions'])} baskets"),
             ("Inventory SKU-days", fmt_count(c["inventory_sku_days"]), f"{c['tracked_skus']} tracked SKUs"),
             ("Risk snapshot", f"{c['snapshot_date']:%Y-%m-%d}", "latest loaded day"),
             ("Stored forecasts", f"{c['forecast_rows']} rows", f"model_run: {c['model_run_rows']} rows")]
    for col, (label, value, note) in zip(cols, facts):
        with col:
            components.kpi(label, value, note)

    components.section("Backtesting is not forecasting")
    for line in notes.backtest_vs_forecast_text(c):
        st.markdown("- " + line)

    components.section("Known limits")
    for line in notes.KNOWN_LIMITS:
        st.markdown("- " + line)

    _table("Exported files and row counts (bi_export_manifest.json)",
           [("table", "Table", str), ("rows", "Rows", fmt_count), ("status", "Status", fmt_text_or_blank),
            ("role", "Role", str)], d["files"])
    _table("Data integrity checks", [("check", "Check", str), ("description", "Description", str),
                                     ("kind", "Kind", str), ("result", "Result", str)], d["checks"])
    recon = d["recon"]
    ok = int((recon["status"] == "OK").sum())
    _table(f"MAE reconciliation: additive sums vs v_forecast_accuracy ({ok} of {len(recon)} OK)",
           [("model_label", "Model", str), ("basis_label", "Evaluation basis", str),
            ("level_label", "Aggregation level", str), ("forecast_mae", "Forecast MAE", fmt_decimal),
            ("sql_view_mae", "SQL View MAE", fmt_decimal),
            ("difference", "MAE Check Difference", lambda x: fmt_decimal(x, 6)),
            ("status", "MAE Check Status", fmt_text_or_blank)], recon)

    components.section("How the key measures are defined")
    st.markdown(ui_tables.matrix_html("Definitions (from the Power BI semantic model)",
                                      [("name", "Measure", str), ("text", "Definition", str)],
                                      [{"name": n, "text": t} for n, t in notes.DEFINITIONS], None),
                unsafe_allow_html=True)
    with st.expander("Data model relationships (relationships.tmdl)"):
        _table(f"{len(notes.RELATIONSHIPS)} many-to-one relationships",
               [("from (many)", "From (many)", str), ("to (one)", "To (one)", str)], notes.relationships_table())
    st.markdown(ui_tables.matrix_html("Intentional differences from the Power BI report",
                                      [("page", "Page", str), ("text", "Difference", str)],
                                      [{"page": p, "text": t} for p, t in notes.DEVIATIONS], None),
                unsafe_allow_html=True)
