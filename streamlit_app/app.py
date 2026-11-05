"""Entry point: DZ Market Intelligence (read-only Streamlit companion to the Power BI report).

Run from the project root:  streamlit run streamlit_app/app.py

All six pages are implemented (Milestones B1 to B6), in the same order as the Power BI report.
"""

import streamlit as st

from dzmi_app import components
from dzmi_app.data_loader import load_dataset
from dzmi_app.integrity import failed_errors, run_integrity_checks
from pages.p1_executive_overview import render as render_executive_overview
from pages.p2_sales_intelligence import render as render_sales_intelligence
from pages.p3_inventory_intelligence import render as render_inventory_intelligence
from pages.p4_forecasting_performance import render as render_forecasting_performance
from pages.p5_forecast_accuracy_by_horizon import render as render_forecast_accuracy_by_horizon
from pages.p6_data_and_method_notes import render as render_data_and_method_notes

st.set_page_config(page_title="DZ Market Intelligence", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Loading and checking the dataset...")
def _integrity_results():
    return run_integrity_checks(load_dataset())


results = _integrity_results()
blocking = failed_errors(results)
if blocking:
    components.inject_css()
    components.show_banner()
    st.error("The dataset failed blocking integrity checks, so the app is stopped:")
    for r in blocking:
        st.write(f"{r.check_id} - {r.name}: {r.detail}")
    st.stop()

st.sidebar.caption("Dataset: smoke_m3_20261025 (synthetic)")
st.sidebar.caption(f"Data integrity: {sum(r.passed for r in results)}/{len(results)} checks passed")

navigation = st.navigation([
    st.Page(render_executive_overview, title="Executive Overview", default=True),
    st.Page(render_sales_intelligence, title="Sales Intelligence", url_path="sales-intelligence"),
    st.Page(render_inventory_intelligence, title="Inventory Intelligence",
            url_path="inventory-intelligence"),
    st.Page(render_forecasting_performance, title="Forecasting Performance",
            url_path="forecasting-performance"),
    st.Page(render_forecast_accuracy_by_horizon, title="Forecast Accuracy by Horizon",
            url_path="forecast-accuracy-by-horizon"),
    st.Page(render_data_and_method_notes, title="Data & Method Notes",
            url_path="data-and-method-notes"),
])
navigation.run()
