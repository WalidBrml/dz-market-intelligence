"""Page 1 - Executive Overview. Mirrors the Power BI page of the same name."""

from __future__ import annotations

import streamlit as st

from dzmi_app import charts, components, state, theme
from dzmi_app.filters import OverviewFilters
from dzmi_app.formatting import (
    fmt_count, fmt_millions, fmt_number, fmt_percent, fmt_thousands,
)
from dzmi_app.overview import OverviewContext, OverviewResult, compute_overview, load_overview_context

PAGE = "p1"
FILTER_NAMES = ("date", "store", "category", "format")
TITLE = "Executive Overview"


def _filter_bar(ctx: OverviewContext) -> OverviewFilters:
    with st.container(border=True):
        c_date, c_store, c_cat, c_fmt, c_reset = st.columns(
            [1.25, 1.1, 1.1, 1.0, 0.6], gap="small", vertical_alignment="bottom")
        with c_date:
            start, end = state.date_range(PAGE, "date", "Date range", ctx.date_min, ctx.date_max)
        with c_store:
            stores = state.multiselect(PAGE, "store", "Store", ctx.stores)
        with c_cat:
            categories = state.multiselect(PAGE, "category", "Category", ctx.categories)
        with c_fmt:
            formats = state.multiselect(PAGE, "format", "Store format", ctx.store_formats)
        with c_reset:
            st.button("Reset filters", key=f"{PAGE}_reset", on_click=state.reset,
                      args=(PAGE, FILTER_NAMES))
    return OverviewFilters(date_start=start, date_end=end, stores=stores,
                           categories=categories, store_formats=formats)


def _kpi_rows(r: OverviewResult) -> None:
    k = r.kpis
    cols = st.columns(6, gap="small")
    with cols[0]:
        components.kpi("Net sales (M DZD)", fmt_millions(k.net_sales, 1))
    with cols[1]:
        components.kpi("Gross profit (M DZD)", fmt_millions(k.gross_profit, 1))
    with cols[2]:
        components.kpi("Gross margin", fmt_percent(k.gross_margin, 1))
    with cols[3]:
        components.kpi("Units sold (thousands)", fmt_thousands(k.units_sold, 1))
    with cols[4]:
        components.kpi("Average basket (DZD)", fmt_number(k.average_basket, 2))
    with cols[5]:
        components.kpi("Stockout rate (SKU-days)", fmt_percent(k.stockout_rate, 2))


def _dzd(v: float) -> str:
    return f"{fmt_number(v, 0)} DZD"


def _sales_charts(r: OverviewResult) -> None:
    c1, c2, c3 = st.columns([1.35, 1.0, 1.15], gap="small")
    with c1:
        m = r.monthly
        fig = charts.column_chart(
            m["year_month_label"], m["net_sales"], [fmt_millions(v, 2) for v in m["net_sales"]],
            [_dzd(v) for v in m["net_sales"]],
            "Monthly net sales (DZD) - May 2025 holds a single day")
        components.show_chart(fig, "p1_chart_monthly")
    with c2:
        s = r.by_store
        fig = charts.hbar_chart(
            s["store_name"], s["net_sales"], [fmt_millions(v, 2) for v in s["net_sales"]],
            [_dzd(v) for v in s["net_sales"]], "Net sales by store (DZD)", colors=theme.BLUE)
        components.show_chart(fig, "p1_chart_store")
    with c3:
        t = r.top_products
        fig = charts.hbar_chart(
            t["product_name"], t["net_sales"], [fmt_millions(v, 2) for v in t["net_sales"]],
            [_dzd(v) for v in t["net_sales"]], "Top 10 products by net sales (DZD)", colors=theme.NAVY)
        components.show_chart(fig, "p1_chart_top_products")


def _inventory_block(r: OverviewResult, ctx: OverviewContext) -> None:
    snap = f"{ctx.snapshot_date:%Y-%m-%d}" if ctx.snapshot_date else "n/a"
    components.section(
        "Inventory",
        f"SKU counts, tiers and stock value describe the latest loaded day ({snap}) and ignore "
        "the date range. The stockout rate by store follows the date range.")
    k = r.kpis
    left, mid, right = st.columns([1.1, 1.25, 1.45], gap="small")
    with left:
        a, b = st.columns(2, gap="small")
        with a:
            components.kpi("Stockout now (SKUs)", fmt_count(k.stockout_now), tone="danger")
        with b:
            components.kpi("Critical (<3 days cover)", fmt_count(k.critical), tone="warn")
        a, b = st.columns(2, gap="small")
        with a:
            components.kpi("Low (3-7 days)", fmt_count(k.low), tone="info")
        with b:
            components.kpi("Stock value at cost (M DZD)", fmt_millions(k.stock_value, 2), tone="neutral")
    with mid:
        t = r.tiers
        fig = charts.hbar_chart(
            t["risk_tier"], t["skus"], [fmt_count(v) for v in t["skus"]],
            [f"{fmt_count(v)} SKUs" for v in t["skus"]],
            "Inventory risk tiers - latest day (ignores the date slicer)", height=300,
            colors=[theme.TIER_COLORS[x] for x in t["risk_tier"]], tickformat="d")
        components.show_chart(fig, "p1_chart_tiers")
    with right:
        s = r.stockout_by_store
        fig = charts.hbar_chart(
            s["store_name"], s["stockout_rate"], [fmt_percent(v, 2) for v in s["stockout_rate"]],
            [fmt_percent(v, 2) for v in s["stockout_rate"]], "Stockout rate by store", height=300,
            colors=theme.RED, tickformat=".1%")
        components.show_chart(fig, "p1_chart_stockout_store")


def _notes() -> None:
    with st.expander("About this page"):
        st.markdown(
            "**Definitions** (ported from the Power BI measures): net sales, gross profit, units and "
            "transactions are sums and distinct counts over `fact_sales`; gross margin = gross profit / "
            "net sales; average basket = net sales / transactions; stockout rate = SKU-days with zero "
            "closing stock / SKU-days.\n\n"
            "**Snapshot rule:** the risk tiers, SKU counts and stock value come from the latest-day "
            "snapshot (`v_inventory_risk`) and are never summed across days.\n\n"
            "**Intentional differences from the Power BI report:** all filters start at All and each "
            "page keeps its own filters; the date range is a range picker instead of a list of single "
            "dates; the risk-tier chart shows the real number of SKUs per tier (the original drew the "
            "same total on every bar); Healthy, Overstock and No recent sales get colours; blank "
            "values show as an em dash and blank SKU counts as 0.")


def render() -> None:
    ctx = load_overview_context()
    components.inject_css()
    components.page_header(TITLE, ctx.coverage_text)
    components.show_banner()
    filters = _filter_bar(ctx)
    result = compute_overview(ctx, filters)
    if result.sales_rows == 0:
        st.info("No sales match the current filters.")
    _kpi_rows(result)
    _sales_charts(result)
    _inventory_block(result, ctx)
    _notes()
