"""Page 3 - Inventory Intelligence. Mirrors the Power BI page of the same name."""

from __future__ import annotations

import streamlit as st

from dzmi_app import charts, charts_inventory, components, components_tables, state, theme, ui_styles, ui_tables
from dzmi_app.formatting import fmt_count, fmt_number, fmt_percent
from dzmi_app.formatting_extra import fmt_percent_spaced
from dzmi_app.inventory_intelligence import (
    InventoryContext, InventoryFilters, InventoryResult, compute_inventory_intelligence, load_inventory_context,
)

PAGE = "p3"
FILTER_NAMES = ("date", "store", "category", "product", "tier")
TITLE = "Inventory Intelligence"


def _blank(text: str) -> str:
    return text if text else "\u2014"


def _filter_bar(ctx: InventoryContext) -> InventoryFilters:
    with st.container(border=True):
        c_date, c_store, c_cat, c_prod, c_tier, c_reset = st.columns(
            [1.2, 1.0, 1.0, 1.3, 0.9, 0.6], gap="small", vertical_alignment="bottom")
        with c_date:
            start, end = state.date_range(PAGE, "date", "Date range", ctx.date_min, ctx.date_max)
        with c_store:
            stores = state.multiselect(PAGE, "store", "Store", ctx.stores)
        with c_cat:
            categories = state.multiselect(PAGE, "category", "Category", ctx.categories)
        with c_prod:
            products = state.multiselect(PAGE, "product", "Product", ctx.products)
        with c_tier:
            tiers = state.multiselect(PAGE, "tier", "Risk tier (table below)", ctx.tiers)
        with c_reset:
            st.button("Reset filters", key=f"{PAGE}_reset", on_click=state.reset, args=(PAGE, FILTER_NAMES))
    return InventoryFilters(date_start=start, date_end=end, stores=stores, categories=categories,
                            products=products, tiers=tiers)


def _cards(r: InventoryResult, ctx: InventoryContext) -> None:
    last = f"{r.last_date:%Y-%m-%d}" if r.last_date else ""
    cols = st.columns(7, gap="small")
    with cols[0]:
        components.kpi("Stockout rate", _blank(fmt_percent(r.stockout_rate, 2) if r.stockout_rate is not None else ""),
                       tone="warn")
    with cols[1]:
        components.kpi("Availability rate",
                       _blank(fmt_percent(r.availability_rate, 2) if r.availability_rate is not None else ""))
    with cols[2]:
        components.kpi("Stockout SKU-days", fmt_count(r.stockout_sku_days), f"of {fmt_count(r.sku_days)} SKU-days",
                       tone="danger")
    with cols[3]:
        components.kpi("Closing stock (last day)",
                       _blank(fmt_number(r.closing_stock, 0) if r.closing_stock is not None else ""),
                       last, tone="neutral")
    with cols[4]:
        components.kpi("Days of cover - overall",
                       _blank(fmt_number(r.overall_days_of_cover, 1) if r.overall_days_of_cover is not None else ""),
                       "stock / avg daily units, 14 days to the last day", tone="info")
    with cols[5]:
        snap = f"{ctx.snapshot_date:%Y-%m-%d}" if ctx.snapshot_date else ""
        components.kpi("Days of cover - median SKU",
                       _blank(fmt_number(r.median_days_of_cover, 1) if r.median_days_of_cover is not None else ""),
                       f"snapshot {snap}, ignores the date range", tone="info")
    with cols[6]:
        components.kpi("At-risk SKUs (latest day)",
                       _blank(fmt_percent_spaced(r.at_risk_share, 1)), "Stockout + Critical + Low", tone="warn")


def _charts(r: InventoryResult) -> None:
    left, right = st.columns([1.0, 1.25], gap="small")
    with left:
        d = r.daily
        components.show_chart(charts_inventory.rate_line_chart(
            d["full_date"], d["stockout_rate"],
            "Daily stockout rate (share of SKU-days with zero closing stock)", height=420), "p3_chart_daily")
    with right:
        h = r.heatmap
        components.show_chart(charts_inventory.rate_heatmap(
            list(h.index), list(h.columns), h.to_numpy(), "Stockout rate: category x store", height=420),
            "p3_chart_heatmap")


def _tiers_and_table(r: InventoryResult) -> None:
    left, right = st.columns([0.8, 1.6], gap="small")
    with left:
        t = r.tiers
        components.show_chart(charts.hbar_chart(
            t["risk_tier"], t["skus"], [fmt_count(v) for v in t["skus"]], [f"{fmt_count(v)} SKUs" for v in t["skus"]],
            "SKUs by risk tier (latest day)", height=380, colors=[theme.TIER_COLORS[x] for x in t["risk_tier"]],
            tickformat="d"), "p3_chart_tiers")
    with right:
        st.markdown('<div class="dz-table-title">SKU risk list, most urgent tier first (latest day)</div>',
                    unsafe_allow_html=True)
        components_tables.show_dataframe(ui_styles.risk_table_styler(r.risk_table), key="p3_risk_table", height=340)


def _notes() -> None:
    with st.expander("About this page"):
        st.markdown(
            "**Snapshot rule:** closing stock is read on the last day of the date range only and is never summed "
            "across days. The risk tiers, the SKU list and the median days of cover come from the latest-day "
            "snapshot and ignore the date range.\n\n"
            "**Two days-of-cover figures:** *overall* = closing stock on the last day / average daily units sold over "
            "the 14 days ending that day (the report's measure); *median SKU* = the median of the per-SKU values in "
            "the snapshot.\n\n"
            "**Intentional differences from the Power BI report:** all filters start at All; the risk-tier filter "
            "applies to the SKU list only; availability uses the same filters as the stockout rate; the "
            "category x store table is a heat map; the tier chart shows the real SKU count per tier.")


def render() -> None:
    ctx = load_inventory_context()
    components.inject_css()
    st.markdown(f"<style>{ui_tables.TABLE_CSS}</style>", unsafe_allow_html=True)
    components.page_header(TITLE, ctx.coverage_text)
    components.show_banner()
    f = _filter_bar(ctx)
    r = compute_inventory_intelligence(ctx, f)
    if r.sku_days == 0:
        st.info("No inventory records match the current filters.")
    _cards(r, ctx)
    _charts(r)
    _tiers_and_table(r)
    _notes()
