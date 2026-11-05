"""Page 2 - Sales Intelligence. Mirrors the Power BI page of the same name."""

from __future__ import annotations

import streamlit as st

from dzmi_app import charts, charts_sales, components, components_tables, state, theme, ui_tables
from dzmi_app.formatting import fmt_millions, fmt_number, fmt_percent
from dzmi_app.sales_intelligence import (
    SalesContext, SalesFilters, SalesResult, compute_sales_intelligence, load_sales_context,
)

PAGE = "p2"
FILTER_NAMES = ("date", "wilaya", "category", "brand")
TITLE = "Sales Intelligence"

MATRIX_COLUMNS = [
    ("year_month_label", "Month", str),
    ("net_sales", "Net Sales", ui_tables.fmt_int),
    ("mom", "Net Sales MoM %", ui_tables.fmt_pct1),
    ("gross_margin", "Gross Margin %", ui_tables.fmt_pct1),
    ("avg_daily_net_sales", "Average Daily Net Sales", ui_tables.fmt_int),
    ("units_sold", "Units Sold", ui_tables.fmt_int),
]


def _filter_bar(ctx: SalesContext) -> SalesFilters:
    with st.container(border=True):
        c_date, c_wil, c_cat, c_brand, c_reset = st.columns(
            [1.25, 1.0, 1.1, 1.0, 0.6], gap="small", vertical_alignment="bottom")
        with c_date:
            start, end = state.date_range(PAGE, "date", "Date range", ctx.date_min, ctx.date_max)
        with c_wil:
            wilayas = state.multiselect(PAGE, "wilaya", "Wilaya", ctx.wilayas)
        with c_cat:
            categories = state.multiselect(PAGE, "category", "Category", ctx.categories)
        with c_brand:
            brands = state.multiselect(PAGE, "brand", "Brand", ctx.brands)
        with c_reset:
            st.button("Reset filters", key=f"{PAGE}_reset", on_click=state.reset,
                      args=(PAGE, FILTER_NAMES))
    return SalesFilters(date_start=start, date_end=end, wilayas=wilayas,
                        categories=categories, brands=brands)


def _dzd(v: float) -> str:
    return f"{fmt_number(v, 0)} DZD"


def _bar_height(n: int) -> int:
    return max(260, 26 * n + 80)


def _row_daily_and_monthly(r: SalesResult) -> None:
    left, right = st.columns([0.9, 1.3], gap="small")
    with left:
        d = r.daily
        fig = charts_sales.line_chart(
            d["full_date"], [("Net Sales", d["net_sales"], theme.BLUE),
                             ("Gross Profit", d["gross_profit"], theme.PRIMARY)],
            "Daily net sales and gross profit (DZD)", height=360)
        components.show_chart(fig, "p2_chart_daily")
    with right:
        rows = r.monthly.to_dict("records")
        html = ui_tables.matrix_html(
            "Monthly performance - MoM only when both months are complete",
            MATRIX_COLUMNS, rows, {**r.monthly_total, "year_month_label": "Total"})
        st.markdown(html, unsafe_allow_html=True)


def _row_bars(r: SalesResult) -> None:
    c1, c2, c3, c4 = st.columns(4, gap="small")
    h = _bar_height(len(r.by_category))
    with c1:
        d = r.by_category
        components.show_chart(charts.hbar_chart(
            d["category_name"], d["net_sales"], [fmt_millions(v, 2) for v in d["net_sales"]],
            [_dzd(v) for v in d["net_sales"]], "Net sales by category", height=h,
            colors=theme.BLUE), "p2_chart_category")
    with c2:
        d = r.margin_by_category
        components.show_chart(charts.hbar_chart(
            d["category_name"], d["gross_margin"], [fmt_percent(v, 1) for v in d["gross_margin"]],
            [fmt_percent(v, 1) for v in d["gross_margin"]], "Gross margin by category", height=h,
            colors=theme.PRIMARY, tickformat=".0%"), "p2_chart_margin")
    with c3:
        d = r.per_sqm_by_store
        components.show_chart(charts.hbar_chart(
            d["store_name"], d["net_sales_per_sqm"], [fmt_number(v, 0) for v in d["net_sales_per_sqm"]],
            [f"{fmt_number(v, 0)} DZD per sqm" for v in d["net_sales_per_sqm"]],
            "Net sales per sqm by store (period total)", height=h, colors=theme.NAVY),
            "p2_chart_sqm")
    with c4:
        d = r.by_household
        components.show_chart(charts.hbar_chart(
            d["household_type"], d["net_sales"], [fmt_millions(v, 2) for v in d["net_sales"]],
            [_dzd(v) for v in d["net_sales"]],
            "Net sales by household type (blank = anonymous basket)", height=h, colors=theme.BLUE),
            "p2_chart_household")


def _row_products_and_weekday(r: SalesResult) -> None:
    left, right = st.columns([1.6, 1.0], gap="small")
    with left:
        st.markdown('<div class="dz-table-title">Products ranked by net sales</div>', unsafe_allow_html=True)
        t = r.products_total
        st.markdown(ui_tables.total_strip_html([
            ("Net Sales", fmt_number(t["net_sales"], 0) if t["net_sales"] is not None else ""),
            ("Gross Profit", fmt_number(t["gross_profit"], 0) if t["gross_profit"] is not None else ""),
            ("Gross Margin %", fmt_percent(t["gross_margin"], 1) if t["gross_margin"] is not None else ""),
            ("Average Selling Price",
             fmt_number(t["avg_selling_price"], 2) if t["avg_selling_price"] is not None else ""),
            ("Units Sold", fmt_number(t["units_sold"], 0) if t["units_sold"] is not None else ""),
            ("Promo Sales Share",
             fmt_percent(t["promo_share"], 1) if t["promo_share"] is not None else ""),
        ]), unsafe_allow_html=True)
        components_tables.show_dataframe(ui_tables.product_styler(r.products), key="p2_products", height=380)
    with right:
        d = r.by_weekday
        components.show_chart(charts.column_chart(
            d["day_name"], d["net_sales"], [fmt_millions(v, 2) for v in d["net_sales"]],
            [_dzd(v) for v in d["net_sales"]], "Net sales by weekday (Fri-Sat = weekend)",
            height=440, color=theme.BLUE), "p2_chart_weekday")


def _notes() -> None:
    with st.expander("About this page"):
        st.markdown(
            "**Definitions** (ported from the Power BI measures): Gross Margin % = Gross Profit / Net "
            "Sales; Average Selling Price = Net Sales / Units Sold; Promo Sales Share = net sales of "
            "lines with a promotion / Net Sales; Average Daily Net Sales = Net Sales / days with "
            "sales; Net Sales per sqm = Net Sales / store floor area.\n\n"
            "**Month-over-month** compares a month with the previous month and is shown only when "
            "both are complete months (May 2025 holds a single day, so it never shows a MoM %). "
            "The previous month is not limited by the date range. The matrix is sorted by MoM % "
            "descending, blanks last, as in the report.\n\n"
            "**Filters:** Wilaya is the wilaya of the *store*; Category and Brand come from the "
            "product. Lines without a customer appear as (Blank) in the household chart.\n\n"
            "**Intentional differences from the Power BI report:** all filters start at All and the "
            "page keeps its own filters; the date range is a range picker; the product grid can be "
            "re-sorted by clicking a column header; the matrix header 'year_month_label' reads "
            "'Month'.")


def render() -> None:
    ctx = load_sales_context()
    components.inject_css()
    st.markdown(f"<style>{ui_tables.TABLE_CSS}</style>", unsafe_allow_html=True)
    components.page_header(TITLE, ctx.coverage_text)
    components.show_banner()
    filters = _filter_bar(ctx)
    result = compute_sales_intelligence(ctx, filters)
    if result.sales_rows == 0:
        st.info("No sales match the current filters.")
    _row_daily_and_monthly(result)
    _row_bars(result)
    _row_products_and_weekday(result)
    _notes()
