"""HTML tables for the Sales Intelligence page (pure strings, no Streamlit import).

* matrix_html: the 'Monthly performance' matrix with its grand-total row.
* total_strip_html: the grand total of the product table, shown above the sortable grid.
* product_styler: number formats of the product table (a pandas Styler, display only: the
  underlying values stay numeric so Streamlit's column sorting still works).

Blank values are shown as empty cells, as Power BI does in matrices and tables.
"""

from __future__ import annotations

from html import escape
from typing import Callable, Sequence

import pandas as pd

from . import theme
from .formatting import fmt_number, fmt_percent, is_blank

TABLE_CSS = (
    f".dz-table{{width:100%;border-collapse:collapse;font-size:.85rem;color:{theme.TEXT}}}"
    f".dz-table th{{text-align:right;font-weight:600;color:{theme.GREY};padding:.45rem .6rem;"
    f"border-bottom:2px solid {theme.BORDER};white-space:nowrap}}"
    f".dz-table th:first-child,.dz-table td:first-child{{text-align:left}}"
    f".dz-table td{{text-align:right;padding:.4rem .6rem;border-bottom:1px solid {theme.GRID};"
    "font-variant-numeric:tabular-nums}"
    f".dz-table tr.dz-total td{{font-weight:700;color:{theme.NAVY};background:{theme.SURFACE};"
    f"border-top:2px solid {theme.BORDER}}}"
    f".dz-table-wrap{{background:{theme.WHITE};border:1px solid {theme.BORDER};border-radius:12px;"
    "padding:.4rem .6rem .6rem;overflow-x:auto}"
    f".dz-table-title{{font-size:.9rem;font-weight:700;color:{theme.NAVY};padding:.4rem .6rem .2rem}}"
    f".dz-strip{{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;background:{theme.SURFACE};"
    f"border:1px solid {theme.BORDER};border-radius:10px;padding:.5rem .8rem;margin-bottom:.5rem;"
    f"font-size:.8rem;color:{theme.GREY}}}"
    f".dz-strip b{{color:{theme.NAVY}}}"
)

Column = tuple[str, str, Callable[[object], str]]     # (key, header, formatter)


def _cell(value, formatter: Callable[[object], str]) -> str:
    return "" if is_blank(value) else formatter(value)


def matrix_html(title: str, columns: Sequence[Column], rows: Sequence[dict], total: dict | None) -> str:
    head = "".join(f"<th>{escape(label)}</th>" for _, label, _ in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_cell(r.get(key), fmt))}</td>" for key, _, fmt in columns) + "</tr>"
        for r in rows)
    foot = ""
    if total is not None:
        foot = '<tr class="dz-total">' + "".join(
            f"<td>{escape(_cell(total.get(key), fmt))}</td>" for key, _, fmt in columns) + "</tr>"
    return (f'<div class="dz-table-wrap"><div class="dz-table-title">{escape(title)}</div>'
            f'<table class="dz-table"><thead><tr>{head}</tr></thead><tbody>{body}{foot}</tbody></table></div>')


def total_strip_html(items: Sequence[tuple[str, str]]) -> str:
    inner = "".join(f"<span>{escape(label)} <b>{escape(value)}</b></span>" for label, value in items)
    return f'<div class="dz-strip"><span><b>Total</b></span>{inner}</div>'


# Display names of the product-table columns (measure names as in Power BI).
PRODUCT_HEADERS = {
    "product_name": "Product", "category_name": "Category", "net_sales": "Net Sales",
    "gross_profit": "Gross Profit", "gross_margin": "Gross Margin %",
    "avg_selling_price": "Average Selling Price", "units_sold": "Units Sold",
    "promo_share": "Promo Sales Share",
}


def product_styler(products: pd.DataFrame):
    """Display-formatted view of the product table (formats from the model's format strings)."""
    shown = products.rename(columns=PRODUCT_HEADERS)
    return shown.style.format({
        "Net Sales": "{:,.0f}", "Gross Profit": "{:,.0f}", "Gross Margin %": "{:.1%}",
        "Average Selling Price": "{:,.2f}", "Units Sold": "{:,.0f}", "Promo Sales Share": "{:.1%}",
    }, na_rep="")


def fmt_int(x) -> str:
    return fmt_number(x, 0)


def fmt_pct1(x) -> str:
    return fmt_percent(x, 1)
