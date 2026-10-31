"""Catalogue of the Milestone 2 analytical views and a safe way to read them.

The view names are an allow-list: a name typed on the command line is only ever pasted into SQL
if it is one of the keys below, so `dzmi query` cannot be used to run arbitrary SQL.
"""
from __future__ import annotations

import pandas as pd

from ..errors import DZMIError
from ..ingestion.db import Database

# name -> (grain, default sort columns). Keep the grain text in sync with docs/analytical_views.md
# (tests/test_analytics_docs.py checks that every view is documented with a Grain line).
ANALYTIC_VIEWS: dict[str, tuple[str, list[str]]] = {
    "v_kpi_network_daily": ("one row per calendar date, whole network", ["date_key"]),
    "v_kpi_network_monthly": ("one row per calendar month, whole network", ["year_month"]),
    "v_kpi_category_monthly": ("one row per category x calendar month", ["category_name", "year_month"]),
    "v_kpi_product": ("one row per product, whole loaded period", ["net_sales_rank"]),
    "v_kpi_store": ("one row per store, whole loaded period", ["store_key"]),
    "v_kpi_wilaya": ("one row per wilaya, whole loaded period", ["wilaya"]),
    "v_stock_availability_daily": ("one row per store x calendar date", ["store_key", "date_key"]),
    "v_inventory_risk": ("one row per store x product on the latest inventory date",
                         ["risk_tier_order", "store_key", "product_key"]),
    "v_dim_date_bi": ("one row per calendar date (Power BI date table)", ["date_key"]),
}


def list_views() -> list[tuple[str, str]]:
    return [(name, grain) for name, (grain, _) in ANALYTIC_VIEWS.items()]


def read_view(db: Database, name: str, catalog: dict | None = None) -> pd.DataFrame:
    """Return a whole analytical view as a DataFrame sorted by its default order.

    `catalog` is the allow-list (default: the Milestone 2 views); `dzmi query` also passes the Milestone 3 views.
    """
    catalog = ANALYTIC_VIEWS if catalog is None else catalog
    if name not in catalog:
        raise DZMIError(f"Unknown view '{name}'. Choose one of: {', '.join(catalog)}  (see `dzmi query --list`).")
    df = db.query_df(f"SELECT * FROM dwh.{name}")   # name is allow-listed above
    return df.sort_values(catalog[name][1], kind="mergesort").reset_index(drop=True)


def _fmt(v: float) -> str:
    return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"


def format_view(df: pd.DataFrame, limit: int | None = 20, order_by: str | None = None,
                descending: bool = False, columns: list[str] | None = None) -> str:
    """Console-friendly text for a view: optional re-sort, column selection and row limit."""
    if order_by:
        if order_by not in df.columns:
            raise DZMIError(f"Unknown column '{order_by}'. Columns: {', '.join(df.columns)}")
        df = df.sort_values(order_by, ascending=not descending, kind="mergesort", na_position="last")
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise DZMIError(f"Unknown column(s): {', '.join(missing)}. Columns: {', '.join(df.columns)}")
        df = df[columns]
    total = len(df)
    shown = df.head(limit) if limit else df
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 40,
                           "display.float_format", _fmt):
        text = shown.to_string(index=False)
    return f"{text}\n({len(shown):,} of {total:,} rows)"
