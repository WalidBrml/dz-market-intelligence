"""Executive Overview (Page 1): context building and result computation. No Streamlit here.

Filter semantics follow the Power BI relationships (relationships.tmdl):

* Date range  -> dim_date; reaches fact_sales and fact_inventory_daily, never v_inventory_risk.
* Store / Store format -> dim_store; reaches the sales facts, the inventory facts and the risk view.
* Category    -> dim_product; reaches the same three tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import measures_inventory as inv
from . import measures_sales as sales_m
from .data_loader import Dataset, load_dataset
from .filters import OverviewFilters


@dataclass(frozen=True)
class OverviewContext:
    """Joined, read-only working frames plus the slicer option lists."""

    sales: pd.DataFrame
    inventory: pd.DataFrame
    risk: pd.DataFrame
    date_min: date
    date_max: date
    snapshot_date: date | None
    stores: tuple[str, ...]
    categories: tuple[str, ...]
    store_formats: tuple[str, ...]
    coverage_text: str


@dataclass(frozen=True)
class OverviewKpis:
    net_sales: float | None
    gross_profit: float | None
    gross_margin: float | None
    units_sold: int | None
    average_basket: float | None
    stockout_rate: float | None
    stockout_now: int
    critical: int
    low: int
    stock_value: float | None


@dataclass(frozen=True)
class OverviewResult:
    kpis: OverviewKpis
    monthly: pd.DataFrame            # year_month_label, net_sales
    by_store: pd.DataFrame           # store_name, net_sales
    top_products: pd.DataFrame       # product_name, net_sales
    tiers: pd.DataFrame              # risk_tier, skus, risk_tier_order
    stockout_by_store: pd.DataFrame  # store_name, stockout_rate
    sales_rows: int
    inventory_rows: int
    risk_rows: int


def build_overview_context(ds: Dataset) -> OverviewContext:
    dd = ds["dim_date"]
    store = ds["dim_store"][["store_key", "store_name", "store_format"]]
    product = ds["dim_product"][["product_key", "product_name", "category_name"]]
    date_cols = dd[["date_key", "full_date", "year_month_label"]]

    sales = (
        ds["fact_sales"][["transaction_id", "date_key", "store_key", "product_key", "quantity",
                          "net_sales", "gross_profit"]]
        .merge(date_cols, on="date_key", how="left", validate="many_to_one")
        .merge(store, on="store_key", how="left", validate="many_to_one")
        .merge(product, on="product_key", how="left", validate="many_to_one")
    )
    inventory = (
        ds["fact_inventory_daily"][["store_key", "product_key", "date_key", "closing_stock"]]
        .merge(date_cols[["date_key", "full_date"]], on="date_key", how="left", validate="many_to_one")
        .merge(store, on="store_key", how="left", validate="many_to_one")
        .merge(product[["product_key", "category_name"]], on="product_key", how="left",
               validate="many_to_one")
    )
    risk = (
        ds["v_inventory_risk"][["store_key", "product_key", "risk_tier", "stock_value_at_current_cost",
                                "snapshot_date"]]
        .merge(store[["store_key", "store_name", "store_format"]], on="store_key", how="left",
               validate="many_to_one")
        .merge(product[["product_key", "category_name"]], on="product_key", how="left",
               validate="many_to_one")
    )
    lo, hi = dd["full_date"].min().date(), dd["full_date"].max().date()
    snapshot = risk["snapshot_date"].max().date() if len(risk) else None
    return OverviewContext(
        sales=sales, inventory=inventory, risk=risk, date_min=lo, date_max=hi,
        snapshot_date=snapshot,
        stores=tuple(sorted(ds["dim_store"]["store_name"].unique())),
        categories=tuple(sorted(ds["dim_product"]["category_name"].unique())),
        store_formats=tuple(sorted(ds["dim_store"]["store_format"].unique())),
        # Measure 'Data Coverage': ALL(dim_date), so it never reacts to a filter.
        coverage_text=f"Synthetic data loaded: {lo:%Y-%m-%d} to {hi:%Y-%m-%d}",
    )


@lru_cache(maxsize=4)
def _context_for(resolved_dir: str) -> OverviewContext:
    return build_overview_context(load_dataset(Path(resolved_dir)))


def load_overview_context(data_dir: Path | str | None = None) -> OverviewContext:
    """Cached per data folder (the dataset itself is cached by load_dataset)."""
    from .data_loader import _resolve_dir
    return _context_for(str(_resolve_dir(data_dir)))


# --------------------------------------------------------------------------- filtering


def _dimension_mask(df: pd.DataFrame, f: OverviewFilters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if f.stores:
        mask &= df["store_name"].isin(f.stores)
    if f.store_formats:
        mask &= df["store_format"].isin(f.store_formats)
    if f.categories:
        mask &= df["category_name"].isin(f.categories)
    return mask


def _date_mask(df: pd.DataFrame, f: OverviewFilters) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if f.date_start is not None:
        mask &= df["full_date"] >= pd.Timestamp(f.date_start)
    if f.date_end is not None:
        mask &= df["full_date"] <= pd.Timestamp(f.date_end)
    return mask


def filter_sales(ctx: OverviewContext, f: OverviewFilters) -> pd.DataFrame:
    return ctx.sales[_dimension_mask(ctx.sales, f) & _date_mask(ctx.sales, f)]


def filter_inventory(ctx: OverviewContext, f: OverviewFilters) -> pd.DataFrame:
    return ctx.inventory[_dimension_mask(ctx.inventory, f) & _date_mask(ctx.inventory, f)]


def filter_risk(ctx: OverviewContext, f: OverviewFilters) -> pd.DataFrame:
    """The risk view is a latest-day snapshot: the date filter does not apply."""
    return ctx.risk[_dimension_mask(ctx.risk, f)]


# --------------------------------------------------------------------------- result


def compute_overview(ctx: OverviewContext, f: OverviewFilters) -> OverviewResult:
    sales = filter_sales(ctx, f)
    inventory = filter_inventory(ctx, f)
    risk = filter_risk(ctx, f)
    kpis = OverviewKpis(
        net_sales=sales_m.net_sales(sales),
        gross_profit=sales_m.gross_profit(sales),
        gross_margin=sales_m.gross_margin(sales),
        units_sold=sales_m.units_sold(sales),
        average_basket=sales_m.average_basket_value(sales),
        stockout_rate=inv.stockout_rate(inventory),
        stockout_now=inv.tier_count(risk, "Stockout"),
        critical=inv.tier_count(risk, "Critical"),
        low=inv.tier_count(risk, "Low"),
        stock_value=inv.stock_value_at_cost(risk),
    )
    return OverviewResult(
        kpis=kpis,
        monthly=sales_m.monthly_net_sales(sales),
        by_store=sales_m.net_sales_by_store(sales),
        top_products=sales_m.top_products_by_net_sales(sales, 10),
        tiers=inv.risk_tier_counts(risk),
        stockout_by_store=inv.stockout_rate_by_store(inventory),
        sales_rows=len(sales),
        inventory_rows=len(inventory),
        risk_rows=len(risk),
    )
