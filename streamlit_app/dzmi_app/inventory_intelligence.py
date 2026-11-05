"""Inventory Intelligence (Page 3): context and result computation. No Streamlit here.

Filter propagation (relationships.tmdl):
* Date range  -> dim_date -> fact_inventory_daily only.
* Store / Category / Product -> dim_store / dim_product -> fact_inventory_daily AND v_inventory_risk.
* Risk tier   -> v_inventory_risk; applied to the SKU list only (approved deviation V6).
v_inventory_risk has no date relationship: it is the latest-day snapshot whatever the date range is.
fact_sales is not used on this page.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import measures_inventory as inv
from . import measures_inventory_intelligence as ii
from .data_loader import Dataset, _resolve_dir, load_dataset

TABLE_COLUMNS = ["risk_tier", "store_code", "sku", "product_name", "closing_stock", "avg_daily_units_14d",
                 "days_of_cover", "stock_value_at_current_cost"]


@dataclass(frozen=True)
class InventoryFilters:
    """Slicers: Date range, Store, Category, Product, Risk tier (empty tuple = All)."""

    date_start: date | None = None
    date_end: date | None = None
    stores: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    tiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventoryContext:
    inventory: pd.DataFrame     # fact_inventory_daily + date / store / product attributes
    risk: pd.DataFrame          # v_inventory_risk + category
    dates: pd.DataFrame         # dim_date.full_date
    date_min: date
    date_max: date
    snapshot_date: date | None
    stores: tuple[str, ...]
    categories: tuple[str, ...]
    products: tuple[str, ...]
    tiers: tuple[str, ...]      # tiers present in the snapshot, in risk_tier_order
    coverage_text: str


@dataclass(frozen=True)
class InventoryResult:
    stockout_rate: float | None
    availability_rate: float | None
    stockout_sku_days: int
    sku_days: int
    closing_stock: int | None
    last_date: date | None
    avg_daily_units_14d: float | None
    overall_days_of_cover: float | None
    median_days_of_cover: float | None
    at_risk_share: float | None
    daily: pd.DataFrame          # full_date, stockout_rate
    heatmap: pd.DataFrame        # category x store matrix
    tiers: pd.DataFrame          # risk_tier, skus, risk_tier_order
    risk_table: pd.DataFrame     # TABLE_COLUMNS
    tracked_skus: int


def build_inventory_context(ds: Dataset) -> InventoryContext:
    dd = ds["dim_date"]
    store = ds["dim_store"][["store_key", "store_name"]]
    product = ds["dim_product"][["product_key", "product_name", "category_name"]]
    inventory = (
        ds["fact_inventory_daily"][["store_key", "product_key", "date_key", "units_sold", "closing_stock"]]
        .merge(dd[["date_key", "full_date"]], on="date_key", how="left", validate="many_to_one")
        .merge(store, on="store_key", how="left", validate="many_to_one")
        .merge(product, on="product_key", how="left", validate="many_to_one")
    )
    risk = ds["v_inventory_risk"].drop(columns=["store_name", "product_name", "category_name"]).merge(
        store, on="store_key", how="left", validate="many_to_one").merge(
        product, on="product_key", how="left", validate="many_to_one")
    lo, hi = dd["full_date"].min().date(), dd["full_date"].max().date()
    tiers = (risk[["risk_tier", "risk_tier_order"]].drop_duplicates().sort_values("risk_tier_order")["risk_tier"])
    return InventoryContext(
        inventory=inventory, risk=risk, dates=dd[["full_date"]], date_min=lo, date_max=hi,
        snapshot_date=risk["snapshot_date"].max().date() if len(risk) else None,
        stores=tuple(sorted(ds["dim_store"]["store_name"].unique())),
        categories=tuple(sorted(ds["dim_product"]["category_name"].unique())),
        products=tuple(sorted(ds["dim_product"]["product_name"].unique())),
        tiers=tuple(tiers), coverage_text=f"Synthetic data loaded: {lo:%Y-%m-%d} to {hi:%Y-%m-%d}")


@lru_cache(maxsize=4)
def _context_for(resolved_dir: str) -> InventoryContext:
    return build_inventory_context(load_dataset(Path(resolved_dir)))


def load_inventory_context(data_dir: Path | str | None = None) -> InventoryContext:
    return _context_for(str(_resolve_dir(data_dir)))


# --------------------------------------------------------------------------- filtering


def _dimension_filter(df: pd.DataFrame, f: InventoryFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f.stores:
        mask &= df["store_name"].isin(f.stores)
    if f.categories:
        mask &= df["category_name"].isin(f.categories)
    if f.products:
        mask &= df["product_name"].isin(f.products)
    return df[mask]


def filter_inventory_without_dates(ctx: InventoryContext, f: InventoryFilters) -> pd.DataFrame:
    """Store / category / product filters only (used for the last-day and 14-day lookups)."""
    return _dimension_filter(ctx.inventory, f)


def filter_inventory(ctx: InventoryContext, f: InventoryFilters) -> pd.DataFrame:
    out = filter_inventory_without_dates(ctx, f)
    if f.date_start is not None:
        out = out[out["full_date"] >= pd.Timestamp(f.date_start)]
    if f.date_end is not None:
        out = out[out["full_date"] <= pd.Timestamp(f.date_end)]
    return out


def filter_risk(ctx: InventoryContext, f: InventoryFilters, with_tiers: bool = False) -> pd.DataFrame:
    """The snapshot after store / category / product filters. The date range never applies; the tier
    filter is applied only when ``with_tiers`` (the SKU list)."""
    out = _dimension_filter(ctx.risk, f)
    if with_tiers and f.tiers:
        out = out[out["risk_tier"].isin(f.tiers)]
    return out


def last_date_in_selection(ctx: InventoryContext, f: InventoryFilters) -> date | None:
    """LASTDATE ( dim_date[full_date] ) under the date slicer (dim_date is not filtered by the rest)."""
    d = ctx.dates["full_date"]
    if f.date_start is not None:
        d = d[d >= pd.Timestamp(f.date_start)]
    if f.date_end is not None:
        d = d[d <= pd.Timestamp(f.date_end)]
    return None if len(d) == 0 else d.max().date()


def risk_list(risk: pd.DataFrame) -> pd.DataFrame:
    """SKU risk list: most urgent tier first (the report sorts by risk_tier, whose sort column is
    risk_tier_order); within a tier by days of cover ascending (assumption - not defined in the
    report), then store code and SKU for a stable order."""
    t = risk.assign(_nan=risk["days_of_cover"].isna())
    t = t.sort_values(["risk_tier_order", "_nan", "days_of_cover", "store_code", "sku"], kind="stable")
    return t[TABLE_COLUMNS].reset_index(drop=True)


def compute_inventory_intelligence(ctx: InventoryContext, f: InventoryFilters) -> InventoryResult:
    inventory = filter_inventory(ctx, f)
    no_date = filter_inventory_without_dates(ctx, f)
    risk = filter_risk(ctx, f)
    last = last_date_in_selection(ctx, f)
    closing = ii.closing_stock_units(no_date, last)
    avg = ii.avg_daily_units_14d(no_date, last)
    return InventoryResult(
        stockout_rate=inv.stockout_rate(inventory), availability_rate=ii.availability_rate(inventory),
        stockout_sku_days=inv.stockout_sku_days(inventory), sku_days=inv.sku_days(inventory),
        closing_stock=closing, last_date=last, avg_daily_units_14d=avg,
        overall_days_of_cover=ii.days_of_cover(closing, avg),
        median_days_of_cover=ii.median_sku_days_of_cover(risk), at_risk_share=ii.at_risk_sku_share(risk),
        daily=ii.stockout_rate_by_day(inventory), heatmap=ii.stockout_rate_matrix(inventory),
        tiers=inv.risk_tier_counts(risk), risk_table=risk_list(filter_risk(ctx, f, with_tiers=True)),
        tracked_skus=inv.tracked_skus(risk))
