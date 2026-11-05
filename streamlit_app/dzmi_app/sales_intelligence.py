"""Sales Intelligence (Page 2): context building and result computation. No Streamlit here.

Filter propagation follows relationships.tmdl:

* Date range  -> dim_date           -> fact_sales
* Wilaya      -> dim_store.wilaya   -> fact_sales   (the STORE's wilaya, not the customer's)
* Category / Brand -> dim_product   -> fact_sales
* dim_customer is only used as an axis (household type); lines without a customer form the
  blank group, exactly as Power BI shows them.

fact_sales is never joined to fact_inventory_daily here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import measures_sales as base
from . import measures_sales_intelligence as si
from .data_loader import Dataset, _resolve_dir, load_dataset

BLANK_HOUSEHOLD = "(Blank)"        # how Power BI labels lines without a customer
MONTH_COLUMNS = ["year_month_label", "net_sales", "mom", "gross_margin", "avg_daily_net_sales",
                 "units_sold"]
PRODUCT_COLUMNS = ["product_name", "category_name", "net_sales", "gross_profit", "gross_margin",
                   "avg_selling_price", "units_sold", "promo_share"]


@dataclass(frozen=True)
class SalesFilters:
    """Sales Intelligence slicers: Date range, Wilaya, Category, Brand (empty tuple = All)."""

    date_start: date | None = None
    date_end: date | None = None
    wilayas: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()


@dataclass(frozen=True)
class SalesContext:
    sales: pd.DataFrame          # fact_sales lines with every attribute the page needs
    dates: pd.DataFrame          # dim_date: full_date, year_month, year_month_label, is_complete_month
    date_min: date
    date_max: date
    wilayas: tuple[str, ...]     # slicer options, in the order of the Power BI slicer (descending)
    categories: tuple[str, ...]
    brands: tuple[str, ...]
    coverage_text: str


@dataclass(frozen=True)
class SalesResult:
    daily: pd.DataFrame           # full_date, net_sales, gross_profit
    monthly: pd.DataFrame         # MONTH_COLUMNS, sorted by MoM % descending
    monthly_total: dict           # the matrix grand-total row
    by_category: pd.DataFrame     # category_name, net_sales
    margin_by_category: pd.DataFrame   # category_name, gross_margin
    per_sqm_by_store: pd.DataFrame     # store_name, net_sales_per_sqm
    by_household: pd.DataFrame    # household_type, net_sales
    products: pd.DataFrame        # PRODUCT_COLUMNS
    products_total: dict
    by_weekday: pd.DataFrame      # day_name, net_sales (Monday..Sunday)
    sales_rows: int


def build_sales_context(ds: Dataset) -> SalesContext:
    dd = ds["dim_date"]
    dates = dd[["full_date", "year_month", "year_month_label", "is_complete_month"]]
    sales = (
        ds["fact_sales"][["transaction_id", "date_key", "store_key", "product_key", "customer_key",
                          "promotion_key", "quantity", "net_sales", "gross_profit"]]
        .merge(dd[["date_key", "full_date", "year_month_label", "day_name", "day_of_week"]],
               on="date_key", how="left", validate="many_to_one")
        .merge(ds["dim_store"][["store_key", "store_name", "wilaya", "floor_area_sqm"]],
               on="store_key", how="left", validate="many_to_one")
        .merge(ds["dim_product"][["product_key", "product_name", "category_name", "brand"]],
               on="product_key", how="left", validate="many_to_one")
        .merge(ds["dim_customer"][["customer_key", "household_type"]],
               on="customer_key", how="left", validate="many_to_one")
    )
    lo, hi = dd["full_date"].min().date(), dd["full_date"].max().date()
    return SalesContext(
        sales=sales, dates=dates, date_min=lo, date_max=hi,
        wilayas=tuple(sorted(ds["dim_store"]["wilaya"].unique(), reverse=True)),
        categories=tuple(sorted(ds["dim_product"]["category_name"].unique())),
        brands=tuple(sorted(ds["dim_product"]["brand"].unique())),
        coverage_text=f"Synthetic data loaded: {lo:%Y-%m-%d} to {hi:%Y-%m-%d}",
    )


@lru_cache(maxsize=4)
def _context_for(resolved_dir: str) -> SalesContext:
    return build_sales_context(load_dataset(Path(resolved_dir)))


def load_sales_context(data_dir: Path | str | None = None) -> SalesContext:
    return _context_for(str(_resolve_dir(data_dir)))


# --------------------------------------------------------------------------- filtering


def _non_date(df: pd.DataFrame, f: SalesFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if f.wilayas:
        mask &= df["wilaya"].isin(f.wilayas)
    if f.categories:
        mask &= df["category_name"].isin(f.categories)
    if f.brands:
        mask &= df["brand"].isin(f.brands)
    return df[mask]


def filter_sales_without_dates(ctx: SalesContext, f: SalesFilters) -> pd.DataFrame:
    """After Wilaya / Category / Brand only. Used for the prior-month lookup (DATEADD replaces the
    date filter), so it must not be limited by the selected date range."""
    return _non_date(ctx.sales, f)


def filter_sales(ctx: SalesContext, f: SalesFilters) -> pd.DataFrame:
    out = filter_sales_without_dates(ctx, f)
    if f.date_start is not None:
        out = out[out["full_date"] >= pd.Timestamp(f.date_start)]
    if f.date_end is not None:
        out = out[out["full_date"] <= pd.Timestamp(f.date_end)]
    return out


def _dates_in_range(ctx: SalesContext, f: SalesFilters) -> pd.DataFrame:
    """dim_date rows inside the date slicer (dim_date is never filtered by the other slicers)."""
    d = ctx.dates
    if f.date_start is not None:
        d = d[d["full_date"] >= pd.Timestamp(f.date_start)]
    if f.date_end is not None:
        d = d[d["full_date"] <= pd.Timestamp(f.date_end)]
    return d


# --------------------------------------------------------------------------- pieces


def _value_or_none(x):
    return None if pd.isna(x) else x


def monthly_performance(ctx: SalesContext, f: SalesFilters) -> tuple[pd.DataFrame, dict]:
    """The 'Monthly performance' matrix: one row per year_month_label plus the grand total.

    A month row is shown when at least one of its five measures is not BLANK (matrix behaviour).
    Sorted by Net Sales MoM % descending, BLANK last.
    """
    sales = filter_sales(ctx, f)
    no_date = filter_sales_without_dates(ctx, f)
    dates = _dates_in_range(ctx, f)
    grouped = si.measures_by(sales, ["year_month_label"]).set_index("year_month_label")

    rows = []
    for label, month_dates in dates.groupby("year_month_label", sort=False):
        start, end = month_dates["full_date"].min().date(), month_dates["full_date"].max().date()
        if label in grouped.index:
            g = grouped.loc[label]
            net, gm = _value_or_none(g["net_sales"]), _value_or_none(g["gross_margin"])
            avg, units = _value_or_none(g["avg_daily_net_sales"]), _value_or_none(g["units_sold"])
        else:
            net = gm = avg = units = None
        mom = si.net_sales_mom(ctx.dates, no_date, net, start, end)
        if any(v is not None for v in (net, mom, gm, avg, units)):
            rows.append({"year_month_label": label, "net_sales": net, "mom": mom, "gross_margin": gm,
                         "avg_daily_net_sales": avg, "units_sold": None if units is None else int(units)})
    frame = pd.DataFrame(rows, columns=MONTH_COLUMNS)
    frame = frame.astype({"net_sales": "float64", "mom": "float64", "gross_margin": "float64",
                          "avg_daily_net_sales": "float64", "units_sold": "float64"})
    frame = si.sort_by_mom_desc(frame)

    total = {"net_sales": base.net_sales(sales), "mom": None, "gross_margin": base.gross_margin(sales),
             "avg_daily_net_sales": si.average_daily_net_sales(sales), "units_sold": base.units_sold(sales)}
    if len(dates):
        total["mom"] = si.net_sales_mom(ctx.dates, no_date, total["net_sales"],
                                        dates["full_date"].min().date(), dates["full_date"].max().date())
    return frame, total


def product_table(sales: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """'Products ranked by net sales' with the sort saved in the report: category A-Z.

    Within a category the report defines no order; net sales descending is used (assumption).
    """
    g = si.measures_by(sales, ["product_name", "category_name"])
    g = g.sort_values(["category_name", "net_sales", "product_name"],
                      ascending=[True, False, True], kind="stable").reset_index(drop=True)
    total = {"net_sales": base.net_sales(sales), "gross_profit": base.gross_profit(sales),
             "gross_margin": base.gross_margin(sales), "avg_selling_price": si.average_selling_price(sales),
             "units_sold": base.units_sold(sales), "promo_share": si.promo_sales_share(sales)}
    return g[PRODUCT_COLUMNS], total


def _sorted_desc(frame: pd.DataFrame, value: str, label: str) -> pd.DataFrame:
    return frame.sort_values([value, label], ascending=[False, True], kind="stable").reset_index(drop=True)


def compute_sales_intelligence(ctx: SalesContext, f: SalesFilters) -> SalesResult:
    sales = filter_sales(ctx, f)
    monthly, monthly_total = monthly_performance(ctx, f)

    daily = (sales.groupby("full_date", as_index=False)[["net_sales", "gross_profit"]].sum()
             .sort_values("full_date").reset_index(drop=True))

    by_cat = si.measures_by(sales, ["category_name"])
    by_store = sales.groupby(["store_name", "floor_area_sqm"], as_index=False)["net_sales"].sum()
    by_store["net_sales_per_sqm"] = by_store["net_sales"] / by_store["floor_area_sqm"].where(
        by_store["floor_area_sqm"] != 0)
    household = sales.assign(household_type=sales["household_type"].fillna(BLANK_HOUSEHOLD))
    by_household = household.groupby("household_type", as_index=False)["net_sales"].sum()
    weekday = (sales.groupby(["day_of_week", "day_name"], as_index=False)["net_sales"].sum()
               .sort_values("day_of_week").reset_index(drop=True))
    products, products_total = product_table(sales)

    return SalesResult(
        daily=daily, monthly=monthly, monthly_total=monthly_total,
        by_category=_sorted_desc(by_cat[["category_name", "net_sales"]], "net_sales", "category_name"),
        margin_by_category=_sorted_desc(by_cat[["category_name", "gross_margin"]].dropna(),
                                        "gross_margin", "category_name"),
        per_sqm_by_store=_sorted_desc(by_store[["store_name", "net_sales_per_sqm"]].dropna(),
                                      "net_sales_per_sqm", "store_name"),
        by_household=_sorted_desc(by_household, "net_sales", "household_type"),
        products=products, products_total=products_total,
        by_weekday=weekday[["day_name", "net_sales"]],
        sales_rows=len(sales),
    )
