"""Inventory measures of the Inventory Intelligence page, ported from _Measures.tmdl.

Reused unchanged from measures_inventory: SKU Days, Stockout SKU Days, Stockout Rate,
Stockout Rate by store, tier counts, tracked SKUs. New here:

    Availability Rate   = DIVIDE ( [SKU Days] - COALESCE ( [Stockout SKU Days], 0 ), [SKU Days] )
    Closing Stock Units = CALCULATE ( SUM ( fact_inventory_daily[closing_stock] ), LASTDATE ( dim_date[full_date] ) )
    Avg Daily Units 14D = units sold / distinct dates over DATESINPERIOD ( dim_date[full_date], MAX ( dim_date[full_date] ), -14, DAY )
    Days of Cover       = DIVIDE ( [Closing Stock Units], [Avg Daily Units 14D] )
    At-Risk SKU Share   = (Stockout + Critical + Low SKUs) / tracked SKUs        (tier filter ignored: V6)

Snapshot rules: closing stock is read on ONE date (the last date of the selection) and is never
summed across dates; the risk view has no date relationship. None stands for DAX BLANK().
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .measures_inventory import sku_days, stockout_sku_days, tier_count, tracked_skus
from .measures_sales import divide

AT_RISK_TIERS = ("Stockout", "Critical", "Low")


def availability_rate(inventory: pd.DataFrame):
    n = sku_days(inventory)
    return divide(n - stockout_sku_days(inventory), n)


def closing_stock_units(inventory: pd.DataFrame, last_date: date | None):
    """Closing stock on ``last_date`` only (LASTDATE); BLANK when no row exists that day."""
    if last_date is None:
        return None
    day = inventory[inventory["full_date"] == pd.Timestamp(last_date)]
    return None if len(day) == 0 else int(day["closing_stock"].sum())


def avg_daily_units_14d(inventory: pd.DataFrame, last_date: date | None):
    """Units sold over the 14 calendar days ending on ``last_date`` / distinct dates with rows.

    The window is not limited by the start of the date range (DATESINPERIOD replaces the filter).
    """
    if last_date is None:
        return None
    start = pd.Timestamp(last_date - timedelta(days=13))
    window = inventory[(inventory["full_date"] >= start) & (inventory["full_date"] <= pd.Timestamp(last_date))]
    if len(window) == 0:
        return None
    return divide(int(window["units_sold"].sum()), int(window["date_key"].nunique()))


def days_of_cover(closing, avg_daily):
    return divide(closing, avg_daily)


def at_risk_sku_share(risk: pd.DataFrame):
    return divide(sum(tier_count(risk, t) for t in AT_RISK_TIERS) if len(risk) else None, tracked_skus(risk) or None)


def median_sku_days_of_cover(risk: pd.DataFrame):
    """Median of the SKU-level days of cover (Stockout SKUs count as 0, BLANK covers are skipped)."""
    values = risk["days_of_cover"].dropna()
    return None if len(values) == 0 else float(values.median())


def stockout_rate_by_day(inventory: pd.DataFrame) -> pd.DataFrame:
    """Daily stockout rate = SKU-days with zero closing stock / SKU-days, per date (ascending)."""
    g = inventory.assign(_so=(inventory["closing_stock"] == 0)).groupby("full_date")["_so"].mean()
    return g.rename("stockout_rate").reset_index().sort_values("full_date").reset_index(drop=True)


def stockout_rate_matrix(inventory: pd.DataFrame) -> pd.DataFrame:
    """Category x store stockout rate; a cell is NaN (blank) when the pair has no SKU-days."""
    flagged = inventory.assign(_so=(inventory["closing_stock"] == 0))
    return flagged.pivot_table(index="category_name", columns="store_name", values="_so", aggfunc="mean",
                               observed=True).sort_index().sort_index(axis=1)
