"""Inventory measures used by the Executive Overview, ported from _Measures.tmdl.

    SKU Days          = COUNTROWS ( fact_inventory_daily )
    Stockout SKU Days = CALCULATE ( COUNTROWS ( fact_inventory_daily ), fact_inventory_daily[closing_stock] = 0 )
    Stockout Rate     = DIVIDE ( COALESCE ( [Stockout SKU Days], 0 ), [SKU Days] )
    Stockout SKUs Now / Critical SKUs / Low Cover SKUs
                      = CALCULATE ( COUNTROWS ( v_inventory_risk ), v_inventory_risk[risk_tier] = "<tier>" )
    Stock Value At Cost = SUM ( v_inventory_risk[stock_value_at_current_cost] )

Snapshot rule: the risk measures read v_inventory_risk, a single latest-day snapshot with no
date relationship, so the date filter never reaches them. Stock levels are never summed
across days.

Intentional correction V2 (documented): the Power BI measure ``Tracked SKUs`` removed the filter
on risk_tier and therefore drew the same total on every tier bar. ``risk_tier_counts`` counts
the SKUs actually in each tier.
"""

from __future__ import annotations

import pandas as pd

from .measures_sales import divide

# Order and names from v_inventory_risk (risk_tier_order 1..6).
TIERS: tuple[str, ...] = ("Stockout", "Critical", "Low", "Healthy", "Overstock", "No recent sales")


def sku_days(inventory: pd.DataFrame) -> int:
    return int(len(inventory))


def stockout_sku_days(inventory: pd.DataFrame) -> int:
    return int((inventory["closing_stock"] == 0).sum())


def stockout_rate(inventory: pd.DataFrame):
    return divide(stockout_sku_days(inventory), sku_days(inventory))


def stockout_rate_by_store(inventory: pd.DataFrame) -> pd.DataFrame:
    """Stockout Rate by store, descending (ties broken by store name)."""
    g = inventory.assign(_so=(inventory["closing_stock"] == 0)).groupby("store_name")["_so"]
    out = g.mean().rename("stockout_rate").reset_index()
    return out.sort_values(["stockout_rate", "store_name"], ascending=[False, True]).reset_index(drop=True)


def tier_count(risk: pd.DataFrame, tier: str) -> int:
    return int((risk["risk_tier"] == tier).sum())


def risk_tier_counts(risk: pd.DataFrame) -> pd.DataFrame:
    """SKUs per risk tier (V2). All six tiers are returned, in tier order, zeros included."""
    counts = risk["risk_tier"].value_counts()
    return pd.DataFrame({
        "risk_tier": list(TIERS),
        "skus": [int(counts.get(t, 0)) for t in TIERS],
        "risk_tier_order": range(1, len(TIERS) + 1),
    })


def tracked_skus(risk: pd.DataFrame) -> int:
    return int(len(risk))


def stock_value_at_cost(risk: pd.DataFrame):
    return None if len(risk) == 0 else float(risk["stock_value_at_current_cost"].sum())
