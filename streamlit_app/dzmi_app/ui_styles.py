"""Display formats of the sortable grids on the inventory page (pandas Styler; values stay numeric)."""

from __future__ import annotations

import pandas as pd

RISK_HEADERS = {
    "risk_tier": "Risk tier", "store_code": "Store", "sku": "SKU", "product_name": "Product",
    "closing_stock": "Closing stock", "avg_daily_units_14d": "Avg daily units (14d)",
    "days_of_cover": "Days of cover", "stock_value_at_current_cost": "Stock value at cost",
}


def risk_table_styler(table: pd.DataFrame):
    """Formats of v_inventory_risk in the model: 0, #,0.00, #,0.0, #,0 (blank = empty cell)."""
    shown = table.rename(columns=RISK_HEADERS)
    return shown.style.format({
        "Closing stock": "{:,.0f}", "Avg daily units (14d)": "{:,.2f}", "Days of cover": "{:,.1f}",
        "Stock value at cost": "{:,.0f}"}, na_rep="")
