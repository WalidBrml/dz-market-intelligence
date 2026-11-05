"""Sales measures used by the Executive Overview, ported from _Measures.tmdl.

Every function takes an already-filtered frame (the equivalent of a DAX filter context) and
returns a plain value; ``None`` stands for DAX BLANK(). Definitions (verbatim from the model):

    Net Sales            = SUM ( fact_sales[net_sales] )
    Gross Profit         = SUM ( fact_sales[gross_profit] )
    Units Sold           = SUM ( fact_sales[quantity] )
    Transactions         = DISTINCTCOUNT ( fact_sales[transaction_id] )
    Gross Margin %       = DIVIDE ( [Gross Profit], [Net Sales] )
    Average Basket Value = DIVIDE ( [Net Sales], [Transactions] )
    Top 10 Product Net Sales =
        VAR _top = TOPN ( 10, ALLSELECTED ( dim_product[product_name] ), [Net Sales], DESC )
        RETURN CALCULATE ( [Net Sales], KEEPFILTERS ( _top ) )
"""

from __future__ import annotations

import pandas as pd


def divide(numerator, denominator):
    """DAX DIVIDE: BLANK when the denominator is 0 or BLANK, or the numerator is BLANK."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _sum(df: pd.DataFrame, column: str):
    """SUM over a column: BLANK when there are no rows in context."""
    return None if len(df) == 0 else df[column].sum()


def net_sales(sales: pd.DataFrame):
    value = _sum(sales, "net_sales")
    return None if value is None else float(value)


def gross_profit(sales: pd.DataFrame):
    value = _sum(sales, "gross_profit")
    return None if value is None else float(value)


def units_sold(sales: pd.DataFrame):
    value = _sum(sales, "quantity")
    return None if value is None else int(value)


def transactions(sales: pd.DataFrame) -> int:
    return int(sales["transaction_id"].nunique())


def gross_margin(sales: pd.DataFrame):
    return divide(gross_profit(sales), net_sales(sales))


def average_basket_value(sales: pd.DataFrame):
    return divide(net_sales(sales), transactions(sales))


def monthly_net_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Net Sales by dim_date[year_month_label], ascending. Months with no sales are absent."""
    out = sales.groupby("year_month_label", as_index=False)["net_sales"].sum()
    return out.sort_values("year_month_label").reset_index(drop=True)


def net_sales_by_store(sales: pd.DataFrame) -> pd.DataFrame:
    """Net Sales by store, descending (ties broken by store name)."""
    out = sales.groupby("store_name", as_index=False)["net_sales"].sum()
    return out.sort_values(["net_sales", "store_name"], ascending=[False, True]).reset_index(drop=True)


def top_products_by_net_sales(sales: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N products by Net Sales within the current selection.

    TOPN keeps every row tied with the N-th value, so more than N rows can come back on a
    tie; products without sales (BLANK) are never shown. Descending, ties broken by name.
    """
    out = sales.groupby("product_name", as_index=False)["net_sales"].sum()
    out = out.sort_values(["net_sales", "product_name"], ascending=[False, True]).reset_index(drop=True)
    if len(out) > n:
        out = out[out["net_sales"] >= out["net_sales"].iloc[n - 1]].reset_index(drop=True)
    return out
