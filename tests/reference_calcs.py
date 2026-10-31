"""Independent pandas calculations of the Milestone 2 metrics, straight from the SOURCE (ops) tables.

These functions deliberately do NOT read dwh.fact_sales, dwh.fact_inventory_daily or any SQL view, and they
find the unit cost with a different method (merge_asof on the price validity date) than the warehouse
transform (join on calendar year). If the SQL views and these numbers agree, both are very likely right.
Money is DZD. Ratios are fractions and NaN when the denominator is 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sales_lines(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per sales line with gross / net / cogs / profit and the descriptive attributes attached."""
    tx = raw["sales_transaction"][["transaction_id", "store_id", "transaction_date"]]
    ln = raw["sales_transaction_line"].merge(tx, on="transaction_id", validate="many_to_one")
    ph = raw["product_price_history"][["product_id", "valid_from", "unit_cost"]].sort_values("valid_from")
    ln = pd.merge_asof(ln.sort_values("transaction_date"), ph, left_on="transaction_date", right_on="valid_from",
                       by="product_id", direction="backward")
    assert ln["unit_cost"].notna().all(), "every sale must have a standard cost"
    ln["gross"] = ln["quantity"] * ln["unit_price"]
    ln["net"] = ln["gross"] - ln["discount_amount"]
    ln["cogs"] = ln["quantity"] * ln["unit_cost"]
    ln["profit"] = ln["net"] - ln["cogs"]
    prod = raw["product"].merge(raw["category"], on="category_id")[["product_id", "category_name"]]
    ln = ln.merge(prod, on="product_id", validate="many_to_one")
    ln = ln.merge(raw["store"][["store_id", "wilaya", "floor_area_sqm"]], on="store_id", validate="many_to_one")
    ln["has_promo"] = ln["promotion_id"].notna()
    ln["date"] = ln["transaction_date"].dt.normalize()
    ln["year_month"] = ln["date"].dt.year * 100 + ln["date"].dt.month
    return ln


def _agg(g) -> pd.DataFrame:
    return g.agg(units_sold=("quantity", "sum"), gross_sales=("gross", "sum"), discount_amount=("discount_amount", "sum"),
                 net_sales=("net", "sum"), cogs=("cogs", "sum"), gross_profit=("profit", "sum"),
                 transactions=("transaction_id", "nunique"))


def _ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.where(den != 0, num / np.where(den != 0, den, 1.0), np.nan)


def add_ratios(df: pd.DataFrame, with_basket: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["gross_margin"] = _ratio(df["gross_profit"], df["net_sales"])
    df["avg_selling_price"] = _ratio(df["net_sales"], df["units_sold"])
    df["discount_rate"] = _ratio(df["discount_amount"], df["gross_sales"])
    if with_basket:
        df["avg_basket_value"] = _ratio(df["net_sales"], df["transactions"])
    return df


def daily(raw, dwh_dates: pd.Series) -> pd.DataFrame:
    """Network sales per calendar date (zero rows for dates without sales). dwh_dates = the calendar to cover."""
    ln = sales_lines(raw)
    g = _agg(ln.groupby("date")).reindex(pd.DatetimeIndex(dwh_dates), fill_value=0)
    g.index.name = "date"
    return add_ratios(g.reset_index())


def monthly(raw) -> pd.DataFrame:
    return add_ratios(_agg(sales_lines(raw).groupby("year_month")).reset_index())


def by_category_month(raw) -> pd.DataFrame:
    return add_ratios(_agg(sales_lines(raw).groupby(["category_name", "year_month"])).drop(columns="transactions")
                      .reset_index(), with_basket=False)


def products(raw) -> pd.DataFrame:
    """One row per product (all products) with sales, promo split, availability, rank and ABC class."""
    ln = sales_lines(raw)
    base = raw["product"][["product_id"]].copy()
    s = _agg(ln.groupby("product_id")).drop(columns="transactions")
    promo = ln[ln["has_promo"]].groupby("product_id").agg(promo_units_sold=("quantity", "sum"), promo_net_sales=("net", "sum"))
    inv = raw["inventory_daily"]
    a = inv.groupby("product_id").agg(sku_days=("units_sold", "size"), stockout_sku_days=("stockout_flag", "sum"),
                                      stores_carrying=("store_id", "nunique"))
    df = base.merge(s, on="product_id", how="left").merge(promo, on="product_id", how="left").merge(a, on="product_id", how="left")
    df = df.fillna(0)
    df = add_ratios(df, with_basket=False)
    df["promo_unit_share"] = _ratio(df["promo_units_sold"], df["units_sold"])
    df["stockout_rate"] = _ratio(df["stockout_sku_days"], df["sku_days"])
    df = df.sort_values(["net_sales", "product_id"], ascending=[False, True]).reset_index(drop=True)
    df["net_sales_rank"] = np.arange(1, len(df) + 1)
    total = df["net_sales"].sum()
    cum = df["net_sales"].cumsum()
    df["net_sales_share"] = _ratio(df["net_sales"], total)
    df["cumulative_net_sales_share"] = _ratio(cum, total)
    before = _ratio(cum - df["net_sales"], total)
    df["abc_class"] = np.where(total == 0, "C", np.where(before < 0.80, "A", np.where(before < 0.95, "B", "C")))
    gp = df.sort_values(["gross_profit", "product_id"], ascending=[False, True])["product_id"].tolist()
    df["gross_profit_rank"] = df["product_id"].map({p: i + 1 for i, p in enumerate(gp)})
    return df


def stores(raw, wilaya_override: dict[int, str] | None = None) -> pd.DataFrame:
    ln = sales_lines(raw)
    st = raw["store"][["store_id", "wilaya", "floor_area_sqm"]].copy()
    if wilaya_override:   # used to test wilaya roll-ups with several stores in one wilaya
        st["wilaya"] = [wilaya_override.get(i, w) for i, w in zip(st["store_id"], st["wilaya"])]
    s = _agg(ln.groupby("store_id"))
    inv = raw["inventory_daily"].groupby("store_id").agg(sku_days=("units_sold", "size"),
                                                         stockout_sku_days=("stockout_flag", "sum"))
    df = st.merge(s, on="store_id", how="left").merge(inv, on="store_id", how="left").fillna(0)
    df = add_ratios(df)
    df["net_sales_per_sqm"] = _ratio(df["net_sales"], df["floor_area_sqm"])
    df["stockout_rate"] = _ratio(df["stockout_sku_days"], df["sku_days"])
    return df


def wilayas(raw, wilaya_override: dict[int, str] | None = None) -> pd.DataFrame:
    """Wilaya totals computed from store rows by SUMMING measures, then re-deriving the ratios (weighted)."""
    st = stores(raw, wilaya_override)
    g = st.groupby("wilaya").agg(store_count=("store_id", "size"), floor_area_sqm=("floor_area_sqm", "sum"),
                                 units_sold=("units_sold", "sum"), gross_sales=("gross_sales", "sum"),
                                 discount_amount=("discount_amount", "sum"), net_sales=("net_sales", "sum"),
                                 cogs=("cogs", "sum"), gross_profit=("gross_profit", "sum"),
                                 transactions=("transactions", "sum"), sku_days=("sku_days", "sum"),
                                 stockout_sku_days=("stockout_sku_days", "sum")).reset_index()
    g = add_ratios(g)
    g["net_sales_per_sqm"] = _ratio(g["net_sales"], g["floor_area_sqm"])
    g["stockout_rate"] = _ratio(g["stockout_sku_days"], g["sku_days"])
    return g


def availability_daily(raw) -> pd.DataFrame:
    inv = raw["inventory_daily"]
    g = inv.groupby(["store_id", "inventory_date"]).agg(sku_count=("units_sold", "size"),
                                                         stockout_sku_count=("stockout_flag", "sum"),
                                                         units_sold=("units_sold", "sum"),
                                                         closing_stock_units=("closing_stock", "sum")).reset_index()
    g["in_stock_sku_count"] = g["sku_count"] - g["stockout_sku_count"]
    g["availability_rate"] = _ratio(g["in_stock_sku_count"], g["sku_count"])
    g["stockout_rate"] = _ratio(g["stockout_sku_count"], g["sku_count"])
    return g


def inventory_risk(raw) -> pd.DataFrame:
    """Snapshot on the latest inventory date with 14-day demand, cover and risk tier (calendar-day window)."""
    inv = raw["inventory_daily"]
    last = inv["inventory_date"].max()
    recent = inv[inv["inventory_date"] >= last - pd.Timedelta(days=13)]
    r = recent.groupby(["store_id", "product_id"]).agg(units_sold_14d=("units_sold", "sum"),
                                                        days_observed_14d=("units_sold", "size"),
                                                        stockout_days_14d=("stockout_flag", "sum")).reset_index()
    snap = inv[inv["inventory_date"] == last][["store_id", "product_id", "closing_stock", "on_order_qty"]]
    df = snap.merge(r, on=["store_id", "product_id"])
    df["avg_daily_units_14d"] = _ratio(df["units_sold_14d"], df["days_observed_14d"])
    df["days_of_cover"] = _ratio(df["closing_stock"] * df["days_observed_14d"], df["units_sold_14d"])
    df["days_of_cover_incl_on_order"] = _ratio((df["closing_stock"] + df["on_order_qty"]) * df["days_observed_14d"],
                                               df["units_sold_14d"])
    cond = [df["closing_stock"] == 0, df["units_sold_14d"] == 0, df["days_of_cover"] < 3, df["days_of_cover"] < 7,
            df["days_of_cover"] > 60]
    df["risk_tier"] = np.select(cond, ["Stockout", "No recent sales", "Critical", "Low", "Overstock"], "Healthy")
    df["risk_tier_order"] = np.select(cond, [1, 6, 2, 3, 5], 4)
    cost = raw["product_price_history"].sort_values("valid_from").groupby("product_id")["unit_cost"].last()
    df["stock_value_at_current_cost"] = df["closing_stock"] * df["product_id"].map(cost)
    return df


def category_stockout_month(raw) -> pd.DataFrame:
    inv = raw["inventory_daily"].merge(raw["product"].merge(raw["category"], on="category_id")[["product_id", "category_name"]],
                                       on="product_id")
    inv["year_month"] = inv["inventory_date"].dt.year * 100 + inv["inventory_date"].dt.month
    return inv.groupby(["category_name", "year_month"]).agg(sku_days=("units_sold", "size"),
                                                           stockout_sku_days=("stockout_flag", "sum")).reset_index()
