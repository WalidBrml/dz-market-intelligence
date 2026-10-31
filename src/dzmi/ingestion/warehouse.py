"""Transform operational-style (ops) tables into the star-schema (dwh) tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_generation.calendar_features import build_calendar

DWH_TABLES = ["dim_date", "dim_store", "dim_product", "dim_customer", "dim_promotion",
              "fact_sales", "fact_inventory_daily", "fact_promotions"]


def _date_key(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype("int64")


def build_warehouse(raw: dict[str, pd.DataFrame], cfg) -> dict[str, pd.DataFrame]:
    inv = raw["inventory_daily"]
    start, end = inv["inventory_date"].min(), inv["inventory_date"].max()
    dim_date = build_calendar(start.date(), end.date(), cfg.calendar)

    st = raw["store"]
    dim_store = st.assign(store_key=st["store_id"])[
        ["store_key", "store_id", "store_code", "store_name", "wilaya", "city", "store_format",
         "floor_area_sqm", "opened_date"]]

    ph = raw["product_price_history"].sort_values(["product_id", "valid_from"])
    latest = ph.groupby("product_id", as_index=False).last()[["product_id", "list_price", "unit_cost"]]
    pr = raw["product"].merge(raw["category"], on="category_id", how="left").merge(latest, on="product_id")
    dim_product = pr.assign(product_key=pr["product_id"]).rename(
        columns={"list_price": "current_list_price", "unit_cost": "current_unit_cost"})[
        ["product_key", "product_id", "sku", "product_name", "category_name", "is_perishable", "brand",
         "pack_size", "current_list_price", "current_unit_cost"]]

    cu = raw["customer"]
    dim_customer = cu.assign(customer_key=cu["customer_id"])[
        ["customer_key", "customer_id", "wilaya", "household_type", "signup_date"]]

    pm = raw["promotion"]
    dim_promotion = pm.assign(promotion_key=pm["promotion_id"])[
        ["promotion_key", "promotion_name", "promotion_type", "discount_pct", "start_date", "end_date"]]

    # fact_sales: line grain; cost = standard cost valid in the sale's calendar year
    tx, ln = raw["sales_transaction"], raw["sales_transaction_line"]
    f = ln.merge(tx[["transaction_id", "store_id", "customer_id", "transaction_date"]], on="transaction_id", how="left")
    f["year"] = f["transaction_date"].dt.year
    cost = ph.assign(year=ph["valid_from"].dt.year)[["product_id", "year", "unit_cost"]]
    f = f.merge(cost, on=["product_id", "year"], how="left")
    f = f.sort_values(["transaction_date", "transaction_id", "line_no"], kind="mergesort").reset_index(drop=True)
    gross = np.round(f["quantity"] * f["unit_price"], 2)
    net = np.round(gross - f["discount_amount"], 2)
    cogs = np.round(f["quantity"] * f["unit_cost"], 2)
    fact_sales = pd.DataFrame({
        "sales_line_id": np.arange(1, len(f) + 1, dtype=np.int64),
        "transaction_id": f["transaction_id"], "line_no": f["line_no"],
        "date_key": _date_key(f["transaction_date"]), "store_key": f["store_id"],
        "product_key": f["product_id"], "customer_key": f["customer_id"], "promotion_key": f["promotion_id"],
        "quantity": f["quantity"], "unit_price": f["unit_price"], "unit_cost": f["unit_cost"],
        "gross_sales": gross, "discount_amount": f["discount_amount"], "net_sales": net,
        "cogs": cogs, "gross_profit": np.round(net - cogs, 2),
    })

    fact_inv = pd.DataFrame({
        "store_key": inv["store_id"], "product_key": inv["product_id"], "date_key": _date_key(inv["inventory_date"]),
        "opening_stock": inv["opening_stock"], "received_qty": inv["received_qty"], "units_sold": inv["units_sold"],
        "closing_stock": inv["closing_stock"], "on_order_qty": inv["on_order_qty"], "stockout_flag": inv["stockout_flag"],
    })

    # fact_promotions: expand each promotion scope over the promotion's days inside the data range
    sc = raw["promotion_scope"].merge(pm[["promotion_id", "start_date", "end_date", "discount_pct"]], on="promotion_id")
    sc["start_date"] = sc["start_date"].clip(lower=start)
    sc["end_date"] = sc["end_date"].clip(upper=end)
    ndays = ((sc["end_date"] - sc["start_date"]).dt.days + 1).clip(lower=0).to_numpy()
    rep = np.repeat(np.arange(len(sc)), ndays)
    offs = np.arange(rep.size) - np.repeat(np.cumsum(ndays) - ndays, ndays)
    days = sc["start_date"].to_numpy()[rep] + offs.astype("timedelta64[D]")
    fact_promo = pd.DataFrame({
        "store_key": sc["store_id"].to_numpy()[rep], "product_key": sc["product_id"].to_numpy()[rep],
        "date_key": _date_key(pd.Series(days)), "promotion_key": sc["promotion_id"].to_numpy()[rep],
        "discount_pct": sc["discount_pct"].to_numpy()[rep],
    })
    return {"dim_date": dim_date[[
        "date_key", "full_date", "year", "quarter", "month", "month_name", "day_of_month", "day_of_week",
        "day_name", "iso_week", "is_weekend", "is_ramadan", "ramadan_day", "is_eid_al_fitr", "is_eid_al_adha",
        "is_pre_eid_al_fitr", "is_pre_eid_al_adha"]],
        "dim_store": dim_store, "dim_product": dim_product, "dim_customer": dim_customer,
        "dim_promotion": dim_promotion, "fact_sales": fact_sales, "fact_inventory_daily": fact_inv,
        "fact_promotions": fact_promo}
