"""Milestone 2: SQL analytical views versus independent pandas calculations from the source (ops) tables.

Every test compares an aggregate read from a view with the same aggregate computed in reference_calcs.py,
which never touches the warehouse facts or the views. Plain functions: run with pytest or the fallback runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import reference_calcs as ref
from dzmi.ingestion.db import Database
from helpers import loaded_sqlite, shared_smoke_db, shared_two_year_db, smoke, two_year

MONEY = 0.01        # DZD tolerance for sums
RATIO = 1e-9        # tolerance for ratios


def view(db, name: str) -> pd.DataFrame:
    return db.query_df(f"SELECT * FROM dwh.{name}")


def assert_match(actual: pd.DataFrame, expected: pd.DataFrame, key, cols, atol=MONEY, what=""):
    """Same keys, and every compared column equal within tolerance (NaN equals NaN)."""
    key = [key] if isinstance(key, str) else list(key)
    a = actual.set_index(key).sort_index()
    e = expected.set_index(key).sort_index()
    assert len(a) == len(e) and a.index.equals(e.index), (
        f"{what}: row keys differ (view {len(a)} rows, reference {len(e)} rows); "
        f"only in view: {list(a.index.difference(e.index))[:3]}, only in reference: {list(e.index.difference(a.index))[:3]}")
    for c in cols:
        av = pd.to_numeric(a[c]).to_numpy(dtype=float)
        ev = pd.to_numeric(e[c]).to_numpy(dtype=float)
        ok = np.isclose(av, ev, rtol=1e-9, atol=atol, equal_nan=True)
        assert ok.all(), f"{what}: column {c} differs at {a.index[~ok][:3].tolist()}: view {av[~ok][:3]}, reference {ev[~ok][:3]}"


MEASURES = ["units_sold", "gross_sales", "discount_amount", "net_sales", "cogs", "gross_profit"]
RATIOS = ["gross_margin", "avg_selling_price", "discount_rate"]


# ------------------------------------------------------------------------------------------ daily / monthly
def test_daily_view_matches_independent_calculation():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_kpi_network_daily")
    v["date"] = pd.to_datetime(v["full_date"])
    exp = ref.daily(raw, dwh["dim_date"]["full_date"])
    assert len(v) == len(dwh["dim_date"]) == 90                       # one row per calendar date, zeros included
    assert_match(v, exp, "date", MEASURES + ["transactions"], MONEY, "daily measures")
    assert_match(v, exp, "date", RATIOS + ["avg_basket_value"], RATIO, "daily ratios")


def test_monthly_view_matches_independent_calculation_and_flags_partial_month():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_kpi_network_monthly")
    exp = ref.monthly(raw)
    assert_match(v, exp, "year_month", MEASURES + ["transactions"], MONEY, "monthly measures")
    assert_match(v, exp, "year_month", RATIOS + ["avg_basket_value"], RATIO, "monthly ratios")
    flags = dict(zip(v["year_month"], v["is_complete_month"]))
    assert flags == {202502: 1, 202503: 1, 202504: 1, 202505: 0}      # data ends on 1 May 2025
    assert v["yoy_net_sales_growth"].isna().all() and (v["has_prior_year_comparison"] == 0).all()   # no 2024 data


def test_category_monthly_view_matches_independent_calculation():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_kpi_category_monthly")
    exp = ref.by_category_month(raw)
    n_categories, n_months = raw["category"]["category_name"].nunique(), 4
    assert len(v) == n_categories * n_months                           # spine: every category x every month
    full = v[["category_name", "year_month"]].merge(exp, on=["category_name", "year_month"], how="left").fillna(0)
    full = ref.add_ratios(full, with_basket=False)
    assert_match(v, full, ["category_name", "year_month"], MEASURES, MONEY, "category measures")
    assert_match(v, full, ["category_name", "year_month"], RATIOS, RATIO, "category ratios")
    stock = ref.category_stockout_month(raw)
    chk = v.merge(stock, on=["category_name", "year_month"], how="left", suffixes=("", "_ref")).fillna(0)
    assert (chk["sku_days"] == chk["sku_days_ref"]).all() and (chk["stockout_sku_days"] == chk["stockout_sku_days_ref"]).all()


# ------------------------------------------------------------------------------------------ year over year
def test_yoy_growth_matches_independent_calculation_on_two_years():
    cfg, raw, dwh = two_year()
    db = shared_two_year_db()
    mv = view(db, "v_kpi_network_monthly")
    m = ref.monthly(raw).set_index("year_month")
    assert len(mv) == 24 and (mv["is_complete_month"] == 1).all()
    for _, r in mv.iterrows():
        ym = int(r["year_month"])
        if ym < 202401:
            assert pd.isna(r["yoy_net_sales_growth"]) and r["has_prior_year_comparison"] == 0
        else:
            py = m.loc[ym - 100]
            for col, src in (("yoy_net_sales_growth", "net_sales"), ("yoy_units_growth", "units_sold"),
                             ("yoy_gross_profit_growth", "gross_profit")):
                assert abs(r[col] - (m.loc[ym, src] / py[src] - 1)) < 1e-9, (ym, col)
            assert r["has_prior_year_comparison"] == 1
    # daily: the same date one year earlier (29 Feb 2024 has no 2023 counterpart)
    dv = view(db, "v_kpi_network_daily")
    dv["date"] = pd.to_datetime(dv["full_date"])
    d = ref.daily(raw, dwh["dim_date"]["full_date"]).set_index("date")
    checked = 0
    for _, r in dv.iterrows():
        dt = r["date"]
        if dt.year == 2023:
            assert pd.isna(r["yoy_net_sales_growth"]) and r["has_prior_year_row"] == 0
        elif (dt.month, dt.day) == (2, 29):
            assert r["has_prior_year_row"] == 0 and pd.isna(r["yoy_net_sales_growth"])
        else:
            prior = d.loc[dt - pd.DateOffset(years=1), "net_sales"]
            assert r["has_prior_year_row"] == 1
            if prior == 0:
                assert pd.isna(r["yoy_net_sales_growth"])                # zero base: growth undefined, never infinity
            else:
                assert abs(r["yoy_net_sales_growth"] - (d.loc[dt, "net_sales"] / prior - 1)) < 1e-9
                checked += 1
    assert checked > 300


def test_category_yoy_matches_independent_calculation():
    cfg, raw, dwh = two_year()
    v = view(shared_two_year_db(), "v_kpi_category_monthly")
    exp = ref.by_category_month(raw).set_index(["category_name", "year_month"])
    cur = v[v["year_month"] >= 202401]
    assert len(cur) > 0 and (cur["has_prior_year_comparison"] == 1).all()
    for _, r in cur.iterrows():
        key, prior = (r["category_name"], int(r["year_month"])), (r["category_name"], int(r["year_month"]) - 100)
        py = exp.loc[prior, "net_sales"] if prior in exp.index else 0
        now = exp.loc[key, "net_sales"] if key in exp.index else 0
        if py == 0:
            assert pd.isna(r["yoy_net_sales_growth"])
        else:
            assert abs(r["yoy_net_sales_growth"] - (now / py - 1)) < 1e-9, key
    assert v[v["year_month"] < 202401]["yoy_net_sales_growth"].isna().all()


def _build_known_answer_db() -> Database:
    """Hand-built data with answers computed by hand (no generator involved)."""
    from datetime import date, timedelta
    db = Database.sqlite()
    db.apply_schema()
    h = db._h
    h.execute("INSERT INTO dwh.dim_store VALUES (1,1,'ST01','S','Alger','Algiers','Supermarket',1000,'2020-01-01')")
    h.execute("INSERT INTO dwh.dim_product VALUES (1,1,'X-1','P','Cat',0,'Brand A',6,100,80)")
    days = []
    for y, m, last in ((2024, 2, 29), (2024, 3, 31), (2024, 4, 30), (2025, 3, 31), (2025, 4, 15)):   # Apr 2025 partial
        days += [date(y, m, d) for d in range(1, last + 1)]
    names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
    for dt in days:
        h.execute("INSERT INTO dwh.dim_date VALUES (?,?,?,?,?,?,?,?,?,?,?,0,NULL,0,0,0,0)",
                  (dt.year * 10000 + dt.month * 100 + dt.day, dt.isoformat(), dt.year, (dt.month - 1) // 3 + 1,
                   dt.month, names[dt.month - 1], dt.day, dt.isoweekday(), dt.strftime("%A"), dt.isocalendar()[1],
                   int(dt.isoweekday() in (5, 6))))
    # (date, quantity): unit price 100, unit cost 80 -> net = 100*qty, cogs = 80*qty, profit = 20*qty
    sales = [(date(2024, 2, 29), 1), (date(2024, 3, 10), 2), (date(2025, 3, 10), 3), (date(2025, 3, 11), 1),
             (date(2024, 4, 5), 4), (date(2025, 4, 5), 6)]
    for i, (dt, q) in enumerate(sales, start=1):
        h.execute("INSERT INTO dwh.fact_sales VALUES (?,?,1,?,1,1,NULL,NULL,?,100,80,?,0,?,?,?)",
                  (i, i, dt.year * 10000 + dt.month * 100 + dt.day, q, 100 * q, 100 * q, 80 * q, 20 * q))
    h.commit()
    return db


def test_yoy_known_answers_zero_base_leap_day_and_partial_month():
    db = _build_known_answer_db()
    d = view(db, "v_kpi_network_daily").set_index("date_key")
    assert abs(d.loc[20250310, "yoy_net_sales_growth"] - 0.5) < 1e-12            # 300 vs 200
    assert abs(d.loc[20250310, "yoy_units_growth"] - 0.5) < 1e-12                # 3 vs 2 units
    assert abs(d.loc[20250310, "yoy_gross_profit_growth"] - 0.5) < 1e-12         # 60 vs 40
    assert d.loc[20250311, "has_prior_year_row"] == 1 and d.loc[20250311, "net_sales_py"] == 0
    assert pd.isna(d.loc[20250311, "yoy_net_sales_growth"])                      # prior day exists but sold nothing
    assert d.loc[20240229, "has_prior_year_row"] == 0 and pd.isna(d.loc[20240229, "yoy_net_sales_growth"])
    assert abs(d.loc[20250405, "yoy_net_sales_growth"] - 0.5) < 1e-12            # 600 vs 400
    m = view(db, "v_kpi_network_monthly").set_index("year_month")
    assert m.loc[202402, "days_in_month"] == 29 and m.loc[202402, "is_complete_month"] == 1   # leap February
    assert abs(m.loc[202503, "yoy_net_sales_growth"] - 1.0) < 1e-12              # (300+100) vs 200
    assert m.loc[202504, "is_complete_month"] == 0 and m.loc[202504, "days_in_data"] == 15
    assert pd.isna(m.loc[202504, "yoy_net_sales_growth"]) and m.loc[202504, "has_prior_year_comparison"] == 0
    assert pd.isna(m.loc[202402, "yoy_net_sales_growth"])                        # no Feb 2023 in the data
    assert m.loc[202503, "net_sales"] == 400 and m.loc[202503, "units_sold"] == 4
    assert abs(m.loc[202503, "avg_selling_price"] - 100) < 1e-12 and abs(m.loc[202503, "gross_margin"] - 0.2) < 1e-12
    db.close()


# ------------------------------------------------------------------------------------------ entity views
def test_product_view_matches_independent_calculation_including_rank_abc_promo_and_stockouts():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_kpi_product")
    exp = ref.products(raw)
    assert len(v) == len(raw["product"]) == 100                        # every product, sold or not
    num = MEASURES + ["promo_units_sold", "promo_net_sales", "sku_days", "stockout_sku_days", "stores_carrying",
                      "net_sales_rank", "gross_profit_rank"]
    assert_match(v, exp.assign(product_key=exp["product_id"]), "product_key", num, MONEY, "product measures")
    ratios = RATIOS + ["promo_unit_share", "stockout_rate", "net_sales_share", "cumulative_net_sales_share"]
    assert_match(v, exp.assign(product_key=exp["product_id"]), "product_key", ratios, RATIO, "product ratios")
    got = dict(zip(v["product_key"], v["abc_class"]))
    want = dict(zip(exp["product_id"], exp["abc_class"]))
    assert got == want and set(got.values()) == {"A", "B", "C"}
    assert sorted(v["net_sales_rank"]) == list(range(1, 101))
    assert v["promo_net_sales"].sum() > 0 and (v["promo_net_sales"] <= v["net_sales"] + MONEY).all()


def test_store_view_matches_independent_calculation():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_kpi_store")
    exp = ref.stores(raw).assign(store_key=lambda d: d["store_id"])
    assert len(v) == 5
    assert_match(v, exp, "store_key", MEASURES + ["transactions", "sku_days", "stockout_sku_days"], MONEY, "store measures")
    assert_match(v, exp, "store_key", RATIOS + ["avg_basket_value", "net_sales_per_sqm", "stockout_rate"], RATIO, "store ratios")


def test_wilaya_view_is_weighted_not_an_average_of_store_ratios():
    """Put stores 1 and 2 in the same wilaya (on a private copy) and compare with the independent roll-up."""
    cfg, raw, dwh = smoke()
    db = loaded_sqlite()
    db._h.execute("UPDATE dwh.dim_store SET wilaya = 'Alger' WHERE store_key IN (1, 2)")
    override = {1: "Alger", 2: "Alger"}
    v = view(db, "v_kpi_wilaya")
    exp = ref.wilayas(raw, override)
    assert len(v) == 4 and v.set_index("wilaya").loc["Alger", "store_count"] == 2
    assert_match(v, exp, "wilaya", MEASURES + ["transactions", "store_count", "floor_area_sqm", "sku_days",
                                               "stockout_sku_days"], MONEY, "wilaya measures")
    assert_match(v, exp, "wilaya", RATIOS + ["avg_basket_value", "net_sales_per_sqm", "stockout_rate"], RATIO, "wilaya ratios")
    st = view(db, "v_kpi_store").set_index("store_key")
    naive = (st.loc[1, "gross_margin"] + st.loc[2, "gross_margin"]) / 2
    assert abs(v.set_index("wilaya").loc["Alger", "gross_margin"] - naive) > 1e-6      # the trap this view avoids
    db.close()


# ------------------------------------------------------------------------------------------ inventory views
def test_stock_availability_view_matches_independent_calculation():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_stock_availability_daily")
    v["inventory_date"] = pd.to_datetime(v["full_date"])
    exp = ref.availability_daily(raw).rename(columns={"store_id": "store_key", "inventory_date": "inventory_date"})
    assert len(v) == len(exp) == 5 * 90
    num = ["sku_count", "stockout_sku_count", "in_stock_sku_count", "units_sold", "closing_stock_units"]
    assert_match(v, exp, ["store_key", "inventory_date"], num, 0, "availability counts")
    assert_match(v, exp, ["store_key", "inventory_date"], ["availability_rate", "stockout_rate"], RATIO, "availability rates")
    assert v["stockout_sku_count"].sum() > 0                            # the smoke data deliberately has stockouts
    # re-aggregation rule: network stockout rate = SUM(stockouts) / SUM(sku_count), not an average of daily rates
    tot = v["stockout_sku_count"].sum() / v["sku_count"].sum()
    assert abs(tot - raw["inventory_daily"]["stockout_flag"].mean()) < 1e-12


def test_inventory_risk_view_matches_independent_calculation():
    cfg, raw, dwh = smoke()
    v = view(shared_smoke_db(), "v_inventory_risk")
    exp = ref.inventory_risk(raw).rename(columns={"store_id": "store_key", "product_id": "product_key"})
    assert len(v) == len(exp) == (raw["inventory_daily"]["inventory_date"] == raw["inventory_daily"]["inventory_date"].max()).sum()
    key = ["store_key", "product_key"]
    assert_match(v, exp, key, ["closing_stock", "on_order_qty", "units_sold_14d", "days_observed_14d", "stockout_days_14d",
                               "risk_tier_order"], 0, "risk counts")
    assert_match(v, exp, key, ["avg_daily_units_14d", "days_of_cover", "days_of_cover_incl_on_order"], RATIO, "risk cover")
    assert_match(v, exp, key, ["stock_value_at_current_cost"], MONEY, "risk value")
    assert (v.set_index(key)["risk_tier"] == exp.set_index(key)["risk_tier"]).all()
    assert {"Stockout", "Critical", "Low", "Healthy"} <= set(v["risk_tier"])        # tiers are actually exercised
    assert (v["snapshot_date"] == "2025-05-01").all()


def test_inventory_risk_tier_rules_on_hand_built_rows():
    db = Database.sqlite()
    db.apply_schema()
    h = db._h
    h.execute("INSERT INTO dwh.dim_store VALUES (1,1,'ST01','S','Alger','Algiers','Supermarket',1000,'2020-01-01')")
    for k in range(1, 8):
        h.execute("INSERT INTO dwh.dim_product VALUES (?,?,?,?,'Cat',0,'B',6,100,50)", (k, k, f"X-{k}", f"P{k}"))
    for d in (20250101, 20250102):
        h.execute("INSERT INTO dwh.dim_date VALUES (?,?,2025,1,1,'January',?,3,'Wednesday',1,0,0,NULL,0,0,0,0)",
                  (d, f"2025-01-0{d - 20250100}", d - 20250100))
    # (product, closing stock, units sold on EACH of the two days): cover = closing / 10
    cases = {1: (0, 10), 2: (20, 10), 3: (50, 10), 4: (300, 10), 5: (800, 10), 6: (40, 0), 7: (0, 0)}
    for p, (closing, sold) in cases.items():
        opening = closing + 2 * sold
        h.execute("INSERT INTO dwh.fact_inventory_daily VALUES (1,?,20250101,?,0,?,?,0,?)",
                  (p, opening, sold, opening - sold, int(opening - sold == 0)))
        h.execute("INSERT INTO dwh.fact_inventory_daily VALUES (1,?,20250102,?,0,?,?,0,?)",
                  (p, opening - sold, sold, closing, int(closing == 0)))
    h.commit()
    r = view(db, "v_inventory_risk").set_index("product_key")
    assert r.loc[1, "risk_tier"] == "Stockout"                         # closing 0 beats everything
    assert r.loc[2, "risk_tier"] == "Critical" and abs(r.loc[2, "days_of_cover"] - 2.0) < 1e-12
    assert r.loc[3, "risk_tier"] == "Low" and abs(r.loc[3, "days_of_cover"] - 5.0) < 1e-12
    assert r.loc[4, "risk_tier"] == "Healthy" and abs(r.loc[4, "days_of_cover"] - 30.0) < 1e-12
    assert r.loc[5, "risk_tier"] == "Overstock" and abs(r.loc[5, "days_of_cover"] - 80.0) < 1e-12
    assert r.loc[6, "risk_tier"] == "No recent sales" and pd.isna(r.loc[6, "days_of_cover"])
    assert r.loc[7, "risk_tier"] == "Stockout"                         # no stock and no sales is still a stockout
    assert (r["days_observed_14d"] == 2).all()                         # window is shorter than 14 days here
    db.close()


# ------------------------------------------------------------------------------------------ no double counting
def test_no_double_counting_across_every_view_and_with_promotions_and_inventory():
    cfg, raw, dwh = smoke()
    db = shared_smoke_db()
    fact = db.scalar("SELECT SUM(net_sales) FROM dwh.fact_sales")
    for name in ("v_kpi_network_daily", "v_kpi_network_monthly", "v_kpi_category_monthly", "v_kpi_product",
                 "v_kpi_store", "v_kpi_wilaya"):
        assert abs(db.scalar(f"SELECT SUM(net_sales) FROM dwh.{name}") - fact) < MONEY, name
    inv_rows = db.row_count("dwh", "fact_inventory_daily")
    assert db.scalar("SELECT SUM(sku_count) FROM dwh.v_stock_availability_daily") == inv_rows
    assert db.scalar("SELECT SUM(sku_days) FROM dwh.v_kpi_product") == inv_rows
    assert db.scalar("SELECT SUM(sku_days) FROM dwh.v_kpi_category_monthly") == inv_rows
    # promotion split comes from the sales line itself, so it equals the independent count of promoted lines
    ln = ref.sales_lines(raw)
    assert abs(db.scalar("SELECT SUM(promo_net_sales) FROM dwh.v_kpi_product") - ln.loc[ln["has_promo"], "net"].sum()) < MONEY
    # the hazard the views avoid: joining sales LINES to inventory repeats the inventory row once per line
    naive_skus = db.scalar("SELECT COUNT(*) FROM dwh.fact_sales f JOIN dwh.fact_inventory_daily i "
                           "ON i.store_key=f.store_key AND i.product_key=f.product_key AND i.date_key=f.date_key")
    assert naive_skus > inv_rows or naive_skus == db.row_count("dwh", "fact_sales")
    naive_stockouts = db.scalar("SELECT SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) FROM dwh.fact_sales f "
                                "JOIN dwh.fact_inventory_daily i ON i.store_key=f.store_key AND "
                                "i.product_key=f.product_key AND i.date_key=f.date_key")
    assert naive_stockouts != db.scalar("SELECT SUM(stockout_sku_days) FROM dwh.v_kpi_product")


def test_views_return_same_totals_on_second_dataset():
    """A second, independent dataset (two years, 3 stores) guards against fitting the smoke data only."""
    cfg, raw, dwh = two_year()
    db = shared_two_year_db()
    ln = ref.sales_lines(raw)
    for name in ("v_kpi_network_daily", "v_kpi_network_monthly", "v_kpi_product", "v_kpi_store", "v_kpi_wilaya",
                 "v_kpi_category_monthly"):
        assert abs(db.scalar(f"SELECT SUM(net_sales) FROM dwh.{name}") - ln["net"].sum()) < MONEY, name
        assert abs(db.scalar(f"SELECT SUM(gross_profit) FROM dwh.{name}") - ln["profit"].sum()) < MONEY, name
        assert db.scalar(f"SELECT SUM(units_sold) FROM dwh.{name}") == ln["quantity"].sum(), name
    assert_match(view(db, "v_kpi_product"), ref.products(raw).assign(product_key=lambda d: d["product_id"]),
                 "product_key", ["net_sales", "stockout_sku_days", "net_sales_rank"], MONEY, "two-year products")
