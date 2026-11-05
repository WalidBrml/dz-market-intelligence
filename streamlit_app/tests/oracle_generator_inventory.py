"""Independent oracle generator for Inventory Intelligence (Milestone B3): plain SQL on an in-memory
SQLite copy of the CSVs, no dzmi_app code. Not a test; prints the oracle dictionary:

    python tests/oracle_generator_inventory.py

Rules encoded here (from _Measures.tmdl): the date filter reaches fact_inventory_daily only;
'Closing stock' is read on the LAST date of the range (LASTDATE); 'Avg Daily Units 14D' uses the 14
calendar days ending on that date, not limited by the range start; v_inventory_risk has no date
relationship; the risk-tier slicer is applied to the SKU table only (approved deviation V6).
"""

import csv
import pprint
import sqlite3
import statistics
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "bi_export" / "smoke_m3_20261025"

SCENARIOS = {
    "default": {},
    "store_alger": {"stores": ["Alger Centre"]},
    "category_dairy": {"categories": ["Dairy & Eggs"]},
    "one_product": {"products": ["Oranges 2 kg (Brand B)"]},
    "ramadan_window": {"start": "2025-03-01", "end": "2025-03-29"},
    "last_day_only": {"start": "2025-05-01", "end": "2025-05-01"},
    "first_days": {"start": "2025-02-01", "end": "2025-02-05"},
    "tier_filter": {"tiers": ["Critical", "Low"]},
    "combined": {"start": "2025-04-01", "end": "2025-04-30", "stores": ["Oran Es Senia"],
                 "categories": ["Dairy & Eggs"]},
    "empty_selection": {"stores": ["Alger Centre"], "products": ["Eggs 1 L (Brand I)"]},
}
TIER_ORDER = ["Stockout", "Critical", "Low", "Healthy", "Overstock", "No recent sales"]


def load(con):
    spec = {
        "dim_date": ["date_key", "full_date"],
        "dim_store": ["store_key", "store_name"],
        "dim_product": ["product_key", "product_name", "category_name"],
        "fact_inventory_daily": ["store_key", "product_key", "date_key", "units_sold", "closing_stock"],
        "v_inventory_risk": ["store_key", "product_key", "store_code", "sku", "risk_tier", "risk_tier_order",
                             "closing_stock", "avg_daily_units_14d", "days_of_cover",
                             "stock_value_at_current_cost"],
    }
    for table, cols in spec.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        with open(DATA / f"{table}.csv", newline="", encoding="utf-8") as fh:
            con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
                            ([r[c] for c in cols] for r in csv.DictReader(fh)))


INV = ("FROM fact_inventory_daily i JOIN dim_date d ON d.date_key = i.date_key "
       "JOIN dim_store st ON st.store_key = i.store_key JOIN dim_product p ON p.product_key = i.product_key")
RISK = ("FROM v_inventory_risk r JOIN dim_store st ON st.store_key = r.store_key "
        "JOIN dim_product p ON p.product_key = r.product_key")


def where(sc, lo=None, hi=None, date_filter=True, tiers=False):
    c, p = [], []
    if date_filter and (lo or sc.get("start")):
        c.append("d.full_date >= ? AND d.full_date <= ?")
        p += [lo or sc["start"], hi or sc["end"]]
    for key, col in (("stores", "st.store_name"), ("categories", "p.category_name"), ("products", "p.product_name")):
        if sc.get(key):
            c.append(f"{col} IN ({', '.join('?' * len(sc[key]))})")
            p += sc[key]
    if tiers and sc.get("tiers"):
        c.append(f"r.risk_tier IN ({', '.join('?' * len(sc['tiers']))})")
        p += sc["tiers"]
    return (" WHERE " + " AND ".join(c)) if c else "", p


def oracle(con, sc):
    w, p = where(sc)
    n, so = con.execute(f"SELECT COUNT(*), SUM(CASE WHEN CAST(i.closing_stock AS INTEGER)=0 THEN 1 ELSE 0 END) {INV}{w}", p).fetchone()
    out = {"sku_days": n, "stockout_sku_days": so if n else None,
           "stockout_rate": (so / n) if n else None,
           "availability_rate": ((n - (so or 0)) / n) if n else None}
    lo, hi = sc.get("start", "0000"), sc.get("end", "9999")
    end = con.execute("SELECT MAX(full_date) FROM dim_date WHERE full_date >= ? AND full_date <= ?", (lo, hi)).fetchone()[0]
    out["last_date"] = end
    w2, p2 = where(sc, end, end)
    out["closing_stock"] = con.execute(f"SELECT SUM(CAST(i.closing_stock AS INTEGER)) {INV}{w2}", p2).fetchone()[0]
    start_w = (date.fromisoformat(end) - timedelta(days=13)).isoformat()
    w3, p3 = where(sc, start_w, end)
    units, days = con.execute(f"SELECT SUM(CAST(i.units_sold AS INTEGER)), COUNT(DISTINCT i.date_key) {INV}{w3}", p3).fetchone()
    out["avg_daily_units_14d"] = (units / days) if units is not None and days else None
    out["days_of_cover"] = (out["closing_stock"] / out["avg_daily_units_14d"]) if out["closing_stock"] is not None and out["avg_daily_units_14d"] else None
    out["daily"] = con.execute(f"SELECT d.full_date, SUM(CASE WHEN CAST(i.closing_stock AS INTEGER)=0 THEN 1.0 ELSE 0 END)/COUNT(*) {INV}{w} GROUP BY d.full_date ORDER BY d.full_date", p).fetchall()
    out["heatmap"] = con.execute(f"SELECT p.category_name, st.store_name, SUM(CASE WHEN CAST(i.closing_stock AS INTEGER)=0 THEN 1.0 ELSE 0 END)/COUNT(*) {INV}{w} GROUP BY p.category_name, st.store_name ORDER BY p.category_name, st.store_name", p).fetchall()

    wr, pr = where(sc, date_filter=False)
    tiers = dict(con.execute(f"SELECT r.risk_tier, COUNT(*) {RISK}{wr} GROUP BY r.risk_tier", pr).fetchall())
    out["tiers"] = tiers
    total = sum(tiers.values())
    atrisk = sum(tiers.get(t, 0) for t in ("Stockout", "Critical", "Low"))
    out["tracked_skus"] = total
    out["at_risk_share"] = (atrisk / total) if total else None
    covers = [float(x[0]) for x in con.execute(f"SELECT r.days_of_cover {RISK}{wr}", pr).fetchall() if x[0] != ""]
    out["median_days_of_cover"] = statistics.median(covers) if covers else None
    wt, pt = where(sc, date_filter=False, tiers=True)
    out["risk_table"] = con.execute(
        f"SELECT r.risk_tier, r.store_code, r.sku, p.product_name, CAST(r.closing_stock AS INTEGER), "
        f"CAST(NULLIF(r.avg_daily_units_14d,'') AS REAL), CAST(NULLIF(r.days_of_cover,'') AS REAL), "
        f"CAST(r.stock_value_at_current_cost AS REAL) {RISK}{wt} "
        f"ORDER BY CAST(r.risk_tier_order AS INTEGER), CAST(NULLIF(r.days_of_cover,'') AS REAL) IS NULL, "
        f"CAST(NULLIF(r.days_of_cover,'') AS REAL), r.store_code, r.sku", pt).fetchall()
    return out


if __name__ == "__main__":
    con = sqlite3.connect(":memory:")
    load(con)
    pprint.pprint({k: oracle(con, v) for k, v in SCENARIOS.items()}, width=110, sort_dicts=False)
