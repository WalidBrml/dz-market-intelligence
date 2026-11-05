"""Independent oracle generator for the Executive Overview (Milestone B1).

It recomputes every Page 1 figure with plain SQL on an in-memory SQLite copy of the CSVs, i.e.
without any code from dzmi_app, so the expected values in oracle_overview.py do not come from
the implementation they check. It is not a test (no test_ prefix) and writes nothing; it prints
the oracle dictionary:  python tests/oracle_generator_b1.py
"""

import csv
import pprint
import sqlite3
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "bi_export" / "smoke_m3_20261025"

SCENARIOS = {
    "default": {},
    "store_alger_centre": {"stores": ["Alger Centre"]},
    "category_bakery": {"categories": ["Bakery & Pastry"]},
    "format_hypermarket": {"formats": ["Hypermarket"]},
    "ramadan_window": {"start": "2025-03-01", "end": "2025-03-29"},
    "combined": {"start": "2025-04-01", "end": "2025-04-30", "stores": ["Oran Es Senia", "Blida Ville"],
                 "categories": ["Dairy & Eggs", "Personal Care"]},
    "last_day_only": {"start": "2025-05-01", "end": "2025-05-01"},
    "two_stores_one_format": {"stores": ["Alger Centre", "Blida Ville"], "formats": ["Supermarket"]},
    "empty_selection": {"stores": ["Alger Centre"], "formats": ["Supermarket"]},
}


def load(con):
    spec = {
        "dim_date": ["date_key", "full_date", "year_month_label"],
        "dim_store": ["store_key", "store_name", "store_format"],
        "dim_product": ["product_key", "product_name", "category_name"],
        "fact_sales": ["transaction_id", "date_key", "store_key", "product_key", "quantity",
                       "net_sales", "gross_profit"],
        "fact_inventory_daily": ["store_key", "product_key", "date_key", "closing_stock"],
        "v_inventory_risk": ["store_key", "product_key", "risk_tier", "stock_value_at_current_cost",
                             "snapshot_date"],
    }
    for table, cols in spec.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        with open(DATA / f"{table}.csv", newline="", encoding="utf-8") as fh:
            rows = csv.DictReader(fh)
            con.executemany(
                f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
                ([r[c] for c in cols] for r in rows))


def where(sc, with_date=True):
    clauses, params = [], []
    if with_date and "start" in sc:
        clauses.append("d.full_date >= ? AND d.full_date <= ?")
        params += [sc["start"], sc["end"]]
    for field, col in (("stores", "st.store_name"), ("categories", "p.category_name"),
                       ("formats", "st.store_format")):
        if sc.get(field):
            clauses.append(f"{col} IN ({', '.join('?' * len(sc[field]))})")
            params += sc[field]
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


SALES_FROM = ("FROM fact_sales s JOIN dim_date d ON d.date_key = s.date_key "
              "JOIN dim_store st ON st.store_key = s.store_key "
              "JOIN dim_product p ON p.product_key = s.product_key")
INV_FROM = ("FROM fact_inventory_daily i JOIN dim_date d ON d.date_key = i.date_key "
            "JOIN dim_store st ON st.store_key = i.store_key "
            "JOIN dim_product p ON p.product_key = i.product_key")
RISK_FROM = ("FROM v_inventory_risk r JOIN dim_store st ON st.store_key = r.store_key "
             "JOIN dim_product p ON p.product_key = r.product_key "
             "JOIN dim_date d ON d.date_key = d.date_key AND d.full_date = r.snapshot_date")


def oracle(con, sc):
    w, p = where(sc)
    n, net, gp, units, txn = con.execute(
        f"SELECT COUNT(*), SUM(CAST(s.net_sales AS REAL)), SUM(CAST(s.gross_profit AS REAL)), "
        f"SUM(CAST(s.quantity AS INTEGER)), COUNT(DISTINCT s.transaction_id) {SALES_FROM}{w}", p).fetchone()
    monthly = con.execute(
        f"SELECT d.year_month_label, SUM(CAST(s.net_sales AS REAL)) {SALES_FROM}{w} "
        f"GROUP BY d.year_month_label ORDER BY d.year_month_label", p).fetchall()
    by_store = con.execute(
        f"SELECT st.store_name, SUM(CAST(s.net_sales AS REAL)) AS v {SALES_FROM}{w} "
        f"GROUP BY st.store_name ORDER BY v DESC, st.store_name", p).fetchall()
    prods = con.execute(
        f"SELECT p.product_name, SUM(CAST(s.net_sales AS REAL)) AS v {SALES_FROM}{w} "
        f"GROUP BY p.product_name ORDER BY v DESC, p.product_name", p).fetchall()
    top10 = [r for r in prods if len(prods) <= 10 or r[1] >= prods[9][1]]

    sku_days, so_days = con.execute(
        f"SELECT COUNT(*), SUM(CASE WHEN CAST(i.closing_stock AS INTEGER) = 0 THEN 1 ELSE 0 END) "
        f"{INV_FROM}{w}", p).fetchone()
    so_by_store = con.execute(
        f"SELECT st.store_name, SUM(CASE WHEN CAST(i.closing_stock AS INTEGER) = 0 THEN 1.0 ELSE 0 END) "
        f"/ COUNT(*) AS v {INV_FROM}{w} GROUP BY st.store_name ORDER BY v DESC, st.store_name", p).fetchall()

    w2, p2 = where(sc, with_date=False)
    tiers = dict(con.execute(
        f"SELECT r.risk_tier, COUNT(*) FROM v_inventory_risk r "
        f"JOIN dim_store st ON st.store_key = r.store_key "
        f"JOIN dim_product p ON p.product_key = r.product_key{w2} GROUP BY r.risk_tier", p2).fetchall())
    stock_value = con.execute(
        f"SELECT SUM(CAST(r.stock_value_at_current_cost AS REAL)) FROM v_inventory_risk r "
        f"JOIN dim_store st ON st.store_key = r.store_key "
        f"JOIN dim_product p ON p.product_key = r.product_key{w2}", p2).fetchone()[0]
    return {
        "sales_rows": n,
        "net_sales": net, "gross_profit": gp, "units_sold": units, "transactions": txn,
        "gross_margin": (gp / net) if net else None,
        "average_basket": (net / txn) if net is not None and txn else None,
        "sku_days": sku_days, "stockout_sku_days": so_days if sku_days else None,
        "stockout_rate": (so_days / sku_days) if sku_days else None,
        "monthly": monthly, "by_store": by_store, "top10": top10,
        "stockout_by_store": so_by_store,
        "tiers": tiers, "tracked_skus": sum(tiers.values()),
        "stock_value": stock_value,
    }


if __name__ == "__main__":
    con = sqlite3.connect(":memory:")
    load(con)
    pprint.pprint({name: oracle(con, sc) for name, sc in SCENARIOS.items()}, width=110, sort_dicts=False)
