"""Independent oracle generator for Sales Intelligence (Milestone B2).

Recomputes every Page 2 figure with plain SQL on an in-memory SQLite copy of the CSVs, without
any dzmi_app code, so the expected values in oracle_sales_intelligence.py do not come from the
implementation they check. Not a test (no test_ prefix); writes nothing, prints the dictionary:

    python tests/oracle_generator_b2.py

Month-over-month follows the audited DAX (_Measures.tmdl, 'Net Sales MoM %'); the DATEADD(-1 MONTH)
shift is encoded here as: first day -> same day one month earlier (clamped to the month's last
day); last day -> same day one month earlier, or the last day of that month when the original is
a month-end. That month-end behaviour of DATEADD is an assumption (see the B2 report).
"""

import calendar
import csv
import pprint
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "bi_export" / "smoke_m3_20261025"

SCENARIOS = {
    "default": {},
    "wilaya_alger": {"wilayas": ["Alger"]},
    "category_dairy": {"categories": ["Dairy & Eggs"]},
    "brand_a": {"brands": ["Brand A"]},
    "two_wilayas": {"wilayas": ["Oran", "Blida"]},
    "feb_mar": {"start": "2025-02-01", "end": "2025-03-31"},
    "partial_range": {"start": "2025-03-10", "end": "2025-04-20"},
    "last_day_only": {"start": "2025-05-01", "end": "2025-05-01"},
    "combined": {"start": "2025-03-01", "end": "2025-04-30", "wilayas": ["Oran", "Blida"],
                 "categories": ["Dairy & Eggs", "Personal Care"], "brands": ["Brand A", "Brand B", "Brand C"]},
    "empty_selection": {"wilayas": ["Alger"], "categories": ["Dairy & Eggs"], "brands": ["Brand E"]},
}


def load(con):
    spec = {
        "dim_date": ["date_key", "full_date", "year_month", "year_month_label", "day_name", "day_of_week",
                     "is_complete_month"],
        "dim_store": ["store_key", "store_name", "wilaya", "floor_area_sqm"],
        "dim_product": ["product_key", "product_name", "category_name", "brand"],
        "dim_customer": ["customer_key", "household_type"],
        "fact_sales": ["transaction_id", "date_key", "store_key", "product_key", "customer_key",
                       "promotion_key", "quantity", "net_sales", "gross_profit"],
    }
    for table, cols in spec.items():
        con.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
        with open(DATA / f"{table}.csv", newline="", encoding="utf-8") as fh:
            con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
                            ([r[c] for c in cols] for r in csv.DictReader(fh)))


FROM = ("FROM fact_sales s JOIN dim_date d ON d.date_key = s.date_key "
        "JOIN dim_store st ON st.store_key = s.store_key "
        "JOIN dim_product p ON p.product_key = s.product_key "
        "LEFT JOIN dim_customer c ON c.customer_key = s.customer_key")


def where(sc, lo=None, hi=None, with_date=True, extra=None):
    clauses, params = [], []
    if with_date:
        start, end = lo or sc.get("start"), hi or sc.get("end")
        if start:
            clauses.append("d.full_date >= ? AND d.full_date <= ?")
            params += [start, end]
    for field, col in (("wilayas", "st.wilaya"), ("categories", "p.category_name"), ("brands", "p.brand")):
        if sc.get(field):
            clauses.append(f"{col} IN ({', '.join('?' * len(sc[field]))})")
            params += sc[field]
    for clause, val in (extra or []):
        clauses.append(clause)
        params.append(val)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def one_month_back(d: date, month_end_rule: bool) -> date:
    y, m = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    last = calendar.monthrange(y, m)[1]
    if month_end_rule and d.day == calendar.monthrange(d.year, d.month)[1]:
        return date(y, m, last)
    return date(y, m, min(d.day, last))


def measures(con, sc, lo=None, hi=None, extra=None):
    w, p = where(sc, lo, hi, extra=extra)
    return con.execute(
        f"SELECT COUNT(*), SUM(CAST(s.net_sales AS REAL)), SUM(CAST(s.gross_profit AS REAL)), "
        f"SUM(CAST(s.quantity AS INTEGER)), COUNT(DISTINCT s.date_key), "
        f"SUM(CASE WHEN s.promotion_key <> '' THEN CAST(s.net_sales AS REAL) END) {FROM}{w}", p).fetchone()


def row_values(n, net, gp, units, days, promo):
    return {
        "net_sales": net, "gross_profit": gp, "units_sold": units,
        "gross_margin": (gp / net) if net else None,
        "avg_daily_net_sales": (net / days) if net is not None and days else None,
        "avg_selling_price": (net / units) if net is not None and units else None,
        "promo_net_sales": promo,
        "promo_share": (promo / net) if promo is not None and net else None,
    }


def mom(con, sc, start, end):
    """MoM for the dates [start, end] (all dim_date rows in that interval)."""
    cur_ok = con.execute("SELECT MIN(CAST(is_complete_month AS INTEGER)) FROM dim_date "
                         "WHERE full_date >= ? AND full_date <= ?", (start, end)).fetchone()[0]
    s0, e0 = date.fromisoformat(start), date.fromisoformat(end)
    ps, pe = one_month_back(s0, False).isoformat(), one_month_back(e0, True).isoformat()
    prev_ok = con.execute("SELECT MIN(CAST(is_complete_month AS INTEGER)) FROM dim_date "
                          "WHERE full_date >= ? AND full_date <= ?", (ps, pe)).fetchone()[0]
    w, p = where(sc, ps, pe)
    prev = con.execute(f"SELECT SUM(CAST(s.net_sales AS REAL)) {FROM}{w}", p).fetchone()[0]
    w, p = where(sc, start, end)
    cur = con.execute(f"SELECT SUM(CAST(s.net_sales AS REAL)) {FROM}{w}", p).fetchone()[0]
    if cur_ok == 1 and prev_ok == 1 and prev is not None and prev != 0:
        return ((cur or 0.0) - prev) / prev, (ps, pe)
    return None, (ps, pe)


def groups(con, sc, select, group_by, order=None):
    w, p = where(sc)
    return con.execute(f"SELECT {select} {FROM}{w} GROUP BY {group_by}" + (f" ORDER BY {order}" if order else ""), p).fetchall()


def oracle(con, sc):
    w, p = where(sc)
    n, net, gp, units, days, promo = measures(con, sc)
    out = {"sales_rows": n, "total": row_values(n, net, gp, units, days, promo)}
    out["daily"] = con.execute(
        f"SELECT d.full_date, SUM(CAST(s.net_sales AS REAL)), SUM(CAST(s.gross_profit AS REAL)) {FROM}{w} "
        f"GROUP BY d.full_date ORDER BY d.full_date", p).fetchall()

    lo, hi = sc.get("start", "0000-00-00"), sc.get("end", "9999-99-99")
    months = con.execute("SELECT year_month_label, MIN(full_date), MAX(full_date) FROM dim_date "
                         "WHERE full_date >= ? AND full_date <= ? GROUP BY year_month_label ORDER BY year_month",
                         (lo, hi)).fetchall()
    rows = []
    for label, first, last in months:
        n_m, net_m, gp_m, u_m, d_m, pr_m = measures(con, sc, first, last)
        m, interval = mom(con, sc, first, last)
        r = row_values(n_m, net_m, gp_m, u_m, d_m, pr_m)
        r.update({"year_month_label": label, "mom": m, "prior_interval": interval})
        if any(r[k] is not None for k in ("net_sales", "mom", "gross_margin", "avg_daily_net_sales", "units_sold")):
            rows.append(r)
    out["monthly"] = rows
    first_all = con.execute("SELECT MIN(full_date), MAX(full_date) FROM dim_date WHERE full_date >= ? AND full_date <= ?",
                            (lo, hi)).fetchone()
    out["total"]["mom"], out["total"]["prior_interval"] = mom(con, sc, first_all[0], first_all[1])

    out["by_category"] = groups(con, sc, "p.category_name, SUM(CAST(s.net_sales AS REAL)) v", "p.category_name",
                                "v DESC, p.category_name")
    out["gm_by_category"] = groups(
        con, sc, "p.category_name, SUM(CAST(s.gross_profit AS REAL)) / SUM(CAST(s.net_sales AS REAL)) v",
        "p.category_name", "v DESC, p.category_name")
    out["sqm_by_store"] = groups(
        con, sc, "st.store_name, SUM(CAST(s.net_sales AS REAL)) / CAST(st.floor_area_sqm AS REAL) v",
        "st.store_name, st.floor_area_sqm", "v DESC, st.store_name")
    out["household"] = groups(
        con, sc, "COALESCE(c.household_type, '(Blank)'), SUM(CAST(s.net_sales AS REAL)) v",
        "COALESCE(c.household_type, '(Blank)')", "v DESC, 1")
    out["weekday"] = groups(con, sc, "d.day_name, CAST(d.day_of_week AS INTEGER), SUM(CAST(s.net_sales AS REAL))",
                            "d.day_name, d.day_of_week", "CAST(d.day_of_week AS INTEGER)")
    prods = []
    for name, cat, n_p, net_p, gp_p, u_p, d_p, pr_p in con.execute(
            f"SELECT p.product_name, p.category_name, COUNT(*), SUM(CAST(s.net_sales AS REAL)), "
            f"SUM(CAST(s.gross_profit AS REAL)), SUM(CAST(s.quantity AS INTEGER)), COUNT(DISTINCT s.date_key), "
            f"SUM(CASE WHEN s.promotion_key <> '' THEN CAST(s.net_sales AS REAL) END) {FROM}{w} "
            f"GROUP BY p.product_name, p.category_name ORDER BY p.category_name, SUM(CAST(s.net_sales AS REAL)) DESC, p.product_name",
            p).fetchall():
        r = row_values(n_p, net_p, gp_p, u_p, d_p, pr_p)
        r.update({"product_name": name, "category_name": cat})
        prods.append(r)
    out["products"] = prods
    return out


if __name__ == "__main__":
    con = sqlite3.connect(":memory:")
    load(con)
    pprint.pprint({name: oracle(con, sc) for name, sc in SCENARIOS.items()}, width=110, sort_dicts=False)
