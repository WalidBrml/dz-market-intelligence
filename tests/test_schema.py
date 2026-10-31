"""Schema constraint tests on the SQLite fallback (same DDL file as PostgreSQL)."""
import sqlite3

from dzmi.ingestion.db import Database, parse_check_file, split_statements, to_sqlite_sql
from dzmi.paths import sql_dir
from helpers import loaded_sqlite, raises


def _fresh() -> Database:
    db = Database.sqlite()
    db.apply_schema()
    return db


def _exec(db, sql):
    db._h.execute(sql)


def _seed_dims(db):
    for s in (
        "INSERT INTO dwh.dim_date VALUES (20250101,'2025-01-01',2025,1,1,'January',1,3,'Wednesday',1,0,0,NULL,0,0,0,0)",
        "INSERT INTO dwh.dim_store VALUES (1,1,'ST01','S','Alger','Algiers','Supermarket',1000,'2020-01-01')",
        "INSERT INTO dwh.dim_product VALUES (1,1,'X-1','P','Cat',0,'Brand A',6,100,80)",
    ):
        _exec(db, s)


def test_schema_applies_and_is_idempotent():
    db = _fresh()
    db.apply_schema()                       # second application must not fail
    names = set(db.query_df("SELECT name FROM dwh.sqlite_master WHERE type='table'")["name"])
    assert {"dim_date", "dim_store", "dim_product", "dim_customer", "dim_promotion", "fact_sales",
            "fact_inventory_daily", "fact_promotions", "fact_demand_forecast", "model_run"} <= names


def test_schema_file_uses_numeric_for_money_and_has_pks():
    ddl = (sql_dir() / "schema" / "001_schema.sql").read_text()
    assert "NUMERIC(12,2)" in ddl and "FLOAT" not in ddl.upper() and "DOUBLE" not in ddl.upper()
    for t in ("dwh.fact_sales", "dwh.fact_inventory_daily", "dwh.dim_store"):
        assert f"CREATE TABLE IF NOT EXISTS {t}" in ddl


def test_foreign_key_rejects_orphan_fact():
    db = _fresh()
    _seed_dims(db)
    raises(sqlite3.IntegrityError, _exec, db,
           "INSERT INTO dwh.fact_inventory_daily VALUES (99,1,20250101,1,0,0,1,0,0)")


def test_inventory_balance_check_constraint():
    db = _fresh()
    _seed_dims(db)
    raises(sqlite3.IntegrityError, _exec, db,
           "INSERT INTO dwh.fact_inventory_daily VALUES (1,1,20250101,10,0,3,5,0,0)")   # 10-3 != 5


def test_stockout_flag_must_match_closing_stock():
    db = _fresh()
    _seed_dims(db)
    raises(sqlite3.IntegrityError, _exec, db,
           "INSERT INTO dwh.fact_inventory_daily VALUES (1,1,20250101,5,0,5,0,0,0)")    # closing 0 but flag false


def test_inventory_primary_key_prevents_duplicates():
    db = _fresh()
    _seed_dims(db)
    _exec(db, "INSERT INTO dwh.fact_inventory_daily VALUES (1,1,20250101,5,0,1,4,0,0)")
    raises(sqlite3.IntegrityError, _exec, db, "INSERT INTO dwh.fact_inventory_daily VALUES (1,1,20250101,5,0,1,4,0,0)")


def test_fact_sales_revenue_arithmetic_check_and_unique_line():
    db = _fresh()
    _seed_dims(db)
    ok = "INSERT INTO dwh.fact_sales VALUES (1,1,1,20250101,1,1,NULL,NULL,2,100,80,200,0,200,160,40)"
    _exec(db, ok)
    raises(sqlite3.IntegrityError, _exec, db,   # duplicate (transaction_id, line_no)
           "INSERT INTO dwh.fact_sales VALUES (2,1,1,20250101,1,1,NULL,NULL,2,100,80,200,0,200,160,40)")
    raises(sqlite3.IntegrityError, _exec, db,   # gross_sales != quantity * unit_price
           "INSERT INTO dwh.fact_sales VALUES (3,2,1,20250101,1,1,NULL,NULL,2,100,80,999,0,999,160,839)")


def test_negative_quantity_rejected():
    db = _fresh()
    _seed_dims(db)
    raises(sqlite3.IntegrityError, _exec, db,
           "INSERT INTO dwh.fact_sales VALUES (1,1,1,20250101,1,1,NULL,NULL,-1,100,80,-100,0,-100,-80,-20)")


def test_promotion_end_before_start_rejected():
    db = _fresh()
    raises(sqlite3.IntegrityError, _exec, db,
           "INSERT INTO ops.promotion VALUES (1,'x','Clearance',10,'2025-03-10','2025-03-01')")


def test_views_exist_and_no_double_counting():
    db = loaded_sqlite()
    direct = db.scalar("SELECT SUM(net_sales) FROM dwh.fact_sales")
    via_view = db.scalar("SELECT SUM(net_sales) FROM dwh.v_store_product_day")
    assert abs(direct - via_view) < 0.01
    assert db.row_count("dwh", "fact_inventory_daily") == db.scalar("SELECT COUNT(*) FROM dwh.v_store_product_day")
    # the naive line-level join fans out inventory (documented hazard)
    naive = db.scalar("SELECT SUM(i.closing_stock) FROM dwh.fact_sales f JOIN dwh.fact_inventory_daily i "
                      "ON i.store_key=f.store_key AND i.product_key=f.product_key AND i.date_key=f.date_key")
    safe = db.scalar("SELECT SUM(closing_stock) FROM dwh.fact_inventory_daily")
    assert naive > safe


def test_sql_helpers_rewrite_for_sqlite():
    out = to_sqlite_sql("CREATE SCHEMA IF NOT EXISTS ops;\n"
                        "CREATE TABLE IF NOT EXISTS ops.a (id INTEGER REFERENCES ops.b (id));\n"
                        "CREATE INDEX IF NOT EXISTS ix_a ON ops.a (id);")
    assert "CREATE SCHEMA" not in out and "REFERENCES b (id)" in out and "INDEX IF NOT EXISTS ops.ix_a ON a" in out
    assert len(split_statements("SELECT 1; -- c\nSELECT 2;")) == 2


def test_check_file_parses_every_check_with_metadata():
    checks = parse_check_file((sql_dir() / "validation" / "001_integrity_checks.sql").read_text())
    assert len(checks) >= 20
    assert all(c["severity"] in ("error", "warning") and c["sql"].upper().startswith("SELECT") for c in checks)
    assert len({c["name"] for c in checks}) == len(checks)
