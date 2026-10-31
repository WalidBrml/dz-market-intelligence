"""Milestone 2 support tests: SQL portability, validation checks, date table, Power BI export and the CLI helpers."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from dzmi import cli
from dzmi.analytics.bi_export import BI_TABLES, BI_VIEWS, export_bi
from dzmi.analytics.views import ANALYTIC_VIEWS, format_view, list_views, read_view
from dzmi.errors import DatabaseError, DZMIError
from dzmi.ingestion.db import Database, parse_check_file, split_statements
from dzmi.ingestion.db_validation import run_db_checks
from dzmi.ingestion.loader import load_all
from dzmi.paths import sql_dir
from helpers import raises, shared_smoke_db, shared_two_year_db, smoke

VIEW_SQL = (sql_dir() / "schema" / "004_analytics_views.sql").read_text(encoding="utf-8")
CHECK_SQL = (sql_dir() / "validation" / "002_analytics_checks.sql").read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.strip().startswith("--")]


# ------------------------------------------------------------------------------------------ portability
def test_view_sql_uses_only_portable_constructs():
    """Cheap guard for the PostgreSQL target, which cannot be executed in every environment."""
    code = "\n".join(_code_lines(VIEW_SQL + "\n" + CHECK_SQL))
    forbidden = {
        r"\bFULL\s+OUTER\b": "FULL OUTER JOIN is missing in older SQLite",
        r"CREATE\s+OR\s+REPLACE": "not supported by SQLite",
        r"\bILIKE\b|::|\bGENERATE_SERIES\b|\bDATE_TRUNC\b|\bEXTRACT\s*\(": "PostgreSQL-only syntax",
        r"\bjulianday\b|\bstrftime\b|\bIFNULL\b|\bdatetime\s*\(": "SQLite-only syntax",
        r"\bDATEDIFF\b|\bISNULL\s*\(|\bTOP\s+\d": "other dialects",
        r"\bSUM\s*\(\s*\w+\.(stockout_flag|is_\w+)\s*\)": "SUM over a boolean fails on PostgreSQL",
        r"\w+\.stockout_flag\s*=\s*(TRUE|1|true)\b": "boolean compared with a literal: use the flag directly",
    }
    for pattern, why in forbidden.items():
        assert not re.search(pattern, code, re.I), f"{why}: {pattern}"
    lines = _code_lines(VIEW_SQL)
    for i, ln in enumerate(lines):
        if "/ NULLIF(" in ln:   # integer / integer truncates on SQLite: every ratio must start with 1.0 *
            assert "1.0 *" in ln or "1.0 *" in lines[i - 1], f"division without 1.0 *: {ln.strip()}"
    assert " / " not in "\n".join(l for l in lines if "NULLIF" not in l and "1.0 *" not in l), "unguarded division"


def test_view_file_statement_structure_and_unique_column_names():
    stmts = split_statements(VIEW_SQL)
    creates = [re.match(r"CREATE VIEW dwh\.(\w+) AS", s).group(1) for s in stmts if s.startswith("CREATE VIEW")]
    drops = [re.match(r"DROP VIEW IF EXISTS dwh\.(\w+)", s).group(1) for s in stmts if s.startswith("DROP VIEW")]
    assert sorted(creates) == sorted(ANALYTIC_VIEWS) == sorted(drops)       # every view is dropped and created once
    assert len(stmts) == 2 * len(ANALYTIC_VIEWS)
    # views must not depend on other views (PostgreSQL refuses to drop a view that another view uses)
    for s in stmts:
        if s.startswith("CREATE VIEW"):
            body = s.split(" AS", 1)[1]
            assert not re.search(r"\bdwh\.v_\w+", body), "a view must be built on tables only"
    db = shared_smoke_db()
    for name in ANALYTIC_VIEWS:
        cols = list(db.query_df(f"SELECT * FROM dwh.{name} LIMIT 0").columns)
        assert len(cols) == len(set(cols)) and not any(":" in c for c in cols), f"{name}: duplicate column names"


def test_applying_the_schema_twice_keeps_all_views():
    db = Database.sqlite()
    db.apply_schema()
    db.apply_schema()
    names = set(db.query_df("SELECT name FROM dwh.sqlite_master WHERE type='view'")["name"])
    assert set(ANALYTIC_VIEWS) | {"v_sales_daily", "v_store_product_day"} <= names
    db.close()


# ------------------------------------------------------------------------------------------ validation checks
def test_analytics_checks_parse_and_pass_on_both_datasets():
    checks = parse_check_file(CHECK_SQL)
    assert len(checks) >= 20 and len({c["name"] for c in checks}) == len(checks)
    assert all(c["sql"].upper().startswith("SELECT") and c["severity"] == "error" for c in checks)
    for db in (shared_smoke_db(), shared_two_year_db()):
        results = run_db_checks(db)
        failed = [(r.name, r.violations) for r in results if not r.passed]
        assert failed == [], failed
        assert sum(r.name.startswith("db_an_") for r in results) == len(checks)


def test_analytics_checks_detect_a_broken_calendar():
    db = Database.sqlite()
    db.apply_schema()
    cfg, raw, dwh = smoke()
    load_all(db, raw, dwh)
    db._h.execute("PRAGMA foreign_keys = OFF")
    db._h.execute("DELETE FROM dwh.dim_date WHERE date_key = 20250215")
    failed = {r.name for r in run_db_checks(db) if not r.passed}
    assert "db_an_dim_date_contiguous" in failed and "db_an_daily_net_sales_reconciles" in failed
    db.close()


def test_validate_db_on_a_pre_milestone2_database_says_how_to_upgrade():
    cfg, raw, dwh = smoke()
    db = Database.sqlite()
    db.apply_schema()
    load_all(db, raw, dwh)
    for name in ANALYTIC_VIEWS:                      # what a Milestone 1 database looks like: data but no new views
        db._h.execute(f"DROP VIEW dwh.{name}")
    try:
        run_db_checks(db)
        raise AssertionError("expected a DatabaseError")
    except DatabaseError as exc:
        assert "init-db" in str(exc) and "an_" in str(exc)
    db.apply_schema()                                # the advertised fix keeps the data and restores the views
    assert all(r.passed for r in run_db_checks(db))
    assert db.row_count("dwh", "fact_sales") == len(dwh["fact_sales"])
    db.close()


# ------------------------------------------------------------------------------------------ date table
def test_date_table_supports_time_intelligence():
    for db, n_days in ((shared_smoke_db(), 90), (shared_two_year_db(), 731)):
        d = db.query_df("SELECT * FROM dwh.v_dim_date_bi ORDER BY date_key")
        dates = pd.to_datetime(d["full_date"])
        assert len(d) == n_days and d["full_date"].is_unique and d["date_key"].is_unique
        assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()            # contiguous: no gaps
        assert (d["date_key"] == dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day).all()
        assert list(d["day_seq"]) == list(range(1, n_days + 1))
        assert d["is_last_loaded_date"].sum() == 1 and d["is_last_loaded_date"].iloc[-1] == 1
        assert (d["year_month"] == dates.dt.year * 100 + dates.dt.month).all()
        assert (d["days_in_month"] == dates.dt.days_in_month).all()
        full = d.groupby("year_month")["full_date"].transform("size") == d["days_in_month"]
        assert ((d["is_complete_month"] == 1) == full).all()
        label = d["year_month_label"]
        assert label.str.fullmatch(r"\d{4}-\d{2}").all() and label.is_monotonic_increasing
    d = shared_two_year_db().query_df("SELECT * FROM dwh.v_dim_date_bi")
    dt = pd.to_datetime(d["full_date"])
    keys = set(d["date_key"])
    has_prior = d["prior_year_date_key"].isin(keys)
    leap = (dt.dt.month == 2) & (dt.dt.day == 29)
    assert has_prior[dt.dt.year == 2024].sum() == (dt.dt.year == 2024).sum() - leap.sum()   # 29 Feb has no prior-year day
    assert not has_prior[dt.dt.year == 2023].any()
    base = db_columns(shared_smoke_db(), "dim_date")
    assert base <= db_columns(shared_smoke_db(), "v_dim_date_bi")               # the Power BI date table is a superset


def db_columns(db, name: str) -> set[str]:
    return set(db.query_df(f"SELECT * FROM dwh.{name} LIMIT 0").columns)


# ------------------------------------------------------------------------------------------ export and CLI helpers
def test_export_bi_writes_consistent_csv_files_for_power_bi():
    db = shared_smoke_db()
    with tempfile.TemporaryDirectory() as d:
        counts = export_bi(db, Path(d), chunksize=5000)                      # small chunks exercise the chunk loop
        assert set(counts) == {n for n, _, _ in BI_TABLES + BI_VIEWS}
        for name in counts:
            assert (Path(d) / f"{name}.csv").exists()
        assert counts["fact_sales"] == db.row_count("dwh", "fact_sales")
        assert counts["fact_inventory_daily"] == db.row_count("dwh", "fact_inventory_daily")
        assert counts["dim_date"] == 90 and counts["v_inventory_risk"] == db.row_count("dwh", "v_inventory_risk")
        raw = pd.read_csv(Path(d) / "fact_sales.csv", dtype=str)
        assert len(raw) == counts["fact_sales"]
        for col in ("customer_key", "promotion_key", "store_key", "date_key"):
            assert not raw[col].dropna().str.contains(r"\.", regex=True).any(), f"{col}: keys must not look like 12.0"
        assert raw["customer_key"].isna().any()                               # anonymous baskets stay empty, not 0
        sales = pd.read_csv(Path(d) / "fact_sales.csv")
        assert abs(sales["net_sales"].sum() - db.scalar("SELECT SUM(net_sales) FROM dwh.fact_sales")) < 0.01
        inv = pd.read_csv(Path(d) / "fact_inventory_daily.csv", dtype=str)
        assert set(inv["stockout_flag"]) == {"True", "False"}
        assert set(pd.read_csv(Path(d) / "dim_product.csv", dtype=str)["is_perishable"]) <= {"True", "False"}
        dim = pd.read_csv(Path(d) / "dim_date.csv")
        assert "day_seq" in dim.columns and "year_month_label" in dim.columns
        manifest = json.loads((Path(d) / "bi_export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["synthetic"] is True and manifest["rows"] == counts


def test_read_view_and_format_view_helpers():
    db = shared_smoke_db()
    raises(DZMIError, read_view, db, "fact_sales; DROP TABLE dwh.fact_sales")   # only allow-listed names reach SQL
    raises(DZMIError, read_view, db, "nope")
    prod = read_view(db, "v_kpi_product")
    assert list(prod["net_sales_rank"]) == list(range(1, 101))                  # default order
    txt = format_view(prod, limit=3, columns=["net_sales_rank", "product_name", "net_sales"])
    assert "(3 of 100 rows)" in txt and "product_name" in txt and "gross_margin" not in txt
    asc = format_view(prod, limit=1, order_by="net_sales", columns=["net_sales"])
    desc = format_view(prod, limit=1, order_by="net_sales", descending=True, columns=["net_sales"])
    assert asc != desc
    raises(DZMIError, format_view, prod, 5, "no_such_column")
    raises(DZMIError, format_view, prod, 5, None, False, ["no_such_column"])
    assert [n for n, _ in list_views()] == list(ANALYTIC_VIEWS)


def test_cli_query_and_export_on_a_real_sqlite_file_and_release_the_files():
    """File-backed SQLite end to end through the CLI functions, then delete the folder (Windows locks open files)."""
    cfg, raw, dwh = smoke()
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "t.sqlite"
        with Database.sqlite(base) as db:
            db.apply_schema()
            load_all(db, raw, dwh)
        ns = argparse.Namespace(profile="smoke", backend="sqlite", sqlite_path=str(base), list=False,
                                view="v_kpi_network_monthly", limit=20, order_by=None, desc=False,
                                columns="year_month,net_sales,yoy_net_sales_growth", csv=None, out=str(Path(d) / "bi"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli.cmd_query(ns) == 0
        assert "202503" in buf.getvalue() and "(4 of 4 rows)" in buf.getvalue()
        ns.csv = str(Path(d) / "out" / "monthly.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.cmd_query(ns) == 0
        assert len(pd.read_csv(ns.csv)) == 4
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.cmd_export_bi(ns) == 0
        assert (Path(d) / "bi" / "fact_sales.csv").exists()
        ns.view = "nope"
        raises(DZMIError, cli.cmd_query, ns)
        ns.view, ns.sqlite_path = "v_kpi_store", str(Path(d) / "missing.sqlite")
        raises(DatabaseError, cli.cmd_query, ns)                              # must not create an empty database
        assert not list(Path(d).glob("missing*"))
    # leaving the with-block means TemporaryDirectory could delete the SQLite files: no connection was leaked


def test_cli_parser_exposes_the_new_commands():
    p = cli.build_parser()
    a = p.parse_args(["query", "--view", "v_kpi_store", "--backend", "sqlite", "--limit", "5", "--desc"])
    assert a.func is cli.cmd_query and a.limit == 5 and a.desc and a.backend == "sqlite"
    a = p.parse_args(["export-bi", "--profile", "default", "--out", "x"])
    assert a.func is cli.cmd_export_bi and a.profile == "default" and a.out == "x"
    assert not hasattr(a, "reset")                                            # read-only commands cannot reset a database
