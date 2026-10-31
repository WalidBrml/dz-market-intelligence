"""End-to-end: generate -> warehouse -> quality -> load -> SQL validation (SQLite fallback)."""
import json
import tempfile
from pathlib import Path

from dzmi.data_generation.io import read_raw, write_raw
from dzmi.errors import DatabaseError
from dzmi.ingestion.db import Database
from dzmi.ingestion.db_validation import run_db_checks
from dzmi.ingestion.loader import load_all
from dzmi.ingestion.quality import run_quality_checks, summarize, write_report
from dzmi.ingestion.warehouse import build_warehouse
from helpers import loaded_sqlite, raises, smoke


def test_pre_load_quality_checks_all_pass():
    _, raw, dwh = smoke()
    results = run_quality_checks(raw, dwh)
    bad = [(r.name, r.violations) for r in results if not r.passed]
    assert bad == [] and summarize(results)["ok"]
    assert len(results) >= 40


def test_database_validation_all_pass_after_load():
    db = loaded_sqlite()
    results = run_db_checks(db)
    bad = [(r.name, r.violations) for r in results if not r.passed]
    assert bad == [] and len(results) >= 20


def test_row_counts_match_between_source_and_database():
    _, raw, dwh = smoke()
    db = loaded_sqlite()
    assert db.row_count("dwh", "fact_sales") == len(raw["sales_transaction_line"])
    assert db.row_count("dwh", "fact_inventory_daily") == len(raw["inventory_daily"])
    assert db.row_count("dwh", "dim_date") == 90
    net_db = db.scalar("SELECT SUM(net_sales) FROM dwh.fact_sales")
    assert abs(net_db - dwh["fact_sales"]["net_sales"].sum()) < 0.01


def test_money_survives_load_without_float_drift():
    _, raw, dwh = smoke()
    db = loaded_sqlite()
    gp = db.scalar("SELECT SUM(gross_profit) FROM dwh.fact_sales")
    assert abs(gp - dwh["fact_sales"]["gross_profit"].sum()) < 0.01
    assert db.scalar("SELECT COUNT(*) FROM dwh.fact_sales WHERE gross_profit > net_sales") == 0


def test_second_load_into_populated_database_is_refused_and_reset_recovers():
    cfg, raw, dwh = smoke()
    db = loaded_sqlite()
    e = raises(DatabaseError, load_all, db, raw, dwh)
    assert "--reset" in str(e)
    db.reset()
    db.apply_schema()
    load_all(db, raw, dwh)
    assert db.row_count("dwh", "fact_sales") == len(dwh["fact_sales"])


def test_csv_round_trip_preserves_data_and_manifest_flags_synthetic():
    cfg, raw, dwh = smoke()
    with tempfile.TemporaryDirectory() as d:
        write_raw(raw, Path(d), cfg)
        manifest = json.loads((Path(d) / "manifest.json").read_text())
        assert manifest["synthetic"] is True and manifest["rows"]["store"] == 5
        back = read_raw(Path(d))
        dwh2 = build_warehouse(back, cfg)
        assert abs(dwh2["fact_sales"]["net_sales"].sum() - dwh["fact_sales"]["net_sales"].sum()) < 0.01
        assert summarize(run_quality_checks(back, dwh2))["ok"]


def test_report_files_written_with_disclaimer():
    _, raw, dwh = smoke()
    with tempfile.TemporaryDirectory() as d:
        jp, mp = write_report(run_quality_checks(raw, dwh), Path(d), "smoke")
        assert "synthetic" in mp.read_text().lower() and json.loads(jp.read_text())["summary"]["ok"]


def test_read_raw_missing_folder_gives_actionable_error():
    from dzmi.errors import DZMIError
    e = raises(DZMIError, read_raw, Path("/nonexistent/path"))
    assert "dzmi generate" in str(e)


def test_file_backed_sqlite_persists_and_reloads():
    cfg, raw, dwh = smoke()
    with tempfile.TemporaryDirectory() as d:
        # Windows cannot delete a file that a connection still holds open, so every connection
        # must be closed BEFORE the temporary directory is removed (even if an assertion fails).
        with Database.sqlite(Path(d) / "t.sqlite") as db:
            db.apply_schema()
            load_all(db, raw, dwh)
        with Database.sqlite(Path(d) / "t.sqlite") as db2:
            assert db2.row_count("dwh", "fact_sales") == len(dwh["fact_sales"])
