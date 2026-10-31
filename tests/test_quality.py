"""The quality checks must DETECT problems, not just pass on clean data."""
from dzmi.errors import DataQualityError
from dzmi.ingestion.db import Database
from dzmi.ingestion.db_validation import run_db_checks
from dzmi.ingestion.quality import assert_quality, run_quality_checks
from helpers import copies, raises


def _failed(raw, dwh):
    return {r.name for r in run_quality_checks(raw, dwh) if not r.passed and r.severity == "error"}


def test_detects_revenue_mismatch():
    _, raw, dwh = copies()
    dwh["fact_sales"].loc[0, "net_sales"] += 50
    assert {"sales_net_arithmetic", "revenue_reconciliation"} <= _failed(raw, dwh)


def test_detects_duplicate_transaction_line():
    _, raw, dwh = copies()
    dwh["fact_sales"] = __import__("pandas").concat([dwh["fact_sales"], dwh["fact_sales"].iloc[[0]]], ignore_index=True)
    assert "unique_transaction_line" in _failed(raw, dwh)


def test_detects_orphan_foreign_key():
    _, raw, dwh = copies()
    dwh["fact_sales"].loc[0, "store_key"] = 999
    assert "fk_sales_store" in _failed(raw, dwh)


def test_detects_broken_stock_balance_and_continuity():
    _, raw, dwh = copies()
    dwh["fact_inventory_daily"].loc[5, "closing_stock"] += 3
    f = _failed(raw, dwh)
    assert "stock_balance" in f and "stock_continuity" in f


def test_detects_missing_inventory_day():
    _, raw, dwh = copies()
    dwh["fact_inventory_daily"] = dwh["fact_inventory_daily"].iloc[1:].reset_index(drop=True)
    assert {"inventory_completeness", "inventory_series_length"} <= _failed(raw, dwh)


def test_detects_sales_inventory_disagreement():
    _, raw, dwh = copies()
    dwh["fact_inventory_daily"].loc[dwh["fact_inventory_daily"]["units_sold"] > 0, "units_sold"] += 0  # no-op guard
    i = dwh["fact_inventory_daily"].index[dwh["fact_inventory_daily"]["units_sold"] > 0][0]
    dwh["fact_inventory_daily"].loc[i, "units_sold"] -= 1
    dwh["fact_inventory_daily"].loc[i, "closing_stock"] += 1          # keep balance valid so only the cross-check fires
    assert "sales_vs_inventory_units" in _failed(raw, dwh)


def test_detects_date_gap():
    _, raw, dwh = copies()
    dwh["dim_date"] = dwh["dim_date"].drop(index=40).reset_index(drop=True)
    assert "date_completeness" in _failed(raw, dwh)


def test_detects_latent_below_observed():
    _, raw, dwh = copies()
    raw["sim_latent_demand"].loc[raw["inventory_daily"]["units_sold"].idxmax(), "latent_demand_units"] = 0
    assert "latent_ge_observed" in _failed(raw, dwh)


def test_detects_discount_without_promotion():
    _, raw, dwh = copies()
    i = dwh["fact_sales"].index[dwh["fact_sales"]["promotion_key"].isna()][0]
    dwh["fact_sales"].loc[i, "discount_amount"] = 5.0
    assert "discount_has_promotion" in _failed(raw, dwh)


def test_assert_quality_raises_actionable_error():
    _, raw, dwh = copies()
    dwh["fact_sales"].loc[0, "store_key"] = 999
    e = raises(DataQualityError, assert_quality, run_quality_checks(raw, dwh))
    assert "fk_sales_store" in str(e)


def test_sql_validation_detects_corruption_when_constraints_are_absent():
    """Defence in depth: the SQL checks catch bad data even if it bypassed constraints (e.g. staged loads)."""
    from helpers import loaded_sqlite
    db = loaded_sqlite()
    db._h.execute("PRAGMA ignore_check_constraints = ON")
    db._h.execute("UPDATE dwh.fact_inventory_daily SET closing_stock = closing_stock + 1 WHERE rowid = 7")
    failed = {r.name for r in run_db_checks(db) if not r.passed}
    assert "db_stock_balance" in failed
