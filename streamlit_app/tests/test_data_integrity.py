"""Data-integrity checks: manifest, keys, relationships, identities, snapshot, backtest tables."""

from dataclasses import replace
from types import MappingProxyType

import pandas as pd
import pytest

import oracle_values as ov
from dzmi_app.data_loader import count_csv_rows
from dzmi_app.errors import DataIntegrityError
from dzmi_app.integrity import (
    CheckResult, assert_dataset_ok, failed_errors, run_integrity_checks,
)
from dzmi_app.schema import APP_TABLES, EXPORT_ONLY_TABLES

EXPECTED_IDS = [f"IC{i:02d}" for i in range(1, 19)]


@pytest.fixture(scope="module")
def results(dataset):
    return {r.check_id: r for r in run_integrity_checks(dataset)}


def tampered(ds, table, **changes):
    """Copy of the dataset with one table replaced (the original stays untouched)."""
    frame = changes.pop("frame")
    tables = {**ds.tables, table: frame}
    return replace(ds, tables=MappingProxyType(tables))


# ------------------------------------------------------------------ manifest


def test_the_manifest_lists_exactly_the_22_exported_tables(dataset):
    assert set(dataset.manifest["rows"]) == set(APP_TABLES) | set(EXPORT_ONLY_TABLES)
    assert len(dataset.manifest["rows"]) == 22
    assert dataset.manifest["rows"] == ov.MANIFEST_ROWS


@pytest.mark.parametrize("table", sorted(ov.MANIFEST_ROWS))
def test_csv_row_count_equals_manifest(dataset, table):
    assert count_csv_rows(table, dataset.data_dir) == ov.MANIFEST_ROWS[table]


def test_manifest_flags_the_data_as_synthetic(dataset):
    assert dataset.manifest["synthetic"] is True
    assert "synthetic" in dataset.manifest["disclaimer"].lower()


# ------------------------------------------------------------------ every check


def test_all_expected_checks_ran_once(results):
    assert sorted(results) == EXPECTED_IDS


@pytest.mark.parametrize("check_id", EXPECTED_IDS)
def test_each_check_passes_on_the_dataset(results, check_id):
    r = results[check_id]
    assert isinstance(r, CheckResult)
    assert r.passed, f"{r.check_id} {r.name}: {r.detail}"


def test_no_blocking_failures_and_gate_passes(dataset):
    assert failed_errors(run_integrity_checks(dataset)) == []
    assert len(assert_dataset_ok(dataset)) == len(EXPECTED_IDS)


def test_info_checks_never_block(results):
    infos = [r.check_id for r in results.values() if r.severity == "info"]
    assert infos == ["IC09", "IC10", "IC17", "IC18"]


# ------------------------------------------------------------------ dataset facts (audit oracles)


def test_date_range_and_completeness(dataset):
    dd = dataset["dim_date"]
    assert f"{dd['full_date'].min():%Y-%m-%d}" == ov.DATE_MIN
    assert f"{dd['full_date'].max():%Y-%m-%d}" == ov.DATE_MAX
    assert len(dd) == ov.N_DAYS
    months = dd.groupby("year_month_label")["is_complete_month"].first().to_dict()
    assert months == {"2025-02": 1, "2025-03": 1, "2025-04": 1, "2025-05": 0}
    assert dd.groupby("year_month_label").size().to_dict() == {
        "2025-02": 28, "2025-03": 31, "2025-04": 30, "2025-05": 1}


def test_sales_and_inventory_cover_the_same_90_days(dataset):
    dd = set(dataset["dim_date"]["date_key"])
    assert set(dataset["fact_sales"]["date_key"]) == dd
    assert set(dataset["fact_inventory_daily"]["date_key"]) == dd


@pytest.mark.parametrize("key,expected", list(ov.NULLS.items()))
def test_documented_null_counts(dataset, key, expected):
    table, col = key
    assert int(dataset[table][col].isna().sum()) == expected


def test_no_unexpected_nulls_anywhere_else(dataset):
    from dzmi_app.schema import SCHEMAS
    for table in APP_TABLES:
        for col in SCHEMAS[table]:
            if not col.nullable:
                assert not dataset[table][col.name].isna().any(), (table, col.name)


def test_sales_line_facts(dataset):
    s = dataset["fact_sales"]
    assert s["promotion_key"].notna().sum() == ov.PROMO_LINES
    assert s["transaction_id"].nunique() == ov.TRANSACTIONS
    assert (s["gross_profit"] < 0).sum() == ov.NEGATIVE_MARGIN_LINES
    assert s.loc[s["gross_profit"] < 0, "promotion_key"].notna().all()


def test_fact_fingerprint(dataset):
    s = dataset["fact_sales"]
    assert s["net_sales"].sum() == pytest.approx(ov.SUM_NET_SALES, abs=0.01)
    assert int(s["quantity"].sum()) == ov.SUM_QUANTITY
    assert int(dataset["fact_inventory_daily"]["units_sold"].sum()) == ov.SUM_QUANTITY


def test_inventory_coverage(dataset):
    inv = dataset["fact_inventory_daily"]
    pairs = inv.groupby(["store_key", "product_key"]).size()
    assert len(pairs) == ov.TRACKED_PAIRS < ov.INVENTORY_PAIRS_POSSIBLE
    assert (pairs == ov.N_DAYS).all()                 # every pair has all 90 days
    assert int(inv["stockout_flag"].sum()) == ov.STOCKOUT_SKU_DAYS
    assert len(inv) == ov.TRACKED_PAIRS * ov.N_DAYS


def test_risk_snapshot_facts(dataset):
    vr = dataset["v_inventory_risk"]
    assert set(vr["snapshot_date_key"]) == {ov.SNAPSHOT_DATE_KEY}
    assert vr["risk_tier"].value_counts().to_dict() == ov.TIER_COUNTS
    assert len(vr) == ov.TRACKED_PAIRS


def test_forecast_tables_are_empty_and_no_forecast_is_stored(dataset):
    assert len(dataset["fact_demand_forecast"]) == 0
    assert len(dataset["model_run"]) == 0


# ------------------------------------------------------------------ the checks must be able to fail


def test_gate_raises_when_a_sales_identity_is_broken(dataset):
    bad = dataset["fact_sales"].copy()
    bad.loc[0, "net_sales"] += 1.0
    ds = tampered(dataset, "fact_sales", frame=bad)
    ids = {r.check_id for r in failed_errors(run_integrity_checks(ds))}
    assert "IC07" in ids
    with pytest.raises(DataIntegrityError, match="IC07"):
        assert_dataset_ok(ds)


def test_orphan_foreign_key_is_detected(dataset):
    bad = dataset["fact_sales"].copy()
    bad.loc[0, "store_key"] = 99
    ds = tampered(dataset, "fact_sales", frame=bad)
    ids = {r.check_id for r in failed_errors(run_integrity_checks(ds))}
    assert "IC06" in ids


def test_duplicate_key_is_detected(dataset):
    bad = pd.concat([dataset["dim_store"], dataset["dim_store"].iloc[[0]]], ignore_index=True)
    ds = tampered(dataset, "dim_store", frame=bad)
    assert "IC05" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_broken_inventory_identity_is_detected(dataset):
    bad = dataset["fact_inventory_daily"].copy()
    bad.loc[0, "closing_stock"] += 1
    ds = tampered(dataset, "fact_inventory_daily", frame=bad)
    assert "IC11" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_inventory_that_disagrees_with_sales_is_detected(dataset):
    bad = dataset["fact_inventory_daily"].copy()
    i = bad.index[bad["units_sold"] > 0][0]
    bad.loc[i, "units_sold"] += 1
    bad.loc[i, "closing_stock"] -= 1                  # keep the identity intact
    ds = tampered(dataset, "fact_inventory_daily", frame=bad)
    ids = {r.check_id for r in failed_errors(run_integrity_checks(ds))}
    assert "IC12" in ids and "IC11" not in ids


def test_wrong_risk_tier_is_detected(dataset):
    bad = dataset["v_inventory_risk"].copy()
    bad.loc[bad["risk_tier"] == "Low", "risk_tier"] = "Healthy"
    ds = tampered(dataset, "v_inventory_risk", frame=bad)
    assert "IC14" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_accuracy_view_that_disagrees_with_the_sums_is_detected(dataset):
    bad = dataset["v_forecast_accuracy"].copy()
    bad.loc[0, "sum_abs_error"] *= 1.01
    ds = tampered(dataset, "v_forecast_accuracy", frame=bad)
    assert "IC16" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_manifest_mismatch_is_detected(dataset):
    rows = {**dataset.manifest["rows"], "dim_store": 6}
    ds = replace(dataset, manifest=MappingProxyType({**dataset.manifest, "rows": rows}))
    assert "IC02" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_non_synthetic_manifest_is_rejected(dataset):
    ds = replace(dataset, manifest=MappingProxyType({**dataset.manifest, "synthetic": False}))
    assert "IC01" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}


def test_broken_month_flag_is_detected(dataset):
    bad = dataset["dim_date"].copy()
    bad.loc[bad["year_month"] == 202505, "is_complete_month"] = 1
    ds = tampered(dataset, "dim_date", frame=bad)
    assert "IC04" in {r.check_id for r in failed_errors(run_integrity_checks(ds))}
