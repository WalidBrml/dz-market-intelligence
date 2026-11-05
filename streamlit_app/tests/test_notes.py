"""Data & Method Notes: content comes from the loaded dataset; the report's own text is kept verbatim."""

from dataclasses import replace
from types import MappingProxyType

import pandas as pd
import pytest

from dzmi_app import notes
from dzmi_app.constants import BANNER_TEXT
from dzmi_app.integrity import run_integrity_checks
from dzmi_app.schema import APP_TABLES, EXPORT_ONLY_TABLES

ORIGINAL = {
    "What this report is": "A portfolio demonstration built on SYNTHETIC data from the DZ Market Intelligence simulator. "
        "No store, customer, price, sale or forecast score here describes the real Algerian market.",
    "How horizon_bucket is defined": "At the series x day level it is the forecast day step (1-14 days after the origin). "
        "At the store x category x week level it is the forecast week (1-2). The two levels have different scales and "
        "are never shown on one axis.",
    "How accuracy is computed": "Errors are stored as additive sums per model, basis, level, origin and horizon. MAE, RMSE, "
        "WAPE and bias are always recomputed from the sums (sum of errors / sum of observations). They are never "
        "averaged or added across rows.",
}
LIMITS = [
    "Year-over-year is blank: the data covers 2025-02-01 to 2025-05-01 only. May holds a single day, so month-over-month "
    "is shown only for complete months.",
    "Inventory risk is a snapshot of the latest loaded day and ignores the date slicer.",
    "No stored forecast exists (fact_demand_forecast is empty): this report shows backtest accuracy, not "
    "forecast-versus-actual lines.",
    "Observed sales are truncated on stockout days; the no-stockout basis shows how much that matters.",
    "Four rolling origins on a small sample cannot rank methods in general.",
]


# ------------------------------------------------------------------ the report's own text


def test_report_text_is_verbatim():
    assert dict(notes.WHAT_THIS_REPORT_IS) == ORIGINAL
    assert [h for h, _ in notes.WHAT_THIS_REPORT_IS] == list(ORIGINAL)
    assert notes.KNOWN_LIMITS == LIMITS


def test_disclaimer_is_the_synthetic_data_banner():
    assert BANNER_TEXT == "SYNTHETIC DATA - demonstration only. Nothing here describes real Algerian retail demand."
    assert "SYNTHETIC" in notes.WHAT_THIS_REPORT_IS[0][1]


# ------------------------------------------------------------------ manifest-driven values


def test_file_table_follows_the_manifest(dataset):
    t = notes.dataset_files(dataset)
    assert len(t) == 22 and dict(zip(t["table"], t["rows"])) == dict(dataset.manifest["rows"])
    assert t.loc[t["table"] == "fact_sales", "rows"].iloc[0] == 48_216
    assert set(t["table"]) == set(APP_TABLES) | set(EXPORT_ONLY_TABLES)


def test_empty_tables_are_flagged_and_everything_else_is_not(dataset):
    t = notes.dataset_files(dataset).set_index("table")
    assert t.loc["fact_demand_forecast", "status"] == "EMPTY (0 rows)" and t.loc["model_run", "status"] == "EMPTY (0 rows)"
    assert (t[t["rows"] > 0]["status"] == "").all()
    assert (t["rows"] == 0).sum() == 2


def test_roles_separate_loaded_tables_from_export_only_tables(dataset):
    t = notes.dataset_files(dataset).set_index("table")
    assert (t.loc[list(APP_TABLES), "role"] == "Loaded by the dashboard").all()
    assert (t.loc[list(EXPORT_ONLY_TABLES), "role"] == "Exported, not loaded by the Power BI model").all()


def test_the_file_table_reads_the_manifest_not_constants(dataset):
    rows = {**dataset.manifest["rows"], "dim_store": 99}
    t = notes.dataset_files(replace(dataset, manifest=MappingProxyType({**dataset.manifest, "rows": rows})))
    assert t.loc[t["table"] == "dim_store", "rows"].iloc[0] == 99


def test_coverage_figures_come_from_the_tables(dataset):
    c = notes.coverage(dataset)
    assert (str(c["first_date"]), str(c["last_date"]), c["days"]) == ("2025-02-01", "2025-05-01", 90)
    assert (c["stores"], c["products"], c["categories"], c["customers"]) == (5, 100, 14, 600)
    assert (c["sales_lines"], c["transactions"]) == (48_216, 5_776)
    assert (c["inventory_sku_days"], c["tracked_skus"], str(c["snapshot_date"])) == (37_710, 419, "2025-05-01")
    assert (c["forecast_rows"], c["model_run_rows"]) == (0, 0)
    assert (c["backtest_run_id"], c["origins"], c["horizon_days"], c["origin_step_days"], c["min_train_days"]) == \
        ("smoke_m3_20261025_01", 4, 14, 7, 42)
    assert (c["n_series"], c["censor_policy"], c["synthetic"]) == (419, "exclude", True)
    assert c["models"] == ["naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba", "gbm"]


def test_backtest_text_states_the_facts_of_the_run(dataset):
    text = " ".join(notes.backtest_vs_forecast_text(notes.coverage(dataset)))
    assert "4 past forecast origins" in text and "14 days after each origin" in text and "7 days apart" in text
    assert "fact_demand_forecast has 0 rows" in text and "model_run has 0 rows" in text
    assert "no future forecast" in text and "'exclude'" in text


# ------------------------------------------------------------------ reconciliation and integrity


def test_reconciliation_covers_the_18_model_basis_level_combinations_and_all_agree(dataset):
    r = notes.reconciliation(dataset)
    assert len(r) == 18 and list(r["status"].unique()) == ["OK"]
    assert r["difference"].abs().max() < 1e-6
    assert set(r["model_label"]) == {"Naive", "Seasonal naive (benchmark)", "Moving average",
                                     "Exp. smoothing + weekday", "Croston-SBA", "Gradient boosting"}
    assert not ((r["basis_label"] == "Observed, no-stockout days") & (r["level_label"] == "Store x category x week")).any()


def test_reconciliation_flags_a_view_that_disagrees(dataset):
    bad = dataset["v_forecast_accuracy"].copy()
    bad.loc[0, "mae"] += 0.001
    tables = {**dataset.tables, "v_forecast_accuracy": bad}
    r = notes.reconciliation(replace(dataset, tables=MappingProxyType(tables)))
    assert (r["status"] == "CHECK").sum() == 1 and (r["status"] == "OK").sum() == 17


def test_integrity_table_lists_the_18_checks(dataset):
    t = notes.integrity_table(run_integrity_checks(dataset))
    assert len(t) == 18 and list(t["result"].unique()) == ["passed"]
    assert list(t["check"]) == [f"IC{i:02d}" for i in range(1, 19)]
    assert (t["kind"] == "information").sum() == 4 and (t["kind"] == "blocking").sum() == 14


# ------------------------------------------------------------------ static content stays verifiable


def test_relationships_are_the_29_of_the_semantic_model():
    rel = notes.relationships_table()
    assert len(rel) == 29 and len(set(map(tuple, rel.values))) == 29
    known = set(APP_TABLES) | {"dim_model", "dim_eval_basis", "dim_eval_level", "dim_horizon", "dim_origin"}
    for src, dst in notes.RELATIONSHIPS:
        assert src.split(".")[0] in known and dst.split(".")[0] in known
    assert ("v_inventory_risk.store_key", "dim_store.store_key") in notes.RELATIONSHIPS
    assert not any(src.startswith("v_inventory_risk") and dst.startswith("dim_date") for src, dst in notes.RELATIONSHIPS)
    assert not any("fact_inventory_daily" in src and "fact_sales" in dst for src, dst in notes.RELATIONSHIPS)


def test_definitions_cover_the_dashboard_measures():
    names = " ".join(n for n, _ in notes.DEFINITIONS)
    for needle in ("Gross Margin", "Average Basket", "MoM", "Stockout Rate", "Closing Stock", "Days of Cover", "Skill"):
        assert needle in names
    text = " ".join(t for _, t in notes.DEFINITIONS)
    assert "never summed across dates" in text and "additive sums" in text


def test_deviation_register_names_each_page_once_in_report_order():
    pages = [p for p, _ in notes.DEVIATIONS]
    assert pages == ["All pages", "Executive Overview", "Sales Intelligence", "Inventory Intelligence",
                     "Forecasting Performance", "Forecast Accuracy by Horizon", "Data & Method Notes"]


def test_no_invented_claims_about_sources_or_processes(dataset):
    import re
    everything = " ".join([
        *(t for _, t in notes.WHAT_THIS_REPORT_IS), *notes.KNOWN_LIMITS, *(t for _, t in notes.DEFINITIONS),
        *(t for _, t in notes.DEVIATIONS), *notes.backtest_vs_forecast_text(notes.coverage(dataset))])
    words = set(re.findall(r"[a-z]+", everything.lower()))
    for banned in ("refresh", "schedule", "etl", "pipeline", "warehouse", "erp", "nightly", "sla", "lineage",
                   "kafka", "sap"):
        assert banned not in words, banned


def test_finite_helper():
    assert notes.finite(1.0) and notes.finite(0) and not notes.finite(None) and not notes.finite(float("nan"))
