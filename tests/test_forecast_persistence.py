"""Milestone 3: additive SQL, run persistence, database-level leakage checks and non-destructive behaviour."""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from dzmi.errors import DatabaseError, DZMIError
from dzmi.features.panel import load_panel
from dzmi.forecasting import metrics as M
from dzmi.forecasting.backtest import run_backtest, single_forecast
from dzmi.forecasting.registry import (check_run_id, default_backtest_id, default_forecast_id, save_backtest,
                                       save_forecast)
from dzmi.forecasting.views import FORECAST_VIEWS
from dzmi.ingestion.db import Database, parse_check_file, split_statements
from dzmi.ingestion.db_validation import run_db_checks
from dzmi.paths import sql_dir
from helpers import forecast_settings, loaded_sqlite, raises, shared_smoke_panel

SCHEMA_SQL = (sql_dir() / "schema" / "005_forecast_metrics.sql").read_text(encoding="utf-8")
CHECK_SQL = (sql_dir() / "validation" / "003_forecast_checks.sql").read_text(encoding="utf-8")
quiet = lambda msg: None
BASE_TABLES = ["dim_date", "dim_store", "dim_product", "dim_customer", "dim_promotion", "fact_sales",
               "fact_inventory_daily", "fact_promotions"]


def failed(db):
    return {r.name: r.violations for r in run_db_checks(db) if not r.passed}


def small_run(db, models="naive,seasonal_naive,croston_sba", origins=2):
    s = forecast_settings(models=models, n_origins=origins)
    return s, run_backtest(shared_smoke_panel(), s, progress=quiet)


# ------------------------------------------------------------------------------------------ SQL is additive and portable
def test_forecast_sql_is_additive_and_portable():
    stmts = split_statements(SCHEMA_SQL)
    assert [s.split("(")[0].split(" AS")[0].strip() for s in stmts] == [
        "CREATE TABLE IF NOT EXISTS dwh.backtest_run", "CREATE TABLE IF NOT EXISTS dwh.forecast_backtest_metric",
        "DROP VIEW IF EXISTS dwh.v_forecast_accuracy_by_horizon", "DROP VIEW IF EXISTS dwh.v_forecast_accuracy",
        "CREATE VIEW dwh.v_forecast_accuracy", "CREATE VIEW dwh.v_forecast_accuracy_by_horizon"]
    code = "\n".join(l for l in (SCHEMA_SQL + "\n" + CHECK_SQL).splitlines() if not l.strip().startswith("--"))
    assert not re.search(r"\b(DROP\s+TABLE|DELETE\s+FROM|TRUNCATE|ALTER\s+TABLE|UPDATE\s+dwh|INSERT\s+INTO)\b", code, re.I)
    for pattern in (r"\bFULL\s+OUTER\b", r"CREATE\s+OR\s+REPLACE", r"::", r"\bSQRT\s*\(", r"\bjulianday\b", r"\bstrftime\b"):
        assert not re.search(pattern, code, re.I), pattern
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if "/ NULLIF(" in ln:
            assert "1.0 *" in ln or "1.0 *" in lines[i - 1], ln.strip()
    for s in stmts:
        if s.startswith("CREATE VIEW"):
            assert not re.search(r"\bdwh\.v_", s.split(" AS", 1)[1]), "a view must be built on tables only"
    assert set(FORECAST_VIEWS) == set(re.findall(r"CREATE VIEW dwh\.(\w+)", SCHEMA_SQL))


def test_forecast_checks_parse_and_pass_on_an_empty_and_a_populated_database():
    checks = parse_check_file(CHECK_SQL)
    assert len(checks) == 9 and len({c["name"] for c in checks}) == 9 and all(c["name"].startswith("fc_") for c in checks)
    assert {c["severity"] for c in checks} == {"error", "warning"}
    db = loaded_sqlite()
    assert failed(db) == {}
    s, res = small_run(db)
    save_backtest(db, "bt_a", res)
    panel = shared_smoke_panel()
    pred, *_ = single_forecast(panel, s, "seasonal_naive", 60)
    save_forecast(db, "fc_a", "seasonal_naive", panel, 60, pred, s)
    assert failed(db) == {}
    db.close()


def test_upgrading_an_existing_database_keeps_every_row():
    db = loaded_sqlite()
    before = {t: db.row_count("dwh", t) for t in BASE_TABLES}
    before["ops_sales"] = db.row_count("ops", "sales_transaction_line")
    for stmt in ("DROP VIEW dwh.v_forecast_accuracy_by_horizon", "DROP VIEW dwh.v_forecast_accuracy",
                 "DROP TABLE dwh.forecast_backtest_metric", "DROP TABLE dwh.backtest_run"):
        db._h.execute(stmt)                                           # what a Milestone 2 database looks like
    raises(DatabaseError, run_db_checks, db)
    try:
        run_db_checks(db)
    except DatabaseError as exc:
        assert "init-db" in str(exc) and "fc_" in str(exc)
    db.apply_schema()
    db.apply_schema()                                                 # idempotent
    after = {t: db.row_count("dwh", t) for t in BASE_TABLES}
    after["ops_sales"] = db.row_count("ops", "sales_transaction_line")
    assert before == after and failed(db) == {}
    assert db.row_count("dwh", "backtest_run") == 0
    db.close()


# ------------------------------------------------------------------------------------------ backtest persistence
def test_stored_metrics_reproduce_the_python_summary_through_sql_views():
    db = loaded_sqlite()
    s, res = small_run(db)
    n = save_backtest(db, "bt_v", res, notes="unit test")
    assert n == len(res.metrics) == db.row_count("dwh", "forecast_backtest_metric")
    head = db.query_df("SELECT * FROM dwh.backtest_run").iloc[0]
    assert head["backtest_run_id"] == "bt_v" and head["models"] == "naive,seasonal_naive,croston_sba"
    assert head["n_series"] == 419 and head["n_origins"] == 2 and head["horizon_days"] == 14 and head["notes"] == "unit test"
    py = M.summarise(res.metrics)
    sql = db.query_df("SELECT * FROM dwh.v_forecast_accuracy")
    m = py.merge(sql, on=["model_name", "evaluation_basis", "level"], suffixes=("_py", "_sql"))
    assert len(m) == len(py) == len(sql) == 3 * 3        # 3 models x (observed day, observed week, uncensored day)
    for col in ("mae", "mse", "wape", "bias", "skill_vs_seasonal_naive"):
        sql_col = "skill_mae_vs_seasonal_naive" if col.startswith("skill") else f"{col}_sql"
        a, b = m[f"{col}_py"] if f"{col}_py" in m else m[col], m[sql_col]
        assert np.allclose(a.astype(float), b.astype(float), rtol=1e-5, atol=1e-6, equal_nan=True), col
    assert (m["n_obs_py"] == m["n_obs_sql"]).all() and (m["n_origins_py"] == m["n_origins_sql"]).all()
    by_h = db.query_df("SELECT * FROM dwh.v_forecast_accuracy_by_horizon")
    assert len(by_h) == 3 * (14 + 14 + 2)                              # day steps x2 bases, plus 2 forecast weeks
    sn = by_h[by_h.model_name == "seasonal_naive"]["skill_mae_vs_seasonal_naive"].astype(float)
    assert np.allclose(sn, 0.0)                                       # the baseline has zero skill against itself
    db.close()


def test_view_mae_equals_an_independent_pandas_calculation_for_the_naive_model():
    db = loaded_sqlite()
    s = forecast_settings(models="naive,seasonal_naive", n_origins=2, censor_policy="keep")   # keep: plain last value
    res = run_backtest(shared_smoke_panel(), s, progress=quiet)
    save_backtest(db, "bt_i", res)
    panel = shared_smoke_panel()
    errs = []
    for origin in res.origins:
        o = panel.index_of(origin)
        last = panel.units[:, o]
        actual = panel.units[:, o + 1:o + 15]
        errs.append(np.abs(last[:, None] - actual))
    expected = np.concatenate([e.ravel() for e in errs]).mean()
    got = db.scalar("SELECT mae FROM dwh.v_forecast_accuracy WHERE model_name='naive' AND evaluation_basis='observed' "
                    "AND level='series_day'")
    assert abs(got - expected) < 1e-4
    db.close()


def test_existing_run_ids_are_never_overwritten_and_ids_are_validated():
    db = loaded_sqlite()
    s, res = small_run(db, models="naive,seasonal_naive", origins=1)
    save_backtest(db, "bt_dup", res)
    n_head, n_rows = db.row_count("dwh", "backtest_run"), db.row_count("dwh", "forecast_backtest_metric")
    raises(DZMIError, save_backtest, db, "bt_dup", res)
    assert (db.row_count("dwh", "backtest_run"), db.row_count("dwh", "forecast_backtest_metric")) == (n_head, n_rows)
    for bad in ("", "x" * 41, "a b", "x'; DROP TABLE dwh.fact_sales; --", "../etc"):
        raises(DZMIError, check_run_id, bad)
    assert re.fullmatch(r"bt_[A-Za-z0-9]+_\d{8}_\d{6}", default_backtest_id("smoke"))
    assert len(default_forecast_id("moving_average", pd.Timestamp("2025-04-17"))) <= 40
    db.close()


def test_running_forecasts_never_changes_the_loaded_data():
    db = loaded_sqlite()
    snapshot = lambda: {t: (db.row_count("dwh", t), db.scalar(f"SELECT COUNT(*) FROM dwh.{t}")) for t in BASE_TABLES}
    sums = lambda: (db.scalar("SELECT SUM(net_sales) FROM dwh.fact_sales"), db.scalar("SELECT SUM(units_sold) FROM dwh.fact_inventory_daily"))
    before, tot = snapshot(), sums()
    s, res = small_run(db)
    save_backtest(db, "bt_n", res)
    panel = shared_smoke_panel()
    pred, *_ = single_forecast(panel, s, "croston_sba", 70)
    save_forecast(db, "fc_n", "croston_sba", panel, 70, pred, s)
    assert snapshot() == before and sums() == tot
    db.close()


# ------------------------------------------------------------------------------------------ forecast persistence
def test_saved_forecasts_have_the_exact_grid_dates_and_run_metadata():
    db = loaded_sqlite()
    panel = shared_smoke_panel()
    s = forecast_settings()
    o = panel.index_of("2025-04-10")
    pred, actual, cens, days = single_forecast(panel, s, "ses_weekday", o)
    n = save_forecast(db, "fc_g", "ses_weekday", panel, o, pred, s, notes="grid test")
    assert n == 419 * 14 == db.row_count("dwh", "fact_demand_forecast")
    run = db.query_df("SELECT * FROM dwh.model_run").iloc[0]
    assert run["forecast_run_id"] == "fc_g" and run["train_end_date"] == "2025-04-10" and run["horizon_days"] == 14
    assert run["model_name"] == "ses_weekday" and "Censoring policy: exclude" in run["feature_definition"]
    f = db.query_df("SELECT * FROM dwh.fact_demand_forecast")
    assert sorted(f["horizon_step"].unique()) == list(range(1, 15)) and f["date_key"].min() == 20250411 and f["date_key"].max() == 20250424
    assert (f["predicted_units"] >= 0).all() and f["lower_bound"].isna().all() and f["upper_bound"].isna().all()
    key = f.merge(pd.DataFrame({"store_key": np.repeat(panel.series.store_key, 14), "product_key": np.repeat(panel.series.product_key, 14),
                                "step": np.tile(np.arange(1, 15), 419), "expected": np.round(pred.reshape(-1), 3)}),
                  left_on=["store_key", "product_key", "horizon_step"], right_on=["store_key", "product_key", "step"])
    assert len(key) == len(f) and np.allclose(key["predicted_units"].astype(float), key["expected"])
    raises(DZMIError, save_forecast, db, "fc_g", "ses_weekday", panel, o, pred, s)
    assert failed(db) == {}
    db.close()


def test_foreign_keys_and_checks_reject_bad_forecast_rows():
    db = loaded_sqlite()
    db._h.execute("INSERT INTO dwh.model_run (forecast_run_id, model_name, model_version, trained_at, train_end_date, horizon_days) "
                  "VALUES ('r1','naive','v','2025-04-01 00:00:00','2025-04-01',14)")
    ins = "INSERT INTO dwh.fact_demand_forecast (forecast_run_id, store_key, product_key, date_key, horizon_step, predicted_units) VALUES "
    raises(sqlite3.IntegrityError, db._h.execute, ins + "('r1',1,1,19990101,1,1.0)")      # date not in dim_date
    raises(sqlite3.IntegrityError, db._h.execute, ins + "('nope',1,1,20250402,1,1.0)")    # unknown run
    raises(sqlite3.IntegrityError, db._h.execute, ins + "('r1',1,1,20250402,1,-5.0)")     # negative forecast
    db.close()


def test_database_checks_catch_leaky_or_incomplete_forecasts():
    db = loaded_sqlite()
    panel, s = shared_smoke_panel(), forecast_settings()
    o = panel.index_of("2025-04-10")
    pred, *_ = single_forecast(panel, s, "naive", o)
    save_forecast(db, "fc_ok", "naive", panel, o, pred, s)
    assert failed(db) == {}
    ins = "INSERT INTO dwh.fact_demand_forecast (forecast_run_id, store_key, product_key, date_key, horizon_step, predicted_units) VALUES "
    db._h.execute(ins + "('fc_ok',1,1,20250409,1,3.0)")                    # a target day BEFORE the train end: leakage
    bad = failed(db)
    assert bad["db_fc_forecast_after_train_end"] == 1 and bad["db_fc_horizon_step_matches_dates"] >= 1
    db._h.execute("DELETE FROM dwh.fact_demand_forecast WHERE date_key = 20250409")
    db._h.execute(ins + "('fc_ok',1,1,20250410,1,3.0)")                    # boundary: the origin day itself is not a forecast day
    assert failed(db)["db_fc_forecast_after_train_end"] == 1
    db._h.execute("DELETE FROM dwh.fact_demand_forecast WHERE date_key = 20250410")
    db._h.execute("DELETE FROM dwh.fact_demand_forecast WHERE store_key = 1 AND product_key = 1 AND horizon_step = 14")
    assert "db_fc_forecast_grid_complete" in failed(db)                    # a series lost a day
    db._h.execute("UPDATE dwh.model_run SET horizon_days = 7 WHERE forecast_run_id = 'fc_ok'")
    assert failed(db)["db_fc_horizon_within_run"] > 0
    db.close()


def test_database_checks_catch_inconsistent_backtest_metrics():
    db = loaded_sqlite()
    s, res = small_run(db, models="naive,seasonal_naive", origins=1)
    save_backtest(db, "bt_c", res)
    assert failed(db) == {}
    db._h.execute("UPDATE dwh.forecast_backtest_metric SET n_obs = n_obs + 1 WHERE model_name = 'naive' AND "
                  "evaluation_basis = 'observed' AND level = 'series_day' AND horizon_bucket = 1")
    assert failed(db).get("db_fc_backtest_same_evaluation_set") == 1        # models scored on different observations
    db._h.execute("UPDATE dwh.forecast_backtest_metric SET sum_abs_error = 0 WHERE model_name = 'naive' AND horizon_bucket = 2 "
                  "AND level = 'series_day' AND evaluation_basis = 'observed'")
    assert failed(db).get("db_fc_backtest_sums_consistent", 0) >= 1         # |sum error| cannot exceed sum |error|
    db._h.execute("INSERT INTO dwh.backtest_run VALUES ('bt_empty','2025-05-01 00:00:00','smoke',14,1,7,42,5,'exclude',1,'naive',NULL,NULL)")
    assert "db_fc_backtest_run_has_metrics" in failed(db)                   # an interrupted run is flagged
    raises(sqlite3.IntegrityError, db._h.execute,
           "INSERT INTO dwh.forecast_backtest_metric VALUES ('bt_c','naive','observed','series_day','2025-04-01',1,1,1,1,-1,1)")
    db.close()


def test_a_backtest_without_the_baseline_is_a_warning_not_an_error():
    db = loaded_sqlite()
    s, res = small_run(db, models="naive,croston_sba", origins=1)
    save_backtest(db, "bt_nb", res)
    r = {x.name: x for x in run_db_checks(db)}["db_fc_backtest_has_baseline"]
    assert not r.passed and r.severity == "warning"
    assert db.query_df("SELECT skill_mae_vs_seasonal_naive AS s FROM dwh.v_forecast_accuracy")["s"].isna().all()
    db.close()
