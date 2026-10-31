"""Milestone 3: metrics, rolling-origin splitting and the backtest engine, checked against independent calculations."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dzmi.errors import ConfigError, DZMIError
from dzmi.features.panel import Panel
from dzmi.forecasting import metrics as M
from dzmi.forecasting.backtest import group_matrix, make_origins, run_backtest
from dzmi.forecasting.config import load_forecast_settings
from dzmi.forecasting.diagnostics import load_latent
from dzmi.forecasting.report import DISCLAIMER, format_summary, markdown_report, write_report
from helpers import forecast_settings, shared_smoke_db, shared_smoke_panel, smoke

BASELINES = "naive,seasonal_naive,moving_average,ses_weekday,croston_sba"
quiet = lambda msg: None


# ------------------------------------------------------------------------------------------ metrics
def test_error_sums_and_derived_metrics_known_answers():
    actual, pred = np.array([2.0, 0.0, 4.0]), np.array([3.0, 1.0, 1.0])       # errors +1, +1, -3
    sums = M.error_sums(actual, pred, np.ones(3, dtype=bool))
    assert sums == {"n_obs": 3, "sum_actual": 6.0, "sum_pred": 5.0, "sum_abs_error": 5.0, "sum_sq_error": 11.0}
    d = M.add_derived(pd.DataFrame([sums])).iloc[0]
    assert abs(d["mae"] - 5 / 3) < 1e-12 and abs(d["mse"] - 11 / 3) < 1e-12 and abs(d["rmse"] - math.sqrt(11 / 3)) < 1e-12
    assert abs(d["wape"] - 5 / 6) < 1e-12 and abs(d["bias"] - (-1 / 6)) < 1e-12
    masked = M.error_sums(actual, pred, np.array([True, False, True]))          # the zero-actual day is excluded
    assert masked["n_obs"] == 2 and masked["sum_abs_error"] == 4.0


def test_ratios_are_nan_not_infinite_when_the_denominator_is_zero():
    z = M.add_derived(pd.DataFrame([M.error_sums(np.zeros(3), np.array([1.0, 0, 2]), np.ones(3, dtype=bool))])).iloc[0]
    assert math.isnan(z["wape"]) and math.isnan(z["bias"]) and z["mae"] == 1.0
    empty = M.add_derived(pd.DataFrame([M.error_sums(np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool))])).iloc[0]
    assert math.isnan(empty["mae"]) and math.isnan(empty["rmse"])


def test_summarise_pools_sums_not_ratios_and_computes_skill():
    rows = []
    for model, errs in (("seasonal_naive", [2.0, 2.0]), ("m1", [1.0, 0.5])):
        for origin, e in zip(("2025-01-01", "2025-01-08"), errs):
            rows.append({"model_name": model, "evaluation_basis": "observed", "level": "series_day",
                         "origin_date": pd.Timestamp(origin), "horizon_bucket": 1, "n_obs": 10 if origin.endswith("01") else 30,
                         "sum_actual": 100.0, "sum_pred": 100.0, "sum_abs_error": e * (10 if origin.endswith("01") else 30),
                         "sum_sq_error": 1.0})
    s = M.summarise(pd.DataFrame(rows)).set_index("model_name")
    assert abs(s.loc["m1", "mae"] - (10 * 1.0 + 30 * 0.5) / 40) < 1e-12                 # pooled, not the mean of 1.0 and 0.5
    assert abs(s.loc["seasonal_naive", "mae"] - 2.0) < 1e-12 and s.loc["seasonal_naive", "skill_vs_seasonal_naive"] == 0
    assert abs(s.loc["m1", "skill_vs_seasonal_naive"] - (1 - 0.625 / 2.0)) < 1e-12 and s.loc["m1", "n_origins"] == 2


def test_weekly_cells_known_answer_and_incomplete_cells_are_skipped():
    groups = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)             # 4 series in 2 store x category cells
    actual = np.arange(1, 4 * 14 + 1, dtype=float).reshape(4, 14)
    pred = actual + 1.0                                                      # every series-day over-forecast by 1
    rows = M.weekly_rows({"model_name": "m", "evaluation_basis": "observed", "origin_date": pd.Timestamp("2025-01-01")},
                         actual, pred, groups)
    assert [r["horizon_bucket"] for r in rows] == [1, 2] and all(r["n_obs"] == 2 for r in rows)
    assert rows[0]["sum_abs_error"] == 28.0           # 2 cells x (2 series x 7 days x error 1) = 28
    assert rows[0]["sum_actual"] == actual[:, :7].sum()
    actual[0, 3] = np.nan                                                    # one missing day invalidates cell 1 in week 1 only
    rows = M.weekly_rows({"model_name": "m", "evaluation_basis": "observed", "origin_date": pd.Timestamp("2025-01-01")},
                         actual, pred, groups)
    assert rows[0]["n_obs"] == 1 and rows[1]["n_obs"] == 2


# ------------------------------------------------------------------------------------------ origins
def test_make_origins_spacing_bounds_and_errors():
    assert make_origins(100, 14, 42, 4, 7) == [64, 71, 78, 85]
    assert make_origins(100, 14, 80, 4, 7) == [85]                           # fewer origins fit: no invented ones
    raises_cfg = False
    try:
        make_origins(50, 14, 42, 4, 7)
    except DZMIError as exc:
        raises_cfg = "Not enough history" in str(exc)
    assert raises_cfg


def test_every_origin_trains_on_the_past_and_targets_stay_inside_the_data():
    n, h = 120, 14
    for o in make_origins(n, h, 56, 6, 10):
        train, targets = np.arange(0, o + 1), np.arange(o + 1, o + h + 1)
        assert train.max() < targets.min() and targets.max() <= n - 1 and len(train) >= 56
        assert set(train).isdisjoint(targets)


# ------------------------------------------------------------------------------------------ backtest vs independent pandas
def _raw_units():
    """Independent (series x date) matrix straight from ops.inventory_daily, keyed by (store_id, product_id)."""
    cfg, raw, dwh = smoke()
    inv = raw["inventory_daily"]
    wide = inv.pivot_table(index=["store_id", "product_id"], columns="inventory_date", values="units_sold")
    stockout = inv.pivot_table(index=["store_id", "product_id"], columns="inventory_date", values="stockout_flag").astype(bool)
    return wide, stockout


def _series_day_sums(result, model, basis="observed"):
    m = result.metrics
    sub = m[(m.model_name == model) & (m.evaluation_basis == basis) & (m.level == "series_day")]
    return sub[M.SUM_COLUMNS].sum()


def test_baselines_match_independent_pandas_forecasts_exactly():
    """naive / seasonal naive / moving average recomputed with plain pandas from the raw tables (policy keep)."""
    wide, _ = _raw_units()
    panel = shared_smoke_panel()
    s = forecast_settings(models="naive,seasonal_naive,moving_average", censor_policy="keep", n_origins=3)
    res = run_backtest(panel, s, progress=quiet)
    h = s.horizon_days
    expected = {m: dict(n=0, a=0.0, p=0.0, abs=0.0, sq=0.0) for m in s.models}
    for origin in res.origins:
        o = wide.columns.get_loc(origin)
        for k in range(1, h + 1):
            actual = wide.iloc[:, o + k].to_numpy(dtype=float)
            same_wd = wide.iloc[:, o + k - 7 * math.ceil(k / 7)].to_numpy(dtype=float)
            preds = {"naive": wide.iloc[:, o].to_numpy(dtype=float), "seasonal_naive": same_wd,
                     "moving_average": wide.iloc[:, o - 27:o + 1].mean(axis=1).to_numpy(dtype=float)}
            for m, f in preds.items():
                e = f - actual
                d = expected[m]
                d["n"] += len(e); d["a"] += actual.sum(); d["p"] += f.sum(); d["abs"] += np.abs(e).sum(); d["sq"] += (e ** 2).sum()
    for m, d in expected.items():
        got = _series_day_sums(res, m)
        assert got["n_obs"] == d["n"]
        for col, key in (("sum_actual", "a"), ("sum_pred", "p"), ("sum_abs_error", "abs"), ("sum_sq_error", "sq")):
            assert abs(got[col] - d[key]) < 1e-6, (m, col, got[col], d[key])


def test_observed_and_uncensored_bases_differ_by_exactly_the_stockout_days():
    wide, stockout = _raw_units()
    s = forecast_settings(models="naive,seasonal_naive", n_origins=3)
    res = run_backtest(shared_smoke_panel(), s, progress=quiet)
    n_stockout_targets = 0
    for origin in res.origins:
        o = wide.columns.get_loc(origin)
        n_stockout_targets += int(stockout.iloc[:, o + 1:o + 15].to_numpy().sum())
    for m in s.models:
        full, unc = _series_day_sums(res, m, "observed"), _series_day_sums(res, m, "observed_uncensored")
        assert full["n_obs"] - unc["n_obs"] == n_stockout_targets > 0
    assert set(res.metrics["evaluation_basis"]) == {"observed", "observed_uncensored"}      # no latent unless requested


def test_all_models_are_scored_on_the_same_observations():
    s = forecast_settings(models=BASELINES, n_origins=2)
    res = run_backtest(shared_smoke_panel(), s, progress=quiet)
    n = res.metrics.groupby(["model_name", "evaluation_basis", "level"])["n_obs"].sum().unstack(0)
    assert (n.nunique(axis=1) == 1).all()                                                    # same n_obs for every model
    assert res.n_series == 419 and len(res.origins) == 2 and res.origins[0] < res.origins[1]
    assert (res.metrics["sum_abs_error"] >= 0).all() and (res.metrics["n_obs"] >= 0).all()


def test_weekly_level_pools_the_same_totals_as_the_daily_level():
    s = forecast_settings(models="naive,croston_sba", n_origins=2)
    res = run_backtest(shared_smoke_panel(), s, progress=quiet)
    m = res.metrics[res.metrics.evaluation_basis == "observed"]
    for model in s.models:
        day = m[(m.model_name == model) & (m.level == "series_day")]
        wk = m[(m.model_name == model) & (m.level == "store_category_week")]
        assert abs(day.sum_actual.sum() - wk.sum_actual.sum()) < 1e-6 and abs(day.sum_pred.sum() - wk.sum_pred.sum()) < 1e-6
        assert wk.sum_abs_error.sum() <= day.sum_abs_error.sum() + 1e-6        # aggregating cannot increase absolute error


def test_latent_diagnostic_is_opt_in_and_separately_labelled():
    panel = shared_smoke_panel()
    s = forecast_settings(models="naive,seasonal_naive", n_origins=1)
    lat = load_latent(shared_smoke_db(), panel)
    assert lat.shape == panel.units.shape and (lat >= panel.units).all()       # demand can only exceed observed sales
    res = run_backtest(panel, s, lat, progress=quiet)
    assert res.latent_used and "latent_simulation_only" in set(res.metrics["evaluation_basis"])
    obs = _series_day_sums(res, "naive", "observed")
    latent = _series_day_sums(res, "naive", "latent_simulation_only")
    assert latent["sum_actual"] > obs["sum_actual"]                              # truncated sales understate demand


def test_censoring_policy_changes_what_a_stockout_day_teaches_the_model():
    v = np.full((1, 80), 10.0)
    v[0, 75] = 0.0                                                              # day 75: stockout, sold 0 of a demand of 10
    c = np.zeros((1, 80), dtype=bool)
    c[0, 75] = True
    panel = Panel.from_arrays(np.concatenate([v, np.full((1, 5), 10.0)], axis=1),
                              censored=np.concatenate([c, np.zeros((1, 5), dtype=bool)], axis=1))
    for policy, expected_zero in (("exclude", False), ("keep", True)):
        s = forecast_settings(models="moving_average", censor_policy=policy, horizon_days=5, n_origins=1, min_train_days=60)
        res = run_backtest(panel, s, progress=quiet)                    # the only origin is day 79 (5 actual days follow)
        # moving average of the last 28 days: includes the zero only under "keep"
        mean28 = 10.0 * 27 / 28 if expected_zero else 10.0
        sums = _series_day_sums(res, "moving_average")
        assert abs(sums["sum_pred"] / sums["n_obs"] - mean28) < 1e-9, policy


def test_backtest_refuses_too_little_history():
    panel = Panel.from_arrays(np.ones((2, 30)))
    s = forecast_settings(models="naive", min_train_days=42)
    try:
        run_backtest(panel, s, progress=quiet)
        raise AssertionError("expected a DZMIError")
    except DZMIError as exc:
        assert "Not enough history" in str(exc)


# ------------------------------------------------------------------------------------------ settings and reports
def test_settings_are_validated():
    s = load_forecast_settings("smoke")
    assert s.horizon_days == 14 and s.min_train_days == 42 and s.max_series == 0
    d = load_forecast_settings("default")
    assert d.min_train_days == 365 and d.max_series == 300 and d.n_origins == 6 and d.horizon_days == 14
    assert load_forecast_settings("smoke", models="naive , gbm").models == ("naive", "gbm")
    for bad in ({"models": "magic"}, {"models": "naive,naive"}, {"censor_policy": "ignore"}, {"horizon_days": 0},
                {"max_series": -1}, {"n_origins": 0}):
        try:
            load_forecast_settings("smoke", **bad)
            raise AssertionError(f"accepted {bad}")
        except ConfigError:
            pass
    try:
        load_forecast_settings("smoke", unknown_option=1)
        raise AssertionError("accepted an unknown option")
    except ConfigError:
        pass


def test_reports_are_ascii_labelled_synthetic_and_written_to_the_requested_folder():
    s = forecast_settings(models="naive,seasonal_naive,croston_sba", n_origins=2)
    res = run_backtest(shared_smoke_panel(), s, progress=quiet)
    text = format_summary(res)
    assert all(ord(ch) < 128 for ch in text) and "croston_sba" in text and "Observed sales" in text
    assert "SIMULATION ONLY" not in text                                         # absent unless requested
    assert "forecasting 0 everywhere would give MAE" in text and "Seconds spent per model" in text
    day = M.summarise(res.metrics)
    day = day[(day.evaluation_basis == "observed") & (day.level == "series_day")].iloc[0]
    assert f"MAE {day['sum_actual'] / day['n_obs']:.3f} (WAPE 100.0%)" in text   # reference = sum_actual / n_obs
    md = markdown_report(res, "bt_test", saved=False)
    assert DISCLAIMER in md and "truncate" in md and "exclude" in md
    with tempfile.TemporaryDirectory() as d:
        path = write_report(res, "bt_test", False, Path(d) / "reports")
        assert path.exists() and path.name == "forecast_backtest_bt_test.md"
        assert path.read_text(encoding="utf-8") == md
