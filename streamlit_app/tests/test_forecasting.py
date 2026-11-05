"""Forecasting Performance (backtest). Every scenario is compared with oracle_forecast.py, produced by
plain SQL sums (tests/oracle_generator_forecast.py), not by dzmi_app. No future forecast exists here."""

import pandas as pd
import pytest

import oracle_forecast as ov
from dzmi_app.forecasting import (
    EMPTY_NOTE, ForecastFilters, best_by_basis_level, build_forecast_context, cards, filters_from_labels,
    load_forecast_context, mae_by_origin, model_comparison, scope_message,
)

NAMES = list(ov.SCENARIOS)
REL = 1e-9


@pytest.fixture(scope="module")
def forecast_context():
    return load_forecast_context()


def filters_for(name):
    sc = ov.SCENARIOS[name]
    return ForecastFilters(bases=tuple(sc.get("bases", ())), levels=tuple(sc.get("levels", ())),
                           models=tuple(sc.get("models", ())), origins=tuple(sc.get("origins", ())))


def close(actual, expected, rel=REL):
    if expected is None:
        assert actual is None or pd.isna(actual)
    else:
        assert actual == pytest.approx(expected, rel=rel)


# ------------------------------------------------------------------ context: backtest vs future forecast


def test_no_future_forecast_exists_and_none_is_produced(forecast_context):
    c = forecast_context
    assert c.forecast_rows == 0 and c.model_run_rows == 0
    assert "predicted_units" not in c.metric.columns                       # only additive backtest sums
    assert set(c.metric["origin_label"]) <= set(c.origin_labels)
    assert c.metric["origin_date"].max() < pd.Timestamp("2025-05-01")      # origins precede the last data day


def test_run_metadata(forecast_context):
    c = forecast_context
    assert c.n_runs == 1
    assert c.run == {"backtest_run_id": "smoke_m3_20261025_01", "n_series": 419, "horizon_days": 14,
                     "censor_policy": "exclude"}
    assert c.origin_labels == ("2025-03-27", "2025-04-03", "2025-04-10", "2025-04-17")
    assert c.model_order == ("naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba", "gbm")


def test_context_is_cached_and_rebuildable(dataset, forecast_context):
    assert load_forecast_context() is forecast_context
    assert len(build_forecast_context(dataset).metric) == 720


def test_labels_are_translated_into_keys(forecast_context):
    f = filters_from_labels(forecast_context, bases=["Observed, no-stockout days"], levels=["Series x day"],
                            models=["Croston-SBA", "Naive"], origins=["2025-04-10"])
    assert f == ForecastFilters(bases=("observed_uncensored",), levels=("series_day",),
                                models=("croston_sba", "naive"), origins=("2025-04-10",))


# ------------------------------------------------------------------ scenarios vs the independent oracle


@pytest.mark.parametrize("name", NAMES)
def test_model_comparison_matches_the_oracle(forecast_context, name):
    t, e = model_comparison(forecast_context, filters_for(name)), ov.ORACLES[name]["models"]
    assert list(t["model_label"]) == [r["label"] for r in e]
    for row, r in zip(t.itertuples(index=False), e):
        assert row.observations == r["n"]
        for col in ("mae", "rmse", "wape", "bias"):
            close(getattr(row, col), r[col])
        close(row.benchmark_mae, r["benchmark_mae"])
        close(row.skill, r["skill"])
        assert row.rank == r["rank"] and row.origins_won == r["origins_won"]


@pytest.mark.parametrize("name", NAMES)
def test_mae_by_origin_matches_the_oracle(forecast_context, name):
    t = mae_by_origin(forecast_context, filters_for(name))
    got = {(m, o): v for m, o, v in zip(t["model_name"], t["origin_label"], t["mae"])}
    exp = {(m, o): v for m, d in ov.ORACLES[name]["by_origin"].items() for o, v in d.items() if v is not None}
    assert set(got) == set(exp)
    assert [got[k] for k in sorted(exp)] == pytest.approx([exp[k] for k in sorted(exp)], rel=REL)


@pytest.mark.parametrize("name", NAMES)
def test_best_model_matrix_matches_the_oracle(forecast_context, name):
    c = forecast_context
    t, e = best_by_basis_level(c, filters_for(name)), ov.ORACLES[name]["best_table"]
    inv_b = {v: k for k, v in c.basis_labels.items()}
    inv_l = {v: k for k, v in c.level_labels.items()}
    assert [(inv_b[r.basis_label], inv_l[r.level_label], r.best_model, r.models_beating) for r in t.itertuples()] \
        == [(r["basis"], r["level"], r["best_model"], r["models_beating"]) for r in e]
    for row, r in zip(t.itertuples(), e):
        close(row.best_mae, r["best_mae"]); close(row.benchmark_mae, r["benchmark_mae"])
        close(row.best_vs_benchmark, r["best_vs_benchmark"])


@pytest.mark.parametrize("name", NAMES)
def test_cards_scope_message_and_empty_state(forecast_context, name):
    c, e = cards(forecast_context, filters_for(name)), ov.ORACLES[name]
    assert c["scope_message"] == e["scope_message"] == scope_message(forecast_context, filters_for(name))
    assert c["rolling_origins"] == e["origins_count"]
    assert (c["empty_note"] is not None) == e["empty"]
    assert c["run"] == "smoke_m3_20261025_01" and c["series_scored"] == 419
    assert c["horizon_days"] == 14 and c["stockout_policy"] == "exclude"


# ------------------------------------------------------------------ audit values and business rules


def test_audit_default_comparison(forecast_context):
    t = model_comparison(forecast_context, ForecastFilters()).set_index("model_label")
    assert t.loc["Croston-SBA", "mae"] == pytest.approx(2.5094, abs=5e-5)
    assert t.loc["Naive", "mae"] == pytest.approx(3.6419, abs=5e-5)
    assert t.loc["Croston-SBA", "skill"] == pytest.approx(0.2642, abs=5e-5)
    assert t.loc["Naive", "skill"] == pytest.approx(-0.0679, abs=5e-5)
    assert t.loc["Seasonal naive (benchmark)", "skill"] == 0.0
    assert (t["observations"] == 23_464).all()
    assert list(t.index)[:2] == ["Croston-SBA", "Exp. smoothing + weekday"]


def test_default_scope_is_observed_series_day(forecast_context):
    assert cards(forecast_context, ForecastFilters())["scope_message"] == \
        "Scope: Observed sales (default)  |  Series x day (default)  |  pooled over all selected origins"


def test_ranking_is_by_mae_ascending_with_dense_ranks(forecast_context):
    t = model_comparison(forecast_context, ForecastFilters())
    assert t["mae"].is_monotonic_increasing
    assert list(t["rank"]) == [1, 2, 3, 4, 5, 6]


def test_origins_won_counts_the_origins_of_the_slicer(forecast_context):
    t = model_comparison(forecast_context, ForecastFilters()).set_index("model_name")
    assert t.loc["croston_sba", "origins_won"] == 3 and t.loc["gbm", "origins_won"] == 1
    assert t["origins_won"].sum() == 4
    w = model_comparison(forecast_context, filters_for("weekly")).set_index("model_name")
    assert w.loc["croston_sba", "origins_won"] == 1 and w.loc["ses_weekday", "origins_won"] == 3
    one = model_comparison(forecast_context, filters_for("one_origin"))
    assert one["origins_won"].sum() == 1


def test_the_benchmark_ignores_the_model_filter_per_the_approved_specification(forecast_context):
    t = model_comparison(forecast_context, filters_for("models_no_benchmark_weekly"))
    assert "seasonal_naive" not in set(t["model_name"])
    assert t["benchmark_mae"].iloc[0] == pytest.approx(ov.ORACLES["models_no_benchmark_weekly"]["models"][0]["benchmark_mae"], rel=REL)
    assert t["benchmark_mae"].notna().all()


def test_rank_is_among_the_selected_models_only(forecast_context):
    t = model_comparison(forecast_context, filters_for("two_models"))
    assert list(t["model_label"]) == ["Croston-SBA", "Naive"] and list(t["rank"]) == [1, 2]


def test_the_benchmark_row_has_skill_exactly_zero_and_the_benchmark_only_view_has_nothing_beating_it(forecast_context):
    t = model_comparison(forecast_context, filters_for("benchmark_only"))
    assert list(t["skill"]) == [0.0]
    assert best_by_basis_level(forecast_context, filters_for("benchmark_only"))["models_beating"].eq(0).all()


def test_no_stockout_basis_exists_only_at_series_day_level(forecast_context):
    f = filters_for("uncensored_weekly")
    assert model_comparison(forecast_context, f).empty and mae_by_origin(forecast_context, f).empty
    assert best_by_basis_level(forecast_context, f).empty
    assert cards(forecast_context, f)["empty_note"] == EMPTY_NOTE


def test_selecting_both_bases_falls_back_to_the_default_scope(forecast_context):
    pd.testing.assert_frame_equal(model_comparison(forecast_context, filters_for("both_bases")),
                                  model_comparison(forecast_context, ForecastFilters()))


def test_origin_filter_pools_only_the_selected_origins(forecast_context):
    two = model_comparison(forecast_context, filters_for("two_origins")).set_index("model_name")
    assert (two["observations"] == 11_732).all()
    assert cards(forecast_context, filters_for("two_origins"))["rolling_origins"] == 2


def test_best_model_matrix_lists_only_existing_combinations_sorted_by_basis_then_level(forecast_context):
    t = best_by_basis_level(forecast_context, ForecastFilters())
    assert list(zip(t["basis_label"], t["level_label"])) == [
        ("Observed sales", "Series x day"), ("Observed sales", "Store x category x week"),
        ("Observed, no-stockout days", "Series x day")]
    restricted = best_by_basis_level(forecast_context, filters_for("weekly"))
    assert list(restricted["level_label"]) == ["Store x category x week"]


def test_mae_by_origin_has_one_point_per_model_and_origin(forecast_context):
    t = mae_by_origin(forecast_context, ForecastFilters())
    assert len(t) == 6 * 4 and set(t["origin_label"]) == set(forecast_context.origin_labels)


def test_the_first_origin_is_the_hardest_for_every_model(forecast_context):
    t = mae_by_origin(forecast_context, ForecastFilters()).pivot(index="origin_label", columns="model_name", values="mae")
    assert (t.idxmax() == "2025-03-27").all()


def test_computing_does_not_modify_the_shared_frame(forecast_context):
    before = int(pd.util.hash_pandas_object(forecast_context.metric, index=True).sum())
    for n in NAMES:
        model_comparison(forecast_context, filters_for(n)); mae_by_origin(forecast_context, filters_for(n))
        best_by_basis_level(forecast_context, filters_for(n))
    assert int(pd.util.hash_pandas_object(forecast_context.metric, index=True).sum()) == before
