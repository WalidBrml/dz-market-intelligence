"""Forecast Accuracy by Horizon. Scenarios are compared with oracle_forecast.py (independent SQL sums)."""

import pandas as pd
import pytest

import oracle_forecast as ov
from dzmi_app.forecast_horizon import AXIS_DAY, AXIS_WEEK, compute_horizon, horizon_axis_title
from dzmi_app.forecasting import ForecastFilters, load_forecast_context, model_comparison

NAMES = list(ov.SCENARIOS)
REL = 1e-9


@pytest.fixture(scope="module")
def forecast_context():
    return load_forecast_context()


def filters_for(name):
    sc = ov.SCENARIOS[name]
    return ForecastFilters(bases=tuple(sc.get("bases", ())), levels=tuple(sc.get("levels", ())),
                           models=tuple(sc.get("models", ())), origins=tuple(sc.get("origins", ())))


def label(level, bucket):
    return ("Week " if level == "store_category_week" else "Day ") + str(bucket)


@pytest.mark.parametrize("name", NAMES)
def test_horizons_and_observations_match_the_oracle(forecast_context, name):
    r, e = compute_horizon(forecast_context, filters_for(name)), ov.ORACLES[name]
    assert list(r.observations["horizon_label"]) == [label(e["level"], x["bucket"]) for x in e["horizon"]]
    assert list(r.observations["observations"]) == [x["observations"] for x in e["horizon"]]


@pytest.mark.parametrize("name", NAMES)
def test_metrics_by_horizon_and_model_match_the_oracle(forecast_context, name):
    r, e = compute_horizon(forecast_context, filters_for(name)), ov.ORACLES[name]
    got = r.by_model.set_index(["horizon_label", "model_name"])
    expected = {(label(e["level"], x["bucket"]), m): v for x in e["horizon"] for m, v in x["per_model"].items()}
    assert set(got.index) == set(expected)
    for key, v in expected.items():
        assert got.loc[key, "observations"] == v["n"]
        for col in ("mae", "wape", "bias"):
            assert got.loc[key, col] == pytest.approx(v[col], rel=REL)


@pytest.mark.parametrize("name", NAMES)
def test_pivot_matches_the_by_model_values(forecast_context, name):
    r = compute_horizon(forecast_context, filters_for(name))
    for _, row in r.by_model.iterrows():
        assert r.mae_pivot.loc[row["horizon_label"], row["model_label"]] == pytest.approx(row["mae"], rel=REL)
    assert list(r.mae_pivot.index) == list(r.observations["horizon_label"])


def test_day_level_has_fourteen_day_buckets_in_numeric_order(forecast_context):
    r = compute_horizon(forecast_context, ForecastFilters())
    assert list(r.observations["horizon_label"]) == [f"Day {i}" for i in range(1, 15)]
    assert list(r.observations["horizon_sort"]) == [101 + i for i in range(14)]
    assert list(r.observations["horizon_label"]).index("Day 2") < list(r.observations["horizon_label"]).index("Day 10")


def test_week_level_has_two_week_buckets_and_never_mixes_with_days(forecast_context):
    r = compute_horizon(forecast_context, filters_for("weekly"))
    assert list(r.observations["horizon_label"]) == ["Week 1", "Week 2"]
    assert not any(x.startswith("Day") for x in r.by_model["horizon_label"])
    assert r.axis_title == AXIS_WEEK
    assert compute_horizon(forecast_context, ForecastFilters()).axis_title == AXIS_DAY


@pytest.mark.parametrize("levels,expected", [((), AXIS_DAY), (("series_day",), AXIS_DAY),
                                             (("store_category_week",), AXIS_WEEK),
                                             (("series_day", "store_category_week"), AXIS_DAY)])
def test_axis_title_follows_the_scope_level(levels, expected):
    assert horizon_axis_title(ForecastFilters(levels=levels)) == expected


def test_observations_are_pooled_over_all_selected_models_like_the_report_measure(forecast_context):
    all_models = compute_horizon(forecast_context, ForecastFilters())
    one_model = compute_horizon(forecast_context, filters_for("benchmark_only"))
    assert list(all_models.observations["observations"]) == [6 * v for v in one_model.observations["observations"]]
    two = compute_horizon(forecast_context, filters_for("two_models"))
    assert list(two.observations["observations"]) == [2 * v for v in one_model.observations["observations"]]


def test_horizon_totals_add_up_to_the_scope_observations(forecast_context):
    r = compute_horizon(forecast_context, ForecastFilters())
    assert int(r.observations["observations"].sum()) == 6 * 23_464


def test_pooling_horizons_is_weighted_not_a_simple_mean(forecast_context):
    """Each model's MAE over all horizons equals sum(n * MAE by horizon) / sum(n)."""
    r = compute_horizon(forecast_context, ForecastFilters())
    pooled = model_comparison(forecast_context, ForecastFilters()).set_index("model_name")["mae"]
    for model, part in r.by_model.groupby("model_name"):
        weighted = (part["mae"] * part["observations"]).sum() / part["observations"].sum()
        assert weighted == pytest.approx(pooled[model], rel=1e-12)


def test_uncensored_basis_at_week_level_has_no_horizons(forecast_context):
    r = compute_horizon(forecast_context, filters_for("uncensored_weekly"))
    assert r.observations.empty and r.by_model.empty and r.mae_pivot.empty
    assert r.axis_title == AXIS_WEEK


def test_origin_and_model_filters_reach_the_horizon_values(forecast_context):
    one = compute_horizon(forecast_context, filters_for("one_origin"))
    assert (one.observations["observations"] == 6 * 419).all()      # 419 series x 1 origin x 6 models per day bucket
    assert set(compute_horizon(forecast_context, filters_for("two_models")).by_model["model_name"]) == {"croston_sba", "naive"}


def test_scope_message_is_carried_through(forecast_context):
    assert compute_horizon(forecast_context, filters_for("weekly")).scope_message == ov.ORACLES["weekly"]["scope_message"]


def test_computing_does_not_modify_the_shared_frame(forecast_context):
    before = int(pd.util.hash_pandas_object(forecast_context.metric, index=True).sum())
    for n in NAMES:
        compute_horizon(forecast_context, filters_for(n))
    assert int(pd.util.hash_pandas_object(forecast_context.metric, index=True).sum()) == before
