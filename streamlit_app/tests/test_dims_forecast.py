"""The five forecast dimensions rebuilt from forecast_backtest_metric (Power Query parity)."""

import pandas as pd
import pytest

import oracle_values as ov
from dzmi_app.dims_forecast import (
    add_horizon_key, build_dim_eval_basis, build_dim_eval_level, build_dim_horizon,
    build_dim_model, build_dim_origin, build_forecast_dims,
)


def test_dim_model(dataset):
    m = dataset.forecast_dims.model
    assert list(m.columns) == ["model_name", "model_label", "model_family", "model_order", "is_benchmark"]
    assert m["model_name"].tolist() == ov.MODELS_IN_ORDER
    assert m["model_order"].tolist() == [1, 2, 3, 4, 5, 6]
    assert dict(zip(m["model_name"], m["model_label"])) == ov.MODEL_LABELS
    assert m.set_index("model_name")["model_family"].to_dict() == {
        "naive": "Baseline", "seasonal_naive": "Baseline", "moving_average": "Baseline",
        "ses_weekday": "Statistical", "croston_sba": "Intermittent demand",
        "gbm": "Machine learning"}
    assert m.loc[m["is_benchmark"], "model_name"].tolist() == ["seasonal_naive"]


def test_dim_eval_basis(dataset):
    b = dataset.forecast_dims.basis
    assert b["evaluation_basis"].tolist() == ["observed", "observed_uncensored"]
    assert b["basis_label"].tolist() == ["Observed sales", "Observed, no-stockout days"]
    assert b["basis_order"].tolist() == [1, 2]
    # The simulator-only latent basis is mapped in the M code but absent from this export.
    assert "latent_simulation_only" not in set(b["evaluation_basis"])


def test_dim_eval_level(dataset):
    lv = dataset.forecast_dims.level
    assert lv["level"].tolist() == ["series_day", "store_category_week"]
    assert lv["level_label"].tolist() == ["Series x day", "Store x category x week"]
    assert lv["horizon_unit"].tolist() == ["Day", "Week"]
    assert lv["level_order"].tolist() == [1, 2]


def test_dim_horizon(dataset):
    h = dataset.forecast_dims.horizon
    assert len(h) == 16
    assert h["horizon_key"].is_unique
    days = h[h["level"] == "series_day"]
    weeks = h[h["level"] == "store_category_week"]
    assert days["horizon_bucket"].tolist() == list(range(1, 15))
    assert weeks["horizon_bucket"].tolist() == [1, 2]
    assert days["horizon_label"].tolist() == [f"Day {i}" for i in range(1, 15)]
    assert weeks["horizon_label"].tolist() == ["Week 1", "Week 2"]
    assert days["horizon_sort"].tolist() == [100 + i for i in range(1, 15)]
    assert weeks["horizon_sort"].tolist() == [201, 202]
    assert h["horizon_sort"].is_monotonic_increasing
    assert h["horizon_key"].iloc[0] == "series_day|1" and h["horizon_key"].iloc[-1] == "store_category_week|2"


def test_dim_origin(dataset):
    o = dataset.forecast_dims.origin
    assert o["origin_label"].tolist() == ov.ORIGINS
    assert o["origin_seq"].tolist() == [1, 2, 3, 4]
    assert o["origin_date"].dt.strftime("%Y-%m-%d").tolist() == ov.ORIGINS
    assert o["origin_date"].is_monotonic_increasing


def test_dimension_keys_cover_every_metric_row(dataset):
    fd, m = dataset.forecast_dims, dataset["forecast_backtest_metric"]
    assert set(m["model_name"]) == set(fd.model["model_name"])
    assert set(m["evaluation_basis"]) == set(fd.basis["evaluation_basis"])
    assert set(m["level"]) == set(fd.level["level"])
    assert set(m["horizon_key"]) == set(fd.horizon["horizon_key"])
    assert set(m["origin_date"]) == set(fd.origin["origin_date"])


def test_metric_grain_is_complete(dataset):
    """6 models x 4 origins x (14 days + 2 weeks) for observed, x 14 days for uncensored."""
    m = dataset["forecast_backtest_metric"]
    assert len(m) == 6 * 4 * (14 + 2) + 6 * 4 * 14 == 720
    counts = m.groupby(["evaluation_basis", "level"]).size().to_dict()
    assert counts == {("observed", "series_day"): 336, ("observed", "store_category_week"): 48,
                      ("observed_uncensored", "series_day"): 336}


def test_dimensions_are_deterministic_and_independent_of_row_order(dataset):
    m = dataset["forecast_backtest_metric"]
    shuffled = m.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a, b = build_forecast_dims(m), build_forecast_dims(shuffled)
    for name in ("model", "basis", "level", "horizon", "origin"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


def test_unknown_keys_use_the_power_query_fallbacks():
    metric = pd.DataFrame({
        "model_name": ["mystery", "naive"],
        "evaluation_basis": ["weird_basis", "observed"],
        "level": ["month", "series_day"],
        "origin_date": pd.to_datetime(["2025-04-03", "2025-03-27"]),
        "horizon_bucket": [3, 1],
    })
    model = build_dim_model(metric).set_index("model_name")
    assert model.loc["mystery", ["model_label", "model_family", "model_order", "is_benchmark"]].tolist() \
        == ["mystery", "Other", 99, False]
    assert build_dim_model(metric)["model_name"].tolist() == ["naive", "mystery"]   # fallback sorts last
    basis = build_dim_eval_basis(metric).set_index("evaluation_basis")
    assert basis.loc["weird_basis", ["basis_label", "basis_order", "basis_description"]].tolist() \
        == ["weird_basis", 99, ""]
    level = build_dim_eval_level(metric).set_index("level")
    assert level.loc["month", ["level_label", "level_order", "horizon_unit"]].tolist() \
        == ["month", 99, "Step"]
    horizon = build_dim_horizon(metric).set_index("horizon_key")
    assert horizon.loc["month|3", ["horizon_label", "horizon_sort"]].tolist() == ["Step 3", 9903]
    assert build_dim_origin(metric)["origin_label"].tolist() == ["2025-03-27", "2025-04-03"]


def test_empty_metric_gives_empty_but_well_formed_dimensions(dataset):
    empty = dataset["forecast_backtest_metric"].iloc[0:0]
    fd = build_forecast_dims(empty)
    for name in ("model", "basis", "level", "horizon", "origin"):
        assert len(getattr(fd, name)) == 0
    assert list(fd.model.columns) == list(dataset.forecast_dims.model.columns)
    assert list(fd.horizon.columns) == list(dataset.forecast_dims.horizon.columns)


def test_add_horizon_key_does_not_mutate_its_input(dataset):
    raw = dataset["v_forecast_accuracy"]
    assert "horizon_key" not in raw.columns
    out = add_horizon_key(dataset["v_forecast_accuracy_by_horizon"].drop(columns="horizon_key"))
    assert "horizon_key" in out.columns
