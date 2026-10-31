"""Milestone 3 documentation tests: docs/forecasting.md, the config, the dependencies and the code must agree."""
from __future__ import annotations

import re

import numpy as np

from dzmi.config import _read_yaml
from dzmi.features.build import FEATURE_COLUMNS
from dzmi.forecasting.config import CENSOR_POLICIES, MODEL_NAMES, load_forecast_settings
from dzmi.forecasting.metrics import BASES, LEVELS
from dzmi.forecasting.models import MODEL_DESCRIPTIONS
from dzmi.forecasting.views import FORECAST_VIEWS
from dzmi.paths import REPO_ROOT, config_dir
from helpers import loaded_sqlite, smoke

DOC = (REPO_ROOT / "docs" / "forecasting.md").read_text(encoding="utf-8")
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
YAML = _read_yaml(config_dir() / "forecast.yaml")


def section(view: str) -> str:
    m = re.search(rf"^### dwh\.{view}\n(.*?)(?=^### |^## |\Z)", DOC, re.S | re.M)
    assert m, f"docs/forecasting.md has no section '### dwh.{view}'"
    return m.group(1)


def test_every_feature_is_documented_once_in_order():
    table = DOC.split("### Features (gradient boosting)")[1].split("## Rolling-origin backtest")[0]
    rows = re.findall(r"^\| `([a-z0-9_]+)` \| (static|history|target day) \| .+ \|$", table, re.M)
    assert [r[0] for r in rows] == FEATURE_COLUMNS                     # same names, same order as the code
    assert len(set(r[0] for r in rows)) == len(FEATURE_COLUMNS)


def test_every_model_parameter_and_setting_is_documented():
    assert set(YAML["model_params"]) == set(MODEL_NAMES) - {"naive"}
    for name in MODEL_NAMES:
        assert f"`{name}`" in DOC and name in MODEL_DESCRIPTIONS
    for model, params in YAML["model_params"].items():
        for key in params:
            assert f"`{key}`" in DOC, f"model_params.{model}.{key} is not documented"
    for key in ("horizon_days", "seed", "censor_policy", "min_train_days", "n_origins", "origin_step_days", "max_series"):
        assert f"`{key}`" in DOC, key
    for policy in CENSOR_POLICIES:
        assert f"`{policy}`" in DOC
    assert set(YAML["backtest"]) == {"min_train_days", "n_origins", "origin_step_days", "max_series"}


def test_documented_profile_defaults_match_the_configuration():
    smoke_s, default_s = load_forecast_settings("smoke"), load_forecast_settings("default")
    assert (smoke_s.min_train_days, smoke_s.n_origins, smoke_s.origin_step_days, smoke_s.max_series) == (42, 4, 7, 0)
    assert (default_s.min_train_days, default_s.n_origins, default_s.origin_step_days, default_s.max_series) == (365, 6, 28, 300)
    assert "`smoke` uses `min_train_days` 42, 4 origins every 7 days and all series" in DOC
    assert "`default` uses 365 days, 6 origins every 28 days and 300 series" in DOC
    assert default_s.horizon_days == 14 and "next 14 days" in DOC


def test_metric_vocabulary_and_evaluation_bases_are_documented():
    for term in BASES + LEVELS:
        assert f"`{term}`" in DOC, term
    for metric in ("mae", "mse", "wape", "bias", "skill_mae_vs_seasonal_naive", "n_obs", "sum_actual", "sum_pred",
                   "sum_abs_error", "sum_sq_error"):
        assert f"`{metric}`" in DOC, metric
    assert "SIMULATION ONLY" in DOC and "forecasting 0" in DOC and "WAPE above 100" in DOC.replace("**", "")


def test_forecast_views_are_documented_with_grain_and_every_column():
    db = loaded_sqlite()
    for view in FORECAST_VIEWS:
        text = section(view)
        assert "**Grain:**" in text
        cols = list(db.query_df(f"SELECT * FROM dwh.{view} LIMIT 0").columns)
        missing = [c for c in cols if f"`{c}`" not in text]
        assert not missing, f"{view}: columns not documented: {missing}"
    assert set(re.findall(r"^### dwh\.(v_\w+)$", DOC, re.M)) == set(FORECAST_VIEWS)
    db.close()


def test_documented_tests_exist_and_commands_are_in_the_readme():
    for ref in set(re.findall(r"tests/(test_\w+\.py)", DOC)):
        assert (REPO_ROOT / "tests" / ref).exists(), ref
    for cmd in ("dzmi backtest", "dzmi forecast", "dzmi query --view v_forecast_accuracy", "--include-forecasts",
                "--latent-diagnostic", "--no-save", "--run-id"):
        assert cmd in README, f"README does not mention {cmd}"
    assert "docs/forecasting.md" in README


def test_numbers_quoted_in_the_docs_are_recomputed_from_the_data():
    cfg, raw, dwh = smoke()
    inv, lat = raw["inventory_daily"], raw["sim_latent_demand"]
    stockout_share = inv["stockout_flag"].mean()
    m = inv.merge(lat, left_on=["store_id", "product_id", "inventory_date"], right_on=["store_id", "product_id", "demand_date"])
    lost = m["unmet_demand_units"].sum() / m["latent_demand_units"].sum()
    assert round(stockout_share * 100) == 2 and round(lost * 100, 1) == 1.4
    assert "about 2 % of store-product-days end in a stockout" in DOC and "about 1.4 % of demand" in DOC
    assert (inv["units_sold"] == 0).mean() < 0.3                          # smoke is dense (the default profile is not)


def test_dependency_is_declared_and_only_gbm_needs_it():
    assert "scikit-learn" in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn" in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    root = REPO_ROOT / "src" / "dzmi"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.match(r"^(from|import)\s+sklearn", line):                # module-level import would make it mandatory
                raise AssertionError(f"{path.name}: sklearn must be imported lazily")
