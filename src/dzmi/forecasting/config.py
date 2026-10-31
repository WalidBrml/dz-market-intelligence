"""Forecasting settings: config/forecast.yaml + profile overrides + command-line overrides, validated."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..config import _deep_merge, _read_yaml
from ..errors import ConfigError
from ..paths import config_dir

MODEL_NAMES = ("naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba", "gbm")
CENSOR_POLICIES = ("exclude", "keep")


@dataclass(frozen=True)
class ForecastSettings:
    profile: str
    horizon_days: int
    seasonal_period: int
    seed: int
    censor_policy: str
    models: tuple[str, ...]
    min_train_days: int
    n_origins: int
    origin_step_days: int
    max_series: int
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "horizon_days": self.horizon_days, "seasonal_period": self.seasonal_period,
                "seed": self.seed, "censor_policy": self.censor_policy, "models": list(self.models),
                "min_train_days": self.min_train_days, "n_origins": self.n_origins,
                "origin_step_days": self.origin_step_days, "max_series": self.max_series,
                "model_params": copy.deepcopy(self.model_params)}


def load_forecast_settings(profile: str = "smoke", **overrides: Any) -> ForecastSettings:
    """Load config/forecast.yaml, apply the profile block, then explicit overrides (None values are ignored)."""
    raw = _read_yaml(config_dir() / "forecast.yaml")
    cfg = {k: v for k, v in raw.items() if k != "profiles"}
    cfg = _deep_merge(cfg, (raw.get("profiles") or {}).get(profile) or {})
    bt = cfg.get("backtest") or {}
    flat = {"horizon_days": cfg.get("horizon_days"), "seasonal_period": cfg.get("seasonal_period"),
            "seed": cfg.get("seed"), "censor_policy": cfg.get("censor_policy"), "models": cfg.get("models"),
            "min_train_days": bt.get("min_train_days"), "n_origins": bt.get("n_origins"),
            "origin_step_days": bt.get("origin_step_days"), "max_series": bt.get("max_series")}
    for k, v in overrides.items():
        if k not in flat:
            raise ConfigError(f"Unknown forecasting option '{k}'. Valid: {', '.join(flat)}")
        if v is not None:
            flat[k] = v
    missing = [k for k, v in flat.items() if v is None]
    if missing:
        raise ConfigError(f"config/forecast.yaml is missing: {', '.join(missing)}")
    models = tuple(flat["models"].split(",")) if isinstance(flat["models"], str) else tuple(flat["models"])
    models = tuple(m.strip() for m in models if m.strip())
    bad = [m for m in models if m not in MODEL_NAMES]
    if bad or not models:
        raise ConfigError(f"Unknown model(s) {bad or models}. Valid: {', '.join(MODEL_NAMES)}")
    if len(set(models)) != len(models):
        raise ConfigError("Each model may be listed only once.")
    if flat["censor_policy"] not in CENSOR_POLICIES:
        raise ConfigError(f"censor_policy must be one of {CENSOR_POLICIES}")
    for key in ("horizon_days", "seasonal_period", "min_train_days", "n_origins", "origin_step_days"):
        if int(flat[key]) < 1:
            raise ConfigError(f"{key} must be at least 1")
    if int(flat["max_series"]) < 0:
        raise ConfigError("max_series must be 0 (all series) or a positive number")
    return ForecastSettings(profile=profile, horizon_days=int(flat["horizon_days"]),
                            seasonal_period=int(flat["seasonal_period"]), seed=int(flat["seed"]),
                            censor_policy=flat["censor_policy"], models=models,
                            min_train_days=int(flat["min_train_days"]), n_origins=int(flat["n_origins"]),
                            origin_step_days=int(flat["origin_step_days"]), max_series=int(flat["max_series"]),
                            model_params=copy.deepcopy(cfg.get("model_params") or {}))
