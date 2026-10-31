"""Configuration loading and validation (YAML profiles + religious calendar)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .paths import config_dir


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}. Check DZMI_CONFIG_DIR or run from the repo root.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping.")
    return data


REQUIRED_KEYS = (
    "profile", "seed", "n_stores", "n_products", "start_date", "end_date", "n_customers",
    "loyalty_share", "assortment_fraction", "demand", "promotions", "prices", "inventory", "baskets",
)


@dataclass(frozen=True)
class GenConfig:
    profile: str
    seed: int
    n_stores: int
    n_products: int
    start_date: date
    end_date: date
    n_customers: int
    loyalty_share: float
    assortment_fraction: float
    demand: dict[str, Any]
    promotions: dict[str, Any]
    prices: dict[str, Any]
    inventory: dict[str, Any]
    baskets: dict[str, Any]
    calendar: dict[str, Any] = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    def fingerprint(self) -> str:
        """Stable hash of everything that influences generated data."""
        payload = json.dumps(self._as_plain(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _as_plain(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def with_overrides(self, **kw: Any) -> "GenConfig":
        clean = {k: v for k, v in kw.items() if v is not None}
        new = GenConfig(**{**self._as_plain(), **clean})
        validate_config(new)
        return new


def validate_config(cfg: GenConfig) -> None:
    errs: list[str] = []
    if cfg.n_stores < 1 or cfg.n_stores > 15:
        errs.append("n_stores must be between 1 and 15 (the built-in store list has 15 entries).")
    if cfg.n_products < 14:
        errs.append("n_products must be >= 14 (at least one product per category).")
    if cfg.end_date < cfg.start_date:
        errs.append("end_date must not be before start_date.")
    if cfg.n_days < 28:
        errs.append("The simulation needs at least 28 days.")
    if not 0 < cfg.assortment_fraction <= 1:
        errs.append("assortment_fraction must be in (0, 1].")
    if not 0 <= cfg.loyalty_share <= 1:
        errs.append("loyalty_share must be in [0, 1].")
    if cfg.n_customers < 1:
        errs.append("n_customers must be >= 1.")
    inv = cfg.inventory
    if inv["lead_time_min"] < 1 or inv["lead_time_max"] < inv["lead_time_min"]:
        errs.append("inventory lead times must satisfy 1 <= lead_time_min <= lead_time_max.")
    if len(cfg.demand["weekday_multipliers"]) != 7:
        errs.append("demand.weekday_multipliers needs exactly 7 values (Mon..Sun).")
    if cfg.demand["demand_scale"] <= 0:
        errs.append("demand.demand_scale must be > 0.")
    if errs:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errs))


def _to_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v))
    except ValueError as exc:
        raise ConfigError(f"Cannot parse date {v!r}; use YYYY-MM-DD.") from exc


def load_calendar(path: Path | None = None) -> dict[str, Any]:
    cal = _read_yaml(path or config_dir() / "calendar.yaml")
    for key in ("ramadan", "eid_al_fitr", "eid_al_adha"):
        if key not in cal:
            raise ConfigError(f"calendar.yaml is missing the '{key}' list.")
        cal[key] = [(_to_date(a), _to_date(b)) for a, b in cal[key]]
    return cal


def load_config(profile_or_path: str = "smoke", **overrides: Any) -> GenConfig:
    """Load a profile name (config/<name>.yaml) or an explicit YAML path."""
    p = Path(profile_or_path)
    path = p if p.suffix in (".yaml", ".yml") else config_dir() / f"{profile_or_path}.yaml"
    raw = _read_yaml(path)
    parent = raw.pop("extends", None)
    if parent:
        raw = _deep_merge(_read_yaml(config_dir() / f"{parent}.yaml"), raw)
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"Config {path.name} is missing keys: {', '.join(missing)}")
    raw["start_date"] = _to_date(raw["start_date"])
    raw["end_date"] = _to_date(raw["end_date"])
    cfg = GenConfig(calendar=load_calendar(), **{k: raw[k] for k in REQUIRED_KEYS})
    cfg = cfg.with_overrides(**overrides) if overrides else cfg
    validate_config(cfg)
    return cfg
