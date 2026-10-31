"""Shared test helpers (importable under pytest via pyproject pythonpath, and by the fallback runner)."""
from __future__ import annotations

from functools import lru_cache

from dzmi.config import load_config
from dzmi.data_generation.generator import generate
from dzmi.ingestion.db import Database
from dzmi.ingestion.loader import load_all
from dzmi.ingestion.warehouse import build_warehouse


@lru_cache(maxsize=1)
def smoke():
    cfg = load_config("smoke")
    raw = generate(cfg)
    return cfg, raw, build_warehouse(raw, cfg)


@lru_cache(maxsize=1)
def two_year():
    """Small two-year dataset (2023-01-01..2024-12-31, includes the leap day 29 Feb 2024).

    The smoke profile spans only Feb-May 2025, so no prior-year comparison exists there. This one is for
    year-over-year tests. 3 stores x 30 products keeps it fast (about a second).
    """
    from datetime import date
    cfg = load_config("smoke", start_date=date(2023, 1, 1), end_date=date(2024, 12, 31), n_stores=3, n_products=30)
    raw = generate(cfg)
    return cfg, raw, build_warehouse(raw, cfg)


@lru_cache(maxsize=1)
def shared_smoke_db() -> Database:
    """Smoke data loaded once into an in-memory SQLite database. READ ONLY: tests must not modify it."""
    cfg, raw, dwh = smoke()
    db = Database.sqlite()
    db.apply_schema()
    load_all(db, raw, dwh)
    return db


@lru_cache(maxsize=1)
def shared_two_year_db() -> Database:
    """The two-year dataset loaded once into in-memory SQLite. READ ONLY."""
    cfg, raw, dwh = two_year()
    db = Database.sqlite()
    db.apply_schema()
    load_all(db, raw, dwh)
    return db


@lru_cache(maxsize=1)
def shared_smoke_panel():
    """Forecasting panel of the smoke data (all 419 series, 90 days). READ ONLY."""
    from dzmi.features.panel import load_panel
    return load_panel(shared_smoke_db())


def forecast_settings(profile: str = "smoke", **overrides):
    from dzmi.forecasting.config import load_forecast_settings
    return load_forecast_settings(profile, **overrides)


def copies():
    """Fresh deep copies so a test can corrupt data without affecting others."""
    cfg, raw, dwh = smoke()
    return cfg, {k: v.copy() for k, v in raw.items()}, {k: v.copy() for k, v in dwh.items()}


def loaded_sqlite() -> Database:
    cfg, raw, dwh = smoke()
    db = Database.sqlite()
    db.apply_schema()
    load_all(db, raw, dwh)
    return db


def raises(exc_type, fn, *a, **kw):
    """Tiny assert-raises that works with or without pytest. Returns the exception."""
    try:
        fn(*a, **kw)
    except exc_type as e:
        return e
    raise AssertionError(f"expected {exc_type.__name__} to be raised")
