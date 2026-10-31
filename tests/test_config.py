from datetime import date

from dzmi.config import load_config
from dzmi.errors import ConfigError
from helpers import raises


def test_smoke_profile_dimensions():
    cfg = load_config("smoke")
    assert (cfg.n_stores, cfg.n_products, cfg.n_days) == (5, 100, 90)
    assert cfg.start_date == date(2025, 2, 1) and cfg.end_date == date(2025, 5, 1)


def test_default_profile_targets_full_scale():
    cfg = load_config("default")
    assert (cfg.n_stores, cfg.n_products) == (15, 500)
    assert cfg.start_date == date(2023, 1, 1) and cfg.end_date == date(2025, 12, 31)


def test_smoke_inherits_and_deep_merges_default():
    cfg = load_config("smoke")
    assert cfg.demand["dispersion"] == 4.0          # inherited from default
    assert cfg.demand["demand_scale"] == 0.5        # overridden
    assert cfg.inventory["lead_time_max"] == 3      # inherited, sibling of overridden keys


def test_invalid_config_has_actionable_message():
    e = raises(ConfigError, load_config, "smoke", n_stores=99)
    assert "n_stores" in str(e)


def test_end_before_start_rejected():
    raises(ConfigError, load_config, "smoke", end_date=date(2025, 1, 1))


def test_missing_profile_reports_path():
    e = raises(ConfigError, load_config, "does_not_exist")
    assert "does_not_exist" in str(e)


def test_fingerprint_stable_and_sensitive():
    a, b = load_config("smoke"), load_config("smoke")
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != load_config("smoke", seed=7).fingerprint()
