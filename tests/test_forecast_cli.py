"""Milestone 3: the backtest / forecast commands end to end on a real SQLite file, plus optional export and errors."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import pandas as pd

from dzmi import cli
from dzmi.analytics.bi_export import BI_FORECAST_TABLES, BI_TABLES, BI_VIEWS, export_bi
from dzmi.errors import DatabaseError, DZMIError
from dzmi.forecasting.gbm import INSTALL_HINT, _regressor
from dzmi.ingestion.db import Database
from dzmi.ingestion.loader import load_all
from helpers import raises, shared_smoke_db, smoke


def run_cli(*argv):
    """Run one dzmi command through the real parser and return (exit code, printed text)."""
    a = cli.build_parser().parse_args(list(argv))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = a.func(a)
    return code, buf.getvalue()


class TempProject:
    """A file-backed SQLite database holding the smoke data, and a private reports folder."""
    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.base = d / "t.sqlite"
        cfg, raw, dwh = smoke()
        with Database.sqlite(self.base) as db:
            db.apply_schema()
            load_all(db, raw, dwh)
        self.db_args = ["--profile", "smoke", "--backend", "sqlite", "--sqlite-path", str(self.base)]
        self.reports = d / "reports"
        self._orig = cli.reports_dir
        cli.reports_dir = lambda: self.reports            # the commands must not write into the real repo folder
        return self

    def count(self, table):
        with Database.sqlite(self.base) as db:
            return db.row_count("dwh", table)

    def __exit__(self, *exc):
        cli.reports_dir = self._orig
        self.tmp.cleanup()                                 # fails on Windows if any connection was left open


def test_parser_exposes_backtest_and_forecast_and_no_destructive_option():
    p = cli.build_parser()
    a = p.parse_args(["backtest", "--profile", "default", "--backend", "sqlite", "--models", "naive,gbm", "--origins", "3",
                      "--step", "14", "--min-train", "100", "--max-series", "50", "--latent-diagnostic", "--no-save"])
    assert a.func is cli.cmd_backtest and a.origins == 3 and a.latent_diagnostic and a.no_save and a.max_series == 50
    f = p.parse_args(["forecast", "--model", "gbm", "--origin", "2025-04-10", "--run-id", "fc_x"])
    assert f.func is cli.cmd_forecast and f.model == "gbm" and f.origin == "2025-04-10"
    d = p.parse_args(["forecast"])
    assert d.model == "seasonal_naive" and d.origin is None
    for a in (a, f, d):
        assert not hasattr(a, "reset")                     # neither command can reset or delete anything


def test_backtest_command_end_to_end_saves_metrics_and_a_report():
    with TempProject() as t:
        code, out = run_cli("backtest", *t.db_args, "--models", "naive,seasonal_naive,croston_sba", "--origins", "2",
                            "--run-id", "bt_cli")
        assert code == 0 and "Saved" in out and "bt_cli" in out and "croston_sba" in out and "SYNTHETIC DATA" in out
        assert all(ord(c) < 128 for c in out)                                  # safe for any Windows console
        assert t.count("backtest_run") == 1 and t.count("forecast_backtest_metric") > 0
        report = t.reports / "forecast_backtest_bt_cli.md"
        assert report.exists() and "saved to the database: yes" in report.read_text(encoding="utf-8")
        before = t.count("forecast_backtest_metric")
        raises(DZMIError, run_cli, "backtest", *t.db_args, "--models", "naive", "--run-id", "bt_cli")   # never overwrites
        assert t.count("forecast_backtest_metric") == before and t.count("backtest_run") == 1
        code, out = run_cli("query", *t.db_args, "--view", "v_forecast_accuracy", "--columns",
                            "model_name,evaluation_basis,level,mae")
        assert code == 0 and "croston_sba" in out and "observed" in out


def test_backtest_no_save_and_latent_diagnostic_flags():
    with TempProject() as t:
        code, out = run_cli("backtest", *t.db_args, "--models", "naive,seasonal_naive", "--origins", "1", "--no-save",
                            "--latent-diagnostic")
        assert "SIMULATION ONLY" in out and "Saved" not in out
        assert t.count("backtest_run") == 0 and t.count("forecast_backtest_metric") == 0
        assert any(t.reports.glob("*.md"))
        code, out = run_cli("backtest", *t.db_args, "--models", "naive,seasonal_naive", "--origins", "1", "--run-id", "bt_l",
                            "--latent-diagnostic")
        with Database.sqlite(t.base) as db:
            bases = set(db.query_df("SELECT DISTINCT evaluation_basis AS b FROM dwh.forecast_backtest_metric")["b"])
        assert bases == {"observed", "observed_uncensored", "latent_simulation_only"}


def test_forecast_command_stores_a_run_and_validates_the_origin():
    with TempProject() as t:
        code, out = run_cli("forecast", *t.db_args, "--model", "croston_sba", "--origin", "2025-04-10", "--run-id", "fc_cli")
        assert code == 0 and "419 series" in out and "5,866 rows" in out and "2025-04-11 to 2025-04-24" in out
        assert t.count("model_run") == 1 and t.count("fact_demand_forecast") == 5866
        raises(DZMIError, run_cli, "forecast", *t.db_args, "--model", "croston_sba", "--run-id", "fc_cli")
        raises(DZMIError, run_cli, "forecast", *t.db_args, "--origin", "2025-04-30")      # no room for 14 days after it
        raises(DZMIError, run_cli, "forecast", *t.db_args, "--origin", "2025-02-20")      # not enough history before it
        raises(DZMIError, run_cli, "forecast", *t.db_args, "--origin", "2024-01-01")      # outside the calendar
        raises(DZMIError, run_cli, "forecast", *t.db_args, "--model", "magic")
        assert t.count("fact_demand_forecast") == 5866 and t.count("model_run") == 1       # failures changed nothing
        code, out = run_cli("forecast", *t.db_args, "--model", "naive")                    # default origin, auto run id
        assert "2025-04-18 to 2025-05-01" in out and t.count("model_run") == 2


def test_commands_do_not_create_a_database_that_does_not_exist():
    with tempfile.TemporaryDirectory() as d:
        args = ["--profile", "smoke", "--backend", "sqlite", "--sqlite-path", str(Path(d) / "none.sqlite")]
        for cmd in (["backtest", "--models", "naive"], ["forecast"]):
            raises(DatabaseError, run_cli, *cmd, *args)
        assert not list(Path(d).iterdir())


def test_missing_scikit_learn_gives_an_install_hint_before_any_work():
    real = sys.modules.get("sklearn.ensemble")
    sys.modules["sklearn.ensemble"] = None                  # makes "from sklearn.ensemble import ..." raise ImportError
    try:
        try:
            _regressor({}, 0)
            raise AssertionError("expected DZMIError")
        except DZMIError as exc:
            assert "scikit-learn" in str(exc) and "pip install" in str(exc) and str(exc) == INSTALL_HINT
        with TempProject() as t:
            raises(DZMIError, run_cli, "backtest", *t.db_args, "--models", "naive,gbm", "--run-id", "bt_g")
            raises(DZMIError, run_cli, "forecast", *t.db_args, "--model", "gbm")
            assert t.count("backtest_run") == 0 and t.count("model_run") == 0
            code, out = run_cli("backtest", *t.db_args, "--models", "naive,seasonal_naive", "--origins", "1")
            assert code == 0                                  # the baselines never need scikit-learn
    finally:
        if real is not None:
            sys.modules["sklearn.ensemble"] = real
        else:
            sys.modules.pop("sklearn.ensemble", None)


def test_query_on_a_database_without_forecast_views_says_how_to_upgrade():
    with TempProject() as t:
        with Database.sqlite(t.base) as db:
            db._h.execute("DROP VIEW dwh.v_forecast_accuracy")
        try:
            run_cli("query", *t.db_args, "--view", "v_forecast_accuracy")
            raise AssertionError("expected DatabaseError")
        except DatabaseError as exc:
            assert "init-db" in str(exc)
        run_cli("init-db", *t.db_args)                                        # the advertised fix keeps the data
        code, out = run_cli("query", *t.db_args, "--view", "v_forecast_accuracy")
        assert code == 0 and "(0 of 0 rows)" in out
        assert t.count("fact_sales") == 48216
        code, out = run_cli("query", "--list")
        assert "v_forecast_accuracy_by_horizon" in out and "v_kpi_store" in out


def test_forecast_export_is_optional_and_the_default_export_is_unchanged():
    with TempProject() as t:
        run_cli("backtest", *t.db_args, "--models", "naive,seasonal_naive", "--origins", "1", "--run-id", "bt_e")
        run_cli("forecast", *t.db_args, "--model", "naive", "--origin", "2025-04-10", "--run-id", "fc_e")
        plain, full = t.reports.parent / "bi_plain", t.reports.parent / "bi_full"
        code, out = run_cli("export-bi", *t.db_args, "--out", str(plain))
        names = {p.stem for p in plain.glob("*.csv")}
        assert names == {n for n, _, _ in BI_TABLES + BI_VIEWS} and len(names) == 16     # exactly the Milestone 2 export
        code, out = run_cli("export-bi", *t.db_args, "--out", str(full), "--include-forecasts")
        extra = {n for n, _, _ in BI_FORECAST_TABLES}
        assert {p.stem for p in full.glob("*.csv")} == names | extra and "fact_demand_forecast" in out
        fc = pd.read_csv(full / "fact_demand_forecast.csv", dtype=str)
        assert len(fc) == 5866 and not fc["store_key"].str.contains(r"\.").any() and not fc["date_key"].str.contains(r"\.").any()
        assert pd.read_csv(full / "model_run.csv").shape[0] == 1
        assert pd.read_csv(full / "v_forecast_accuracy.csv").shape[0] == 6                 # 2 models x 3 (basis, level) pairs


def test_export_of_forecast_tables_on_an_empty_forecast_store_writes_headers_only():
    with tempfile.TemporaryDirectory() as d:
        counts = export_bi(shared_smoke_db(), Path(d), include_forecasts=True)
        assert counts["model_run"] == 0 and counts["fact_demand_forecast"] == 0
        header = pd.read_csv(Path(d) / "fact_demand_forecast.csv", nrows=0).columns.tolist()
        assert header[:3] == ["forecast_run_id", "store_key", "product_key"] and "predicted_units" in header
