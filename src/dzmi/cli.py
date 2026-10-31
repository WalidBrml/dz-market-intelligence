"""Command-line interface: dzmi <command>. Run `dzmi --help`."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

from . import SYNTHETIC_DISCLAIMER
from .analytics.bi_export import export_bi
from .analytics.views import ANALYTIC_VIEWS, format_view, list_views, read_view
from .config import load_config
from .data_generation.generator import generate
from .data_generation.io import read_raw, write_raw
from .errors import DatabaseError, DZMIError
from .forecasting.views import FORECAST_VIEWS
from .ingestion.db import Database
from .ingestion.db_validation import run_db_checks
from .ingestion.loader import load_all
from .ingestion.quality import assert_quality, run_quality_checks, summarize, write_report
from .ingestion.warehouse import build_warehouse
from .logging_utils import setup_logging
from .paths import default_sqlite_path, bi_export_dir, raw_dir, reports_dir

log = logging.getLogger("dzmi.cli")


def _cfg(a: argparse.Namespace):
    ov = {"seed": a.seed, "n_stores": a.stores, "n_products": a.products,
          "start_date": date.fromisoformat(a.start) if a.start else None,
          "end_date": date.fromisoformat(a.end) if a.end else None}
    return load_config(a.profile, **ov)


def _db(a: argparse.Namespace) -> Database:
    if a.backend == "sqlite":
        path = None if a.sqlite_path == ":memory:" else (a.sqlite_path or default_sqlite_path(a.profile))
        return Database.sqlite(path)
    return Database.postgres()


def _require_existing_db(a: argparse.Namespace) -> None:
    """Read-only commands must not silently create an empty SQLite database."""
    if a.backend != "sqlite":
        return
    if a.sqlite_path == ":memory:":
        raise DatabaseError(":memory: databases disappear when a command ends. Use a file path (the default).")
    base = Path(a.sqlite_path or default_sqlite_path(a.profile))
    dwh_file = base.with_name(f"{base.stem}_dwh.sqlite")
    if not dwh_file.exists():
        raise DatabaseError(f"SQLite database not found: {dwh_file}. Load it first: "
                            f"dzmi load --profile {a.profile} --backend sqlite")


def _print_summary(results, title: str) -> bool:
    s = summarize(results)
    print(f"\n{title}: {s['passed']}/{s['checks']} checks passed, {s['errors']} error(s), {s['warnings']} warning(s)")
    for r in results:
        if not r.passed:
            print(f"  [{r.severity.upper()}] {r.name}: {r.violations:,} violation(s) - {r.hint or r.description}")
    return s["ok"]


def cmd_generate(a) -> int:
    cfg = _cfg(a)
    t = time.perf_counter()
    raw = generate(cfg)
    out = Path(a.out) if a.out else raw_dir(cfg.profile)
    write_raw(raw, out, cfg)
    print(f"Generated {len(raw['sales_transaction_line']):,} sales lines, {len(raw['inventory_daily']):,} inventory rows "
          f"in {time.perf_counter() - t:.1f}s -> {out}\n{SYNTHETIC_DISCLAIMER}")
    return 0


def cmd_validate(a) -> int:
    cfg = _cfg(a)
    raw = read_raw(Path(a.out) if a.out else raw_dir(cfg.profile))
    results = run_quality_checks(raw, build_warehouse(raw, cfg))
    jp, mp = write_report(results, reports_dir(), cfg.profile)
    ok = _print_summary(results, "Data quality (pre-load)")
    print(f"Report: {mp}  |  {jp}")
    return 0 if ok else 1


def cmd_init_db(a) -> int:
    with _db(a) as db:
        if a.reset:
            db.reset()
        print("Applied:", ", ".join(db.apply_schema()), f"({db.description})")
    return 0


def cmd_load(a) -> int:
    cfg = _cfg(a)
    raw = read_raw(Path(a.out) if a.out else raw_dir(cfg.profile))
    dwh = build_warehouse(raw, cfg)
    assert_quality(run_quality_checks(raw, dwh))          # never load data that fails blocking checks
    with _db(a) as db:
        if a.reset:
            db.reset()
        db.apply_schema()
        counts = load_all(db, raw, dwh)
        print(f"Loaded {sum(counts.values()):,} rows into {len(counts)} tables ({db.description}).")
        results = run_db_checks(db)
    jp, mp = write_report(results, reports_dir(), f"{cfg.profile}_db")
    ok = _print_summary(results, "Database validation")
    print(f"Report: {mp}")
    return 0 if ok else 1


def cmd_validate_db(a) -> int:
    with _db(a) as db:
        results = run_db_checks(db)
    write_report(results, reports_dir(), f"{a.profile}_db")
    return 0 if _print_summary(results, "Database validation") else 1


def cmd_query(a) -> int:
    catalog = {**ANALYTIC_VIEWS, **FORECAST_VIEWS}
    if a.list:
        for name, (grain, _) in catalog.items():
            print(f"{name:32s} {grain}")
        return 0
    if not a.view:
        raise DZMIError("Choose a view with --view NAME, or list them with: dzmi query --list")
    if a.view not in catalog:
        raise DZMIError(f"Unknown view '{a.view}'. List the valid names with: dzmi query --list")
    _require_existing_db(a)
    with _db(a) as db:
        try:
            df = read_view(db, a.view, catalog)
        except DatabaseError:
            raise
        except Exception as exc:
            if a.view in FORECAST_VIEWS:
                raise DatabaseError(f"{exc}. If this database was created before Milestone 3, add the forecast tables "
                                    "and views (no data is lost) with: dzmi init-db --backend <postgres|sqlite>") from exc
            raise
    if a.csv:
        out = Path(a.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {out}")
        return 0
    cols = [c.strip() for c in a.columns.split(",") if c.strip()] if a.columns else None
    print(format_view(df, None if a.limit == 0 else a.limit, a.order_by, a.desc, cols))
    return 0


def cmd_export_bi(a) -> int:
    _require_existing_db(a)
    out = Path(a.out) if a.out else bi_export_dir(a.profile)
    with _db(a) as db:
        counts = export_bi(db, out, include_forecasts=getattr(a, "include_forecasts", False))
    for name, n in counts.items():
        print(f"  {name:28s} {n:>10,} rows")
    print(f"Power BI CSV files written to {out}\nPower Query files (one per table): {out / 'power_query'}\n{SYNTHETIC_DISCLAIMER}")
    return 0


def _forecast_overrides(a) -> dict:
    return {"horizon_days": a.horizon, "max_series": a.max_series, "censor_policy": a.censor_policy, "seed": a.seed,
            "models": getattr(a, "models", None), "n_origins": getattr(a, "origins", None),
            "origin_step_days": getattr(a, "step", None), "min_train_days": getattr(a, "min_train", None)}


def cmd_backtest(a) -> int:
    """Rolling-origin backtest of the forecasting models. Reads the database, writes only NEW rows (never overwrites)."""
    from .features.panel import load_panel
    from .forecasting.backtest import run_backtest
    from .forecasting.config import load_forecast_settings
    from .forecasting.diagnostics import load_latent
    from .forecasting.registry import default_backtest_id, ensure_free, save_backtest
    from .forecasting.report import DISCLAIMER, format_summary, write_report
    st = load_forecast_settings(a.profile, **_forecast_overrides(a))
    if "gbm" in st.models:
        from .forecasting.gbm import _regressor
        _regressor({}, 0)                      # fails fast with the install hint if scikit-learn is missing
    _require_existing_db(a)
    with _db(a) as db:
        run_id = a.run_id or default_backtest_id(a.profile)
        if not a.no_save:
            ensure_free(db, "backtest_run", "backtest_run_id", run_id)
        panel = load_panel(db, st.max_series, st.seed)
        print(f"Series panel: {panel.n_series:,} series x {panel.n_days} days "
              f"({panel.dates[0].date()} to {panel.dates[-1].date()}). Models: {', '.join(st.models)}. "
              f"Horizon {st.horizon_days} days, censoring policy: {st.censor_policy}.")
        latent = load_latent(db, panel) if a.latent_diagnostic else None
        result = run_backtest(panel, st, latent, progress=print)
        saved = False
        if not a.no_save:
            n = save_backtest(db, run_id, result, a.notes or "")
            saved = True
            print(f"Saved {n:,} metric rows as backtest run '{run_id}' (dwh.backtest_run, dwh.forecast_backtest_metric).")
    print("\n" + format_summary(result))
    path = write_report(result, run_id, saved, reports_dir())
    print(f"\nReport: {path}\n{DISCLAIMER}")
    return 0


def cmd_forecast(a) -> int:
    """Forecast `horizon` days after one origin and store them in dwh.model_run / dwh.fact_demand_forecast."""
    import numpy as np
    import pandas as pd
    from .features.panel import load_panel
    from .forecasting import metrics as fm
    from .forecasting.backtest import single_forecast
    from .forecasting.config import load_forecast_settings
    from .forecasting.registry import default_forecast_id, ensure_free, save_forecast
    st = load_forecast_settings(a.profile, **_forecast_overrides(a))
    from .forecasting.config import MODEL_NAMES
    if a.model not in MODEL_NAMES:
        raise DZMIError(f"Unknown model '{a.model}'. Valid: {', '.join(MODEL_NAMES)}")
    if a.model == "gbm":
        from .forecasting.gbm import _regressor
        _regressor({}, 0)
    _require_existing_db(a)
    with _db(a) as db:
        panel = load_panel(db, st.max_series, st.seed)
        latest = panel.n_days - 1 - st.horizon_days
        o = latest if not a.origin else panel.index_of(a.origin)
        if o > latest:
            raise DZMIError(f"The origin must leave {st.horizon_days} loaded days after it (fact_demand_forecast "
                            f"references dim_date): latest possible origin is {panel.dates[latest].date()}.")
        if o + 1 < st.min_train_days:
            raise DZMIError(f"At least {st.min_train_days} days of history are needed before the origin "
                            f"(first possible origin: {panel.dates[st.min_train_days - 1].date()}).")
        run_id = a.run_id or default_forecast_id(a.model, panel.dates[o])
        ensure_free(db, "model_run", "forecast_run_id", run_id)
        pred, actual, cens, days = single_forecast(panel, st, a.model, o)
        n = save_forecast(db, run_id, a.model, panel, o, pred, st,
                          a.notes or "Hold-out forecast: origin inside the loaded data, so actuals exist for comparison.")
    ok = ~np.isnan(actual)
    e = fm.add_derived(pd.DataFrame([fm.error_sums(np.nan_to_num(actual), pred, ok)])).iloc[0]
    print(f"Run '{run_id}': model {a.model}, origin {panel.dates[o].date()}, forecasts for {days[0].date()} to "
          f"{days[-1].date()} ({st.horizon_days} days), {panel.n_series:,} series, {n:,} rows saved.")
    print(f"Against the observed sales of those days: MAE {e['mae']:.3f}, WAPE {e['wape'] * 100:.1f}%, "
          f"bias {e['bias'] * 100:+.1f}% (observed sales are truncated on stockout days).")
    print(SYNTHETIC_DISCLAIMER)
    return 0


def cmd_pipeline(a) -> int:
    for step in (cmd_generate, cmd_validate, cmd_load):
        rc = step(a)
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dzmi", description="DZ Market Intelligence (synthetic data only).")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, db=False):
        sp.add_argument("--profile", default="smoke", help="config/<profile>.yaml (default: smoke)")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--stores", type=int, help="override n_stores")
        sp.add_argument("--products", type=int, help="override n_products")
        sp.add_argument("--start", help="override start_date (YYYY-MM-DD)")
        sp.add_argument("--end", help="override end_date (YYYY-MM-DD)")
        sp.add_argument("--out", help="raw data folder (default data/raw/<profile>)")
        if db:
            sp.add_argument("--backend", choices=["postgres", "sqlite"], default="postgres")
            sp.add_argument("--sqlite-path", help="SQLite base path or ':memory:' (default data/dzmi_<profile>.sqlite)")
            sp.add_argument("--reset", action="store_true", help="DESTRUCTIVE: drop ops/dwh schemas before loading")

    for name, fn, hlp, db in [
        ("generate", cmd_generate, "generate synthetic ops tables to CSV", False),
        ("validate", cmd_validate, "run pre-load data-quality checks and write reports/", False),
        ("init-db", cmd_init_db, "create schemas/tables/indexes/views (idempotent)", True),
        ("load", cmd_load, "validate, load into the database and run SQL validation", True),
        ("validate-db", cmd_validate_db, "run SQL validation checks on an existing database", True),
        ("pipeline", cmd_pipeline, "generate -> validate -> load -> validate-db", True),
    ]:
        sp = sub.add_parser(name, help=hlp)
        common(sp, db)
        sp.set_defaults(func=fn)

    def read_db_args(sp):
        sp.add_argument("--profile", default="smoke", help="config/<profile>.yaml (default: smoke)")
        sp.add_argument("--backend", choices=["postgres", "sqlite"], default="postgres")
        sp.add_argument("--sqlite-path", help="SQLite base path (default data/dzmi_<profile>.sqlite)")

    sp = sub.add_parser("query", help="print or save one of the analytical views (read-only)")
    read_db_args(sp)
    sp.add_argument("--list", action="store_true", help="list the available views and their grain")
    sp.add_argument("--view", help="view name, e.g. v_kpi_network_monthly")
    sp.add_argument("--limit", type=int, default=20, help="rows to print (0 = all, default 20)")
    sp.add_argument("--order-by", help="sort by this column")
    sp.add_argument("--desc", action="store_true", help="sort descending (with --order-by)")
    sp.add_argument("--columns", help="comma-separated columns to show")
    sp.add_argument("--csv", help="save the whole view to this CSV file instead of printing")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("export-bi", help="write the star schema and analytical views as CSV files for Power BI")
    read_db_args(sp)
    sp.add_argument("--out", help="output folder (default data/bi_export/<profile>)")
    sp.add_argument("--include-forecasts", action="store_true",
                    help="also export the forecast and backtest tables (optional, Milestone 3)")
    sp.set_defaults(func=cmd_export_bi)

    def forecast_args(sp):
        read_db_args(sp)
        sp.add_argument("--horizon", type=int, help="forecast horizon in days (default from config/forecast.yaml)")
        sp.add_argument("--max-series", type=int, help="series to use: stratified sample, 0 = all (default from config)")
        sp.add_argument("--censor-policy", choices=["exclude", "keep"], help="treatment of stockout days (default exclude)")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--run-id", help="name of the run (letters, digits, _ and -). Existing ids are never overwritten")
        sp.add_argument("--notes", help="free text stored with the run")

    sp = sub.add_parser("backtest", help="rolling-origin backtest of the forecasting models (needs scikit-learn for gbm)")
    forecast_args(sp)
    sp.add_argument("--models", help="comma-separated: naive,seasonal_naive,moving_average,ses_weekday,croston_sba,gbm")
    sp.add_argument("--origins", type=int, help="number of rolling origins")
    sp.add_argument("--step", type=int, help="days between origins")
    sp.add_argument("--min-train", type=int, help="minimum training days before the first origin")
    sp.add_argument("--latent-diagnostic", action="store_true",
                    help="ALSO score against simulator latent demand (simulation-only diagnostic, never used for training)")
    sp.add_argument("--no-save", action="store_true", help="do not write the metrics to the database")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("forecast", help="forecast the days after one origin and store them in dwh.fact_demand_forecast")
    forecast_args(sp)
    sp.add_argument("--model", default="seasonal_naive", help="model name (default seasonal_naive)")
    sp.add_argument("--origin", help="last training day, YYYY-MM-DD (default: the latest origin that leaves a full horizon)")
    sp.set_defaults(func=cmd_forecast)
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DZMIError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
