"""Export the star schema and the analytical views to CSV files that Power BI can import.

Why: Power BI has a native PostgreSQL connector but no native SQLite connector. This export lets the
SQLite development database feed the same Power BI model (see powerbi/README.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import SYNTHETIC_DISCLAIMER
from ..ingestion.db import Database
from .powerquery import write_power_query_files
from .views import ANALYTIC_VIEWS

# (output name, SQL, boolean columns to write as True/False). dim_date comes from the Power BI date view,
# a superset of dwh.dim_date, and is written under the name dim_date so the model and the DAX stay simple.
BI_TABLES: list[tuple[str, str, set[str]]] = [
    ("dim_date", "SELECT * FROM dwh.v_dim_date_bi ORDER BY date_key", set()),
    ("dim_store", "SELECT * FROM dwh.dim_store ORDER BY store_key", set()),
    ("dim_product", "SELECT * FROM dwh.dim_product ORDER BY product_key", {"is_perishable"}),
    ("dim_customer", "SELECT * FROM dwh.dim_customer ORDER BY customer_key", set()),
    ("dim_promotion", "SELECT * FROM dwh.dim_promotion ORDER BY promotion_key", set()),
    ("fact_sales", "SELECT * FROM dwh.fact_sales ORDER BY sales_line_id", set()),
    ("fact_inventory_daily",
     "SELECT * FROM dwh.fact_inventory_daily ORDER BY store_key, product_key, date_key", {"stockout_flag"}),
    ("fact_promotions", "SELECT * FROM dwh.fact_promotions ORDER BY store_key, product_key, date_key", set()),
]
# Views are exported under their own names (not part of the star schema: used for tables and checks).
BI_VIEWS: list[tuple[str, str, set[str]]] = [
    (name, f"SELECT * FROM dwh.{name} ORDER BY {', '.join(order)}", set())
    for name, (_, order) in ANALYTIC_VIEWS.items() if name != "v_dim_date_bi"
]
# Optional (dzmi export-bi --include-forecasts): Milestone 3 tables and views. No Power Query files are generated for them.
BI_FORECAST_TABLES: list[tuple[str, str, set[str]]] = [
    ("model_run", "SELECT * FROM dwh.model_run ORDER BY forecast_run_id", set()),
    ("fact_demand_forecast", "SELECT * FROM dwh.fact_demand_forecast "
                             "ORDER BY forecast_run_id, store_key, product_key, date_key", set()),
    ("backtest_run", "SELECT * FROM dwh.backtest_run ORDER BY backtest_run_id", set()),
    ("forecast_backtest_metric", "SELECT * FROM dwh.forecast_backtest_metric "
                                 "ORDER BY backtest_run_id, model_name, evaluation_basis, level, origin_date, horizon_bucket", set()),
    ("v_forecast_accuracy", "SELECT * FROM dwh.v_forecast_accuracy "
                            "ORDER BY backtest_run_id, evaluation_basis, level, model_name", set()),
    ("v_forecast_accuracy_by_horizon", "SELECT * FROM dwh.v_forecast_accuracy_by_horizon "
                                       "ORDER BY backtest_run_id, evaluation_basis, level, model_name, horizon_bucket", set()),
]
NULLABLE_INT_COLUMNS = {"ramadan_day"}


def _normalise(df: pd.DataFrame, bool_cols: set[str]) -> pd.DataFrame:
    """Make the CSV identical for SQLite and PostgreSQL: integer keys without '.0', booleans as True/False."""
    df = df.copy()
    for c in df.columns:
        if c.endswith("_key") or c in NULLABLE_INT_COLUMNS:
            df[c] = pd.to_numeric(df[c]).astype("Int64")   # nullable keys (customer, promotion) stay integers
        elif c in bool_cols:
            df[c] = df[c].map(lambda v: pd.NA if pd.isna(v) else bool(v)).astype("boolean")
    return df


def export_bi(db: Database, out_dir: Path, chunksize: int = 200_000, include_forecasts: bool = False) -> dict[str, int]:
    """Write one CSV per table/view into out_dir (overwriting) plus bi_export_manifest.json. Returns row counts.

    include_forecasts=True also exports the Milestone 3 forecast and backtest tables/views (default: unchanged export).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, sql, bool_cols in BI_TABLES + BI_VIEWS + (BI_FORECAST_TABLES if include_forecasts else []):
        path = out_dir / f"{name}.csv"
        n = 0
        with path.open("w", encoding="utf-8", newline="") as fh:
            header = True
            for chunk in db.iter_query_df(sql, chunksize):
                _normalise(chunk, bool_cols).to_csv(fh, index=False, header=header, lineterminator="\n")
                header = False
                n += len(chunk)
            if header:   # empty result: still write the column names
                _normalise(db.query_df(sql + " LIMIT 0"), bool_cols).to_csv(fh, index=False, lineterminator="\n")
        counts[name] = n
    write_power_query_files(out_dir / "power_query", str(out_dir.resolve()))
    manifest = {"synthetic": True, "disclaimer": SYNTHETIC_DISCLAIMER, "source": db.description, "rows": counts,
                "note": "dim_date.csv is dwh.v_dim_date_bi (dim_date plus time-intelligence helper columns). "
                        "power_query/*.pq are Power BI queries with explicit column types for the star-schema CSV files "
                        "(none are generated for the optional forecast files)."}
    (out_dir / "bi_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts
