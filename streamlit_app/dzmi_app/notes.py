"""Data & Method Notes (Page 6): content builders. No Streamlit here.

Every figure comes from the loaded dataset (manifest, tables, integrity checks). The two text blocks
are the report's own words, verbatim. Nothing about source systems, refresh schedules or lineage is
stated because none of it is in the files.
"""

from __future__ import annotations

import math

import pandas as pd

from . import measures_forecast as mf
from .data_loader import Dataset
from .integrity import CheckResult
from .schema import APP_TABLES, EXPORT_ONLY_TABLES

# --- verbatim from the Power BI page 'Data & Method Notes' (text boxes) ---------------------------
WHAT_THIS_REPORT_IS = [
    ("What this report is",
     "A portfolio demonstration built on SYNTHETIC data from the DZ Market Intelligence simulator. No store, "
     "customer, price, sale or forecast score here describes the real Algerian market."),
    ("How horizon_bucket is defined",
     "At the series x day level it is the forecast day step (1-14 days after the origin). At the store x "
     "category x week level it is the forecast week (1-2). The two levels have different scales and are never "
     "shown on one axis."),
    ("How accuracy is computed",
     "Errors are stored as additive sums per model, basis, level, origin and horizon. MAE, RMSE, WAPE and bias "
     "are always recomputed from the sums (sum of errors / sum of observations). They are never averaged or "
     "added across rows."),
]
KNOWN_LIMITS = [
    "Year-over-year is blank: the data covers 2025-02-01 to 2025-05-01 only. May holds a single day, so "
    "month-over-month is shown only for complete months.",
    "Inventory risk is a snapshot of the latest loaded day and ignores the date slicer.",
    "No stored forecast exists (fact_demand_forecast is empty): this report shows backtest accuracy, not "
    "forecast-versus-actual lines.",
    "Observed sales are truncated on stockout days; the no-stockout basis shows how much that matters.",
    "Four rolling origins on a small sample cannot rank methods in general.",
]

# --- relationships.tmdl (29 many-to-one relationships, all single direction by default) -----------
RELATIONSHIPS = [
    ("fact_sales.date_key", "dim_date.date_key"), ("fact_sales.store_key", "dim_store.store_key"),
    ("fact_sales.product_key", "dim_product.product_key"), ("fact_sales.customer_key", "dim_customer.customer_key"),
    ("fact_sales.promotion_key", "dim_promotion.promotion_key"),
    ("fact_inventory_daily.date_key", "dim_date.date_key"), ("fact_inventory_daily.store_key", "dim_store.store_key"),
    ("fact_inventory_daily.product_key", "dim_product.product_key"),
    ("fact_promotions.date_key", "dim_date.date_key"), ("fact_promotions.store_key", "dim_store.store_key"),
    ("fact_promotions.product_key", "dim_product.product_key"),
    ("fact_promotions.promotion_key", "dim_promotion.promotion_key"),
    ("v_inventory_risk.store_key", "dim_store.store_key"), ("v_inventory_risk.product_key", "dim_product.product_key"),
    ("forecast_backtest_metric.model_name", "dim_model.model_name"),
    ("forecast_backtest_metric.evaluation_basis", "dim_eval_basis.evaluation_basis"),
    ("forecast_backtest_metric.level", "dim_eval_level.level"),
    ("forecast_backtest_metric.horizon_key", "dim_horizon.horizon_key"),
    ("forecast_backtest_metric.origin_date", "dim_origin.origin_date"),
    ("forecast_backtest_metric.backtest_run_id", "backtest_run.backtest_run_id"),
    ("v_forecast_accuracy.model_name", "dim_model.model_name"),
    ("v_forecast_accuracy.evaluation_basis", "dim_eval_basis.evaluation_basis"),
    ("v_forecast_accuracy.level", "dim_eval_level.level"),
    ("v_forecast_accuracy.backtest_run_id", "backtest_run.backtest_run_id"),
    ("v_forecast_accuracy_by_horizon.model_name", "dim_model.model_name"),
    ("v_forecast_accuracy_by_horizon.evaluation_basis", "dim_eval_basis.evaluation_basis"),
    ("v_forecast_accuracy_by_horizon.level", "dim_eval_level.level"),
    ("v_forecast_accuracy_by_horizon.horizon_key", "dim_horizon.horizon_key"),
    ("v_forecast_accuracy_by_horizon.backtest_run_id", "backtest_run.backtest_run_id"),
]

# --- definitions, as defined in _Measures.tmdl -------------------------------------------------------
DEFINITIONS = [
    ("Net Sales / Gross Profit / Units Sold", "Sums of fact_sales net_sales, gross_profit and quantity."),
    ("Gross Margin %", "Gross Profit / Net Sales (a ratio of sums, never an average of ratios)."),
    ("Average Basket Value", "Net Sales / distinct transaction_id."),
    ("Net Sales MoM %", "Month versus the previous month; shown only when both months are complete (is_complete_month)."),
    ("Stockout Rate", "SKU-days with closing stock = 0 / SKU-days (fact_inventory_daily)."),
    ("Availability Rate", "1 - Stockout Rate in the same filter context."),
    ("Closing Stock (last day)", "Closing stock on the last date of the date range only; never summed across dates."),
    ("Days of Cover (overall)", "Closing stock on the last date / average daily units sold over the 14 days ending that date."),
    ("Days of Cover (median SKU)", "Median of the SKU-level days of cover in the latest-day snapshot (v_inventory_risk)."),
    ("Risk tiers", "Stockout (zero stock), Critical (< 3 days of cover), Low (< 7), Healthy, Overstock (> 60), No recent sales."),
    ("MAE / RMSE / WAPE / Bias", "Recomputed from additive sums: MAE = abs. error / observations; RMSE = sqrt(sq. error / observations); "
                                "WAPE = abs. error / actual; Bias = (predicted - actual) / actual (positive = over-forecast)."),
    ("Skill vs Benchmark", "1 - MAE / MAE of the seasonal_naive model."),
]

# --- intentional differences from the Power BI report -------------------------------------------------
DEVIATIONS = [
    ("All pages", "Every filter starts at All and each page keeps its own filters; the date range is a range picker."),
    ("Executive Overview", "The risk-tier chart shows the real number of SKUs per tier; the original drew the same total on every bar."),
    ("Sales Intelligence", "The product grid is sortable and opens in the saved order (category A-Z); matrix header reads 'Month'."),
    ("Inventory Intelligence", "Two days-of-cover figures (overall and median SKU); the risk-tier filter applies to the SKU list only; "
                              "availability uses the same filters as the stockout rate; the category x store table is a heat map."),
    ("Forecasting Performance", "Benchmark MAE ignores the model filter (intended meaning); the literal DAX may differ and has not been "
                               "executed in Power BI Desktop. The best-model matrix is flat (no subtotal / grand-total rows)."),
    ("Forecast Accuracy by Horizon", "Grand-total rows and columns of the matrix are not reproduced: at total level the measures would pool "
                                    "different models."),
    ("Data & Method Notes", "Adds dataset coverage, integrity checks, relationships, definitions and this register."),
]


def dataset_files(ds: Dataset) -> pd.DataFrame:
    """One row per exported table (manifest order), flagging the empty forecast tables."""
    rows = []
    for name, n in dict(ds.manifest["rows"]).items():
        used = "Loaded by the dashboard" if name in APP_TABLES else "Exported, not loaded by the Power BI model"
        if name in EXPORT_ONLY_TABLES:
            used = "Exported, not loaded by the Power BI model"
        rows.append({"table": name, "rows": int(n), "status": "EMPTY (0 rows)" if n == 0 else "", "role": used})
    return pd.DataFrame(rows)


def coverage(ds: Dataset) -> dict:
    dd, inv, run = ds["dim_date"], ds["fact_inventory_daily"], ds["backtest_run"]
    sales = ds["fact_sales"]
    return {
        "first_date": dd["full_date"].min().date(), "last_date": dd["full_date"].max().date(), "days": len(dd),
        "stores": len(ds["dim_store"]), "products": len(ds["dim_product"]),
        "categories": int(ds["dim_product"]["category_name"].nunique()), "customers": len(ds["dim_customer"]),
        "sales_lines": len(sales), "transactions": int(sales["transaction_id"].nunique()),
        "inventory_sku_days": len(inv), "tracked_skus": len(ds["v_inventory_risk"]),
        "snapshot_date": ds["v_inventory_risk"]["snapshot_date"].max().date(),
        "forecast_rows": len(ds["fact_demand_forecast"]), "model_run_rows": len(ds["model_run"]),
        "backtest_run_id": run["backtest_run_id"].iloc[0], "origins": int(ds["forecast_backtest_metric"]["origin_date"].nunique()),
        "horizon_days": int(run["horizon_days"].iloc[0]), "origin_step_days": int(run["origin_step_days"].iloc[0]),
        "min_train_days": int(run["min_train_days"].iloc[0]), "n_series": int(run["n_series"].iloc[0]),
        "censor_policy": run["censor_policy"].iloc[0], "models": str(run["models"].iloc[0]).split(","),
        "synthetic": bool(ds.manifest.get("synthetic")),
    }


def backtest_vs_forecast_text(c: dict) -> list[str]:
    """What the dataset can and cannot show, stated from the loaded tables."""
    return [
        f"Backtest accuracy compares predictions made at {c['origins']} past forecast origins with what was observed in the "
        f"{c['horizon_days']} days after each origin. The origins are {c['origin_step_days']} days apart, so the evaluated "
        f"windows overlap.",
        f"No forecast of the future is stored: fact_demand_forecast has {c['forecast_rows']} rows and model_run has "
        f"{c['model_run_rows']} rows. The dashboard therefore shows no future forecast and no forecast-versus-actual line.",
        f"Stockout days are handled with the policy '{c['censor_policy']}' in the run metadata; the 'observed' basis keeps "
        f"them and 'observed_uncensored' excludes them.",
    ]


def reconciliation(ds: Dataset) -> pd.DataFrame:
    """MAE recomputed from the additive sums versus v_forecast_accuracy.mae (measures 'MAE Check ...')."""
    dims, metric, view = ds.forecast_dims, ds["forecast_backtest_metric"], ds["v_forecast_accuracy"]
    rows = []
    for m in dims.model.itertuples():
        for b in dims.basis.itertuples():
            for lv in dims.level.itertuples():
                s = mf.sums_of(metric[(metric["model_name"] == m.model_name) & (metric["evaluation_basis"] == b.evaluation_basis)
                                      & (metric["level"] == lv.level)])
                v = view[(view["model_name"] == m.model_name) & (view["evaluation_basis"] == b.evaluation_basis)
                         & (view["level"] == lv.level)]
                sql = float(v["mae"].iloc[0]) if len(v) == 1 and pd.notna(v["mae"].iloc[0]) else None
                calc = mf.mae(s)
                diff = None if calc is None or sql is None else calc - sql
                status = None if diff is None else ("OK" if abs(diff) < 0.000001 else "CHECK")
                if calc is None and sql is None:
                    continue                                    # the combination does not exist
                rows.append({"model_label": m.model_label, "basis_label": b.basis_label, "level_label": lv.level_label,
                             "forecast_mae": calc, "sql_view_mae": sql, "difference": diff, "status": status})
    return pd.DataFrame(rows)


def integrity_table(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([{"check": r.check_id, "description": r.name,
                          "kind": "blocking" if r.severity == "error" else "information",
                          "result": "passed" if r.passed else "FAILED"} for r in results])


def relationships_table() -> pd.DataFrame:
    return pd.DataFrame(RELATIONSHIPS, columns=["from (many)", "to (one)"])


def finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
