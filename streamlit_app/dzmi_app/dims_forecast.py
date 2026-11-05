"""Forecast evaluation dimensions, rebuilt in pandas.

In the Power BI semantic model these five dimensions are *not* separate CSV files: they are
Power Query tables derived from ``forecast_backtest_metric.csv`` (Table.Distinct) and joined to
hard-coded label / order lookups. This module reproduces those queries exactly:

* dim_model        -> label, family, order, is_benchmark
* dim_eval_basis   -> label, order, description
* dim_eval_level   -> label, order, horizon unit
* dim_horizon      -> key "level|bucket", label "<unit> <bucket>", sort = level_order*100 + bucket
* dim_origin       -> ISO label and 1-based sequence, ordered by date

Unknown keys fall back exactly as in the M code (label = key, family "Other", order 99,
not a benchmark, unit "Step", empty description).

Only the order of rows differs from Power Query (which keeps first-appearance order and lets
the report sort by the *_order column): here rows are returned already sorted by that column.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FALLBACK_ORDER = 99

# key -> (label, family, order, is_benchmark)
MODEL_META: dict[str, tuple[str, str, int, bool]] = {
    "naive": ("Naive", "Baseline", 1, False),
    "seasonal_naive": ("Seasonal naive (benchmark)", "Baseline", 2, True),
    "moving_average": ("Moving average", "Baseline", 3, False),
    "ses_weekday": ("Exp. smoothing + weekday", "Statistical", 4, False),
    "croston_sba": ("Croston-SBA", "Intermittent demand", 5, False),
    "gbm": ("Gradient boosting", "Machine learning", 6, False),
}

# key -> (label, order, description)
BASIS_META: dict[str, tuple[str, int, str]] = {
    "observed": (
        "Observed sales", 1,
        "Observed units sold, every day: what a retailer can measure",
    ),
    "observed_uncensored": (
        "Observed, no-stockout days", 2,
        "Observed sales without days that ended with zero stock",
    ),
    "latent_simulation_only": (
        "Latent demand (simulation only)", 3,
        "Simulator ground truth: unobservable in real life",
    ),
}

# key -> (label, order, horizon unit)
LEVEL_META: dict[str, tuple[str, int, str]] = {
    "series_day": ("Series x day", 1, "Day"),
    "store_category_week": ("Store x category x week", 2, "Week"),
}

MODEL_COLUMNS = ["model_name", "model_label", "model_family", "model_order", "is_benchmark"]
BASIS_COLUMNS = ["evaluation_basis", "basis_label", "basis_order", "basis_description"]
LEVEL_COLUMNS = ["level", "level_label", "level_order", "horizon_unit"]
HORIZON_COLUMNS = ["horizon_key", "level", "horizon_bucket", "horizon_label", "horizon_sort"]
ORIGIN_COLUMNS = ["origin_date", "origin_label", "origin_seq"]


@dataclass(frozen=True)
class ForecastDims:
    model: pd.DataFrame
    basis: pd.DataFrame
    level: pd.DataFrame
    horizon: pd.DataFrame
    origin: pd.DataFrame


def make_horizon_key(level: pd.Series, bucket: pd.Series) -> pd.Series:
    """``level|bucket`` - the key shared by dim_horizon and the metric tables."""
    return level.astype(str) + "|" + bucket.astype(str)


def add_horizon_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the Power Query ``horizon_key`` column added."""
    out = df.copy()
    out["horizon_key"] = make_horizon_key(out["level"], out["horizon_bucket"])
    return out


def _distinct(series: pd.Series) -> list:
    """Distinct non-null values in first-appearance order (Table.Distinct)."""
    return list(pd.unique(series.dropna()))


def _frame(rows: list[tuple], columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns).astype(dtypes)


def build_dim_model(metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in _distinct(metric["model_name"]):
        label, family, order, bench = MODEL_META.get(key, (key, "Other", FALLBACK_ORDER, False))
        rows.append((key, label, family, order, bench))
    df = _frame(rows, MODEL_COLUMNS, {"model_order": "int64", "is_benchmark": "bool"})
    return df.sort_values(["model_order", "model_name"], kind="stable").reset_index(drop=True)


def build_dim_eval_basis(metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in _distinct(metric["evaluation_basis"]):
        label, order, desc = BASIS_META.get(key, (key, FALLBACK_ORDER, ""))
        rows.append((key, label, order, desc))
    df = _frame(rows, BASIS_COLUMNS, {"basis_order": "int64"})
    return df.sort_values(["basis_order", "evaluation_basis"], kind="stable").reset_index(drop=True)


def build_dim_eval_level(metric: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in _distinct(metric["level"]):
        label, order, unit = LEVEL_META.get(key, (key, FALLBACK_ORDER, "Step"))
        rows.append((key, label, order, unit))
    df = _frame(rows, LEVEL_COLUMNS, {"level_order": "int64"})
    return df.sort_values(["level_order", "level"], kind="stable").reset_index(drop=True)


def build_dim_horizon(metric: pd.DataFrame) -> pd.DataFrame:
    pairs = metric[["level", "horizon_bucket"]].drop_duplicates()
    rows = []
    for level, bucket in pairs.itertuples(index=False, name=None):
        _, order, unit = LEVEL_META.get(level, (level, FALLBACK_ORDER, "Step"))
        bucket = int(bucket)
        rows.append((f"{level}|{bucket}", level, bucket, f"{unit} {bucket}", order * 100 + bucket))
    df = _frame(rows, HORIZON_COLUMNS, {"horizon_bucket": "int64", "horizon_sort": "int64"})
    return df.sort_values("horizon_sort", kind="stable").reset_index(drop=True)


def build_dim_origin(metric: pd.DataFrame) -> pd.DataFrame:
    origin = (
        metric["origin_date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    )
    df = pd.DataFrame({"origin_date": origin})
    df["origin_label"] = df["origin_date"].dt.strftime("%Y-%m-%d").astype(str)
    df["origin_seq"] = pd.Series(range(1, len(df) + 1), dtype="int64")
    return df[ORIGIN_COLUMNS]


def build_forecast_dims(metric: pd.DataFrame) -> ForecastDims:
    """Rebuild all five dimensions from the ``forecast_backtest_metric`` table."""
    return ForecastDims(
        model=build_dim_model(metric),
        basis=build_dim_eval_basis(metric),
        level=build_dim_eval_level(metric),
        horizon=build_dim_horizon(metric),
        origin=build_dim_origin(metric),
    )
