"""Read-only CSV loading with in-process caching.

* Every file is read with pandas.read_csv(..., encoding="utf-8") - nothing is ever written.
* Headers must match the schema exactly (names and order) and dtypes are explicit, so a
  malformed export fails loudly instead of being coerced silently.
* Only the empty string is treated as null (keep_default_na=False), so text such as "NA"
  inside a product name can never be mistaken for a missing value.
* Power Query's added columns are reproduced here: dim_date.calendar_period / week_part,
  dim_product.perishability, and horizon_key on the two tables that carry it.
* Caching uses functools.lru_cache so this module has no Streamlit dependency and can be
  tested on its own. The UI layer (Milestone C) can wrap load_dataset in st.cache_resource.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from . import paths
from .constants import MANIFEST_FILE
from .dims_forecast import ForecastDims, add_horizon_key, build_forecast_dims
from .errors import DataIntegrityError
from .schema import APP_TABLES, SCHEMAS, ColumnSpec

_READ_DTYPES = {"int": "int64", "Int": "Int64", "float": "float64", "bool": "bool", "str": str}
_DATE_FORMATS = {"date": "%Y-%m-%d", "datetime": "%Y-%m-%d %H:%M:%S"}


@dataclass(frozen=True)
class Dataset:
    """Immutable bundle of every loaded table plus the derived forecast dimensions.

    The mapping itself is read-only; the DataFrames are shared through the cache, so callers
    must treat them as read-only (copy before modifying).
    """

    data_dir: Path
    manifest: Mapping[str, object]
    tables: Mapping[str, pd.DataFrame]
    forecast_dims: ForecastDims

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def row_counts(self) -> dict[str, int]:
        return {name: len(df) for name, df in self.tables.items()}


def _resolve_dir(data_dir: Path | str | None) -> Path:
    return (paths.DATA_DIR if data_dir is None else Path(data_dir)).resolve()


def _csv_path(name: str, data_dir: Path) -> Path:
    path = data_dir / f"{name}.csv"
    if not path.is_file():
        raise DataIntegrityError(f"Missing data file: {paths.display_path(path)}")
    return path


def read_manifest(data_dir: Path | str | None = None) -> dict:
    path = _resolve_dir(data_dir) / MANIFEST_FILE
    if not path.is_file():
        raise DataIntegrityError(f"Missing manifest: {paths.display_path(path)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"Manifest is not valid JSON: {exc}") from exc


def count_csv_rows(name: str, data_dir: Path | str | None = None) -> int:
    """Row count of any exported CSV without needing its schema (header excluded)."""
    path = _csv_path(name, _resolve_dir(data_dir))
    return len(pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False, encoding="utf-8"))


def _postprocess(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the columns added by the Power Query steps of the semantic model."""
    if name == "dim_date":
        df["calendar_period"] = np.select(
            [df["is_eid_al_fitr"] == 1, df["is_eid_al_adha"] == 1, df["is_ramadan"] == 1],
            ["Eid al-Fitr", "Eid al-Adha", "Ramadan"],
            default="Regular days",
        )
        df["week_part"] = np.where(df["is_weekend"] == 1, "Weekend (Fri-Sat)", "Weekday")
    elif name == "dim_product":
        df["perishability"] = np.where(df["is_perishable"], "Perishable", "Non-perishable")
    elif name in ("forecast_backtest_metric", "v_forecast_accuracy_by_horizon"):
        df = add_horizon_key(df)
    return df


def _validate_nulls(name: str, df: pd.DataFrame, columns: tuple[ColumnSpec, ...]) -> None:
    for col in columns:
        if not col.nullable and df[col.name].isna().any():
            n = int(df[col.name].isna().sum())
            raise DataIntegrityError(f"{name}.{col.name}: {n} unexpected null value(s)")


def read_table(name: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Read one CSV with the schema of the semantic model. Raises DataIntegrityError."""
    if name not in SCHEMAS:
        raise DataIntegrityError(f"Unknown table: {name}")
    columns = SCHEMAS[name]
    path = _csv_path(name, _resolve_dir(data_dir))
    where = paths.display_path(path)

    try:
        header = list(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataIntegrityError(f"{where}: unreadable header ({exc})") from exc
    expected = [c.name for c in columns]
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        raise DataIntegrityError(
            f"{where}: header differs from schema (missing={missing}, unexpected={extra}, "
            f"order_ok={sorted(header) == sorted(expected)})"
        )

    dtype = {c.name: _READ_DTYPES[c.kind] for c in columns if c.kind in _READ_DTYPES}
    try:
        df = pd.read_csv(
            path, encoding="utf-8", dtype=dtype, keep_default_na=False, na_values=[""]
        )
        for col in columns:
            fmt = _DATE_FORMATS.get(col.kind)
            if fmt:
                df[col.name] = pd.to_datetime(df[col.name], format=fmt)
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        raise DataIntegrityError(f"{where}: cannot parse values as declared ({exc})") from exc

    _validate_nulls(name, df, columns)
    return _postprocess(name, df)


@lru_cache(maxsize=4)
def _load_cached(resolved_dir: str) -> Dataset:
    data_dir = Path(resolved_dir)
    if not data_dir.is_dir():
        raise DataIntegrityError(f"Data folder not found: {paths.display_path(data_dir)}")
    tables = {name: read_table(name, data_dir) for name in APP_TABLES}
    return Dataset(
        data_dir=data_dir,
        manifest=MappingProxyType(read_manifest(data_dir)),
        tables=MappingProxyType(tables),
        forecast_dims=build_forecast_dims(tables["forecast_backtest_metric"]),
    )


def load_dataset(data_dir: Path | str | None = None) -> Dataset:
    """Load (once per process) every table of the dataset. Cached by resolved folder."""
    return _load_cached(str(_resolve_dir(data_dir)))


def clear_cache() -> None:
    _load_cached.cache_clear()
