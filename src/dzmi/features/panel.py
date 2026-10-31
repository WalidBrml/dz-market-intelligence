"""The series panel: store x product x day arrays that forecasting models read.

LEAKAGE BY CONSTRUCTION. A model never receives the full panel. It receives
  * `panel.slice_to(i)`        history up to and including day i (nothing later exists in the object), and
  * `panel.future_known(i, h)`  for days i+1..i+h ONLY what is known in advance: calendar flags and planned
                                promotions. It has no units, stock or stockout fields at all.
`ops.sim_latent_demand` is never read here (see forecasting/diagnostics.py, the only module allowed to read it,
for evaluation labelled "simulation only").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..errors import DZMIError
from ..ingestion.db import Database

CALENDAR_COLUMNS = ["dow", "is_weekend", "month", "day_of_month", "is_ramadan", "ramadan_day", "is_eid_al_fitr",
                    "is_eid_al_adha", "is_pre_eid_al_fitr", "is_pre_eid_al_adha"]
SERIES_COLUMNS = ["store_key", "product_key", "category_code", "store_format_code", "floor_area_sqm", "list_price",
                  "is_perishable"]


@dataclass(frozen=True)
class Future:
    """What is known in advance about the days after an origin. Deliberately has no sales/stock fields."""
    dates: pd.DatetimeIndex
    calendar: pd.DataFrame          # one row per future day, CALENDAR_COLUMNS
    promo_discount: np.ndarray      # (n_series, h) planned discount percent, 0 = no promotion


@dataclass(frozen=True)
class Panel:
    series: pd.DataFrame            # one row per series: SERIES_COLUMNS (+ names for reports)
    dates: pd.DatetimeIndex         # T consecutive days
    units: np.ndarray               # (n, T) observed units sold, NaN = not observed / not assorted
    censored: np.ndarray            # (n, T) bool, True = end-of-day stock was 0 (sales may be censored)
    closing_stock: np.ndarray       # (n, T)
    on_order: np.ndarray            # (n, T)
    promo_discount: np.ndarray      # (n, T) discount percent of a promotion running that day, 0 if none
    calendar: pd.DataFrame          # T rows, CALENDAR_COLUMNS

    @property
    def n_series(self) -> int:
        return self.units.shape[0]

    @property
    def n_days(self) -> int:
        return self.units.shape[1]

    def slice_to(self, i: int) -> "Panel":
        """History up to and including day index i. Later days do not exist in the returned object."""
        if not 0 <= i < self.n_days:
            raise DZMIError(f"Origin index {i} is outside the panel (0..{self.n_days - 1}).")
        k = i + 1
        return Panel(self.series, self.dates[:k], self.units[:, :k], self.censored[:, :k], self.closing_stock[:, :k],
                     self.on_order[:, :k], self.promo_discount[:, :k], self.calendar.iloc[:k])

    def future_known(self, i: int, horizon: int) -> Future:
        """Calendar and planned promotions for days i+1..i+horizon (no sales, no stock)."""
        if i + horizon >= self.n_days:
            raise DZMIError("The forecast window must lie inside the loaded calendar (fact_demand_forecast "
                            "references dim_date).")
        sl = slice(i + 1, i + 1 + horizon)
        return Future(self.dates[sl], self.calendar.iloc[sl].reset_index(drop=True), self.promo_discount[:, sl])

    def index_of(self, day) -> int:
        day = pd.Timestamp(day).normalize()
        pos = self.dates.get_indexer([day])[0]
        if pos < 0:
            raise DZMIError(f"{day.date()} is not in the loaded calendar ({self.dates[0].date()} to {self.dates[-1].date()}).")
        return int(pos)

    @classmethod
    def from_arrays(cls, units, start="2025-01-06", censored=None, closing_stock=None, on_order=None,
                    promo_discount=None, series=None, calendar=None) -> "Panel":
        """Build a panel from plain arrays (hand-made series in tests, notebooks). Default calendar has no religious flags."""
        units = np.asarray(units, dtype=float)
        n, t = units.shape
        dates = pd.date_range(start, periods=t, freq="D")
        z = lambda a, fill: np.full((n, t), fill, dtype=float) if a is None else np.asarray(a, dtype=float)
        if series is None:
            series = pd.DataFrame({"store_key": 1, "product_key": np.arange(1, n + 1), "category_code": 0,
                                   "store_format_code": 0, "floor_area_sqm": 1000.0, "list_price": 100.0,
                                   "is_perishable": 0})
        return cls(series.reset_index(drop=True), dates, units,
                   np.zeros((n, t), dtype=bool) if censored is None else np.asarray(censored, dtype=bool),
                   z(closing_stock, 10.0), z(on_order, 0.0), z(promo_discount, 0.0),
                   calendar_from_dates(dates) if calendar is None else calendar)


def calendar_from_dates(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar columns derived from the dates alone (weekday, month...). Religious flags are 0."""
    cal = pd.DataFrame({"dow": dates.dayofweek.to_numpy(), "month": dates.month.to_numpy(),
                        "day_of_month": dates.day.to_numpy()})
    cal["is_weekend"] = cal["dow"].isin([4, 5]).astype(int)       # Friday and Saturday (Algerian weekend)
    for c in ("is_ramadan", "ramadan_day", "is_eid_al_fitr", "is_eid_al_adha", "is_pre_eid_al_fitr", "is_pre_eid_al_adha"):
        cal[c] = 0
    return cal[CALENDAR_COLUMNS]


# ------------------------------------------------------------------------------------------------ loading
def select_series(db: Database, max_series: int = 0, seed: int = 42) -> pd.DataFrame:
    """Series (store_key, product_key) that have inventory rows, optionally a volume-stratified sample.

    The sample is deterministic for a given seed: series are ranked by total units sold and split into up to
    10 volume strata, and the same share is drawn from each stratum (so rare and best-selling items are both present).
    """
    s = db.query_df("SELECT store_key, product_key, SUM(units_sold) AS total_units FROM dwh.fact_inventory_daily "
                    "GROUP BY store_key, product_key ORDER BY store_key, product_key")
    if s.empty:
        raise DZMIError("No inventory rows found: load data first (dzmi load).")
    if max_series <= 0 or max_series >= len(s):
        return s.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = np.argsort(s["total_units"].to_numpy(), kind="mergesort")
    strata = np.array_split(order, min(10, max_series))
    quota = np.full(len(strata), max_series // len(strata))
    quota[: max_series % len(strata)] += 1
    chosen = np.concatenate([rng.choice(idx, size=min(q, len(idx)), replace=False) for idx, q in zip(strata, quota)])
    return s.iloc[np.sort(chosen)].reset_index(drop=True)


def load_panel(db: Database, max_series: int = 0, seed: int = 42) -> Panel:
    """Read the panel from the warehouse (dim_date, dim_store, dim_product, fact_inventory_daily, fact_promotions)."""
    cal = db.query_df(
        "SELECT date_key, full_date, month, day_of_month, ramadan_day, "
        "CASE WHEN is_weekend THEN 1 ELSE 0 END AS is_weekend, CASE WHEN is_ramadan THEN 1 ELSE 0 END AS is_ramadan, "
        "CASE WHEN is_eid_al_fitr THEN 1 ELSE 0 END AS is_eid_al_fitr, CASE WHEN is_eid_al_adha THEN 1 ELSE 0 END AS is_eid_al_adha, "
        "CASE WHEN is_pre_eid_al_fitr THEN 1 ELSE 0 END AS is_pre_eid_al_fitr, "
        "CASE WHEN is_pre_eid_al_adha THEN 1 ELSE 0 END AS is_pre_eid_al_adha "
        "FROM dwh.dim_date ORDER BY date_key")
    if cal.empty:
        raise DZMIError("dwh.dim_date is empty: load data first (dzmi load).")
    dates = pd.DatetimeIndex(pd.to_datetime(cal["full_date"]))
    if not (np.diff(dates.to_numpy()).astype("timedelta64[D]") == np.timedelta64(1, "D")).all():
        raise DZMIError("dim_date has gaps: forecasting needs consecutive days.")
    calendar = calendar_from_dates(dates)
    for c in ("is_weekend", "is_ramadan", "is_eid_al_fitr", "is_eid_al_adha", "is_pre_eid_al_fitr", "is_pre_eid_al_adha"):
        calendar[c] = cal[c].to_numpy().astype(int)
    calendar["ramadan_day"] = pd.to_numeric(cal["ramadan_day"]).fillna(0).to_numpy().astype(int)

    chosen = select_series(db, max_series, seed)
    keys = chosen["store_key"].to_numpy(dtype=np.int64) * 1_000_000 + chosen["product_key"].to_numpy(dtype=np.int64)
    key_index = pd.Index(keys)
    n, t = len(chosen), len(dates)
    units = np.full((n, t), np.nan)
    censored = np.zeros((n, t), dtype=bool)
    closing = np.full((n, t), np.nan)
    on_order = np.full((n, t), np.nan)
    date_keys = cal["date_key"].to_numpy(dtype=np.int64)
    sql = ("SELECT store_key, product_key, date_key, units_sold, closing_stock, on_order_qty, "
           "CASE WHEN stockout_flag THEN 1 ELSE 0 END AS stockout FROM dwh.fact_inventory_daily")
    for chunk in db.iter_query_df(sql, 500_000):
        k = chunk["store_key"].to_numpy(dtype=np.int64) * 1_000_000 + chunk["product_key"].to_numpy(dtype=np.int64)
        row = key_index.get_indexer(k)
        keep = row >= 0
        if not keep.any():
            continue
        r, c = row[keep], np.searchsorted(date_keys, chunk["date_key"].to_numpy(dtype=np.int64)[keep])
        units[r, c] = chunk["units_sold"].to_numpy(dtype=float)[keep]
        closing[r, c] = chunk["closing_stock"].to_numpy(dtype=float)[keep]
        on_order[r, c] = chunk["on_order_qty"].to_numpy(dtype=float)[keep]
        censored[r, c] = chunk["stockout"].to_numpy()[keep] == 1
    promo = np.zeros((n, t))
    pr = db.query_df("SELECT store_key, product_key, date_key, discount_pct FROM dwh.fact_promotions")
    if not pr.empty:
        k = pr["store_key"].to_numpy(dtype=np.int64) * 1_000_000 + pr["product_key"].to_numpy(dtype=np.int64)
        row = key_index.get_indexer(k)
        keep = row >= 0
        promo[row[keep], np.searchsorted(date_keys, pr["date_key"].to_numpy(dtype=np.int64)[keep])] = \
            pd.to_numeric(pr["discount_pct"]).to_numpy(dtype=float)[keep]

    attrs = db.query_df(
        "SELECT p.product_key, p.category_name, p.current_list_price AS list_price, "
        "CASE WHEN p.is_perishable THEN 1 ELSE 0 END AS is_perishable FROM dwh.dim_product p")
    stores = db.query_df("SELECT store_key, store_format, floor_area_sqm FROM dwh.dim_store")
    cat_codes = {c: i for i, c in enumerate(sorted(attrs["category_name"].unique()))}
    fmt_codes = {f: i for i, f in enumerate(sorted(stores["store_format"].unique()))}
    ser = chosen[["store_key", "product_key"]].merge(attrs, on="product_key", how="left").merge(stores, on="store_key", how="left")
    ser["category_code"] = ser["category_name"].map(cat_codes).astype(int)
    ser["store_format_code"] = ser["store_format"].map(fmt_codes).astype(int)
    ser["list_price"] = pd.to_numeric(ser["list_price"]).astype(float)
    ser["floor_area_sqm"] = pd.to_numeric(ser["floor_area_sqm"]).astype(float)
    ser = ser[SERIES_COLUMNS + ["category_name", "store_format"]]
    return Panel(ser, dates, units, censored, closing, on_order, promo, calendar)
