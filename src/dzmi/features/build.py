"""Leakage-safe features for demand forecasting (used by the gradient boosting model).

Every feature for an origin day `o` is computed from columns 0..o of the panel only, plus calendar flags and planned
promotions of the TARGET day, which are known in advance. Forbidden: sales, stock or stockout of days after `o`, and
`ops.sim_latent_demand`. tests/test_features.py proves it by overwriting the future with random numbers and checking
that the features do not change.

Censoring: with censor_policy "exclude", days whose end-of-day stock was 0 (sales possibly truncated) are treated as
missing when computing the demand statistics below (they still count in `stockout_share_28`, which describes them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel
from .stats import last_k, nanmean_last, nanmean_rows

FEATURE_COLUMNS = [
    # static attributes of the series
    "category_code", "store_format_code", "floor_area_sqm", "list_price", "is_perishable",
    # history known at the origin
    "mean_7", "mean_14", "mean_28", "mean_56", "zero_share_28", "days_since_last_sale", "stockout_share_28",
    "promo_days_28", "closing_stock_origin", "on_order_origin", "cover_days_origin", "last_same_weekday",
    "mean_same_weekday_8w",
    # target day: calendar and planned promotions are known in advance
    "horizon", "dow", "is_weekend", "month", "day_of_month", "is_ramadan", "ramadan_day", "is_eid_al_fitr",
    "is_eid_al_adha", "is_pre_eid_al_fitr", "is_pre_eid_al_adha", "promo_flag", "promo_discount_pct",
]
CATEGORICAL_FEATURES = ("category_code", "store_format_code")
FEATURE_DEFINITION = (
    "Direct multi-horizon gradient boosting (Poisson loss) on observed units sold. Features: static series attributes; "
    "demand means over the last 7/14/28/56 days, zero share and stockout share of the last 28 days, days since last "
    "sale (cap 60), promotion days in the last 28, stock and on-order at the origin and cover in days, last and 8-week "
    "mean value on the target weekday; target-day calendar flags, horizon step and planned promotion discount. "
    "No data after the origin and no latent demand is used.")


def features_at(panel: Panel, o: int, fut_calendar: pd.DataFrame, fut_promo: np.ndarray,
                policy: str = "exclude") -> np.ndarray:
    """Feature matrix for origin day index `o`: shape (n_series * H, len(FEATURE_COLUMNS)), series-major, then horizon 1..H."""
    n, horizon = panel.n_series, len(fut_calendar)
    u = panel.units[:, : o + 1]                      # NOTHING beyond column o is read below
    c = panel.censored[:, : o + 1]
    promo_hist = panel.promo_discount[:, : o + 1]
    closing_o, on_order_o = panel.closing_stock[:, o], panel.on_order[:, o]
    y = np.where(c, np.nan, u) if policy == "exclude" else u

    mean_7, mean_14, mean_28, mean_56 = (nanmean_last(y, k) for k in (7, 14, 28, 56))
    y28 = last_k(y, 28)
    valid28 = ~np.isnan(y28)
    zero_share = np.where(valid28.sum(1) > 0, ((y28 == 0) & valid28).sum(1) / np.maximum(valid28.sum(1), 1), np.nan)
    sold = np.nan_to_num(u, nan=0.0) > 0
    since = np.where(sold.any(1), np.argmax(sold[:, ::-1], axis=1), 60)
    since = np.minimum(since, 60).astype(float)
    stockout_share = last_k(c, 28).mean(axis=1)
    promo_days = (last_k(promo_hist, 28) > 0).sum(axis=1).astype(float)
    cover = np.clip(closing_o / (np.nan_to_num(mean_28, nan=0.0) + 0.05), 0, 365)

    dow_o = panel.dates[o].dayofweek
    last_wd = np.full((n, 7), np.nan)
    mean_wd = np.full((n, 7), np.nan)
    for d in range(7):
        idx0 = o - ((dow_o - d) % 7)
        lags = [idx0 - 7 * i for i in range(8) if idx0 - 7 * i >= 0]
        if lags:
            last_wd[:, d] = y[:, lags[0]]
            mean_wd[:, d] = nanmean_rows(y[:, lags])

    target_dow = fut_calendar["dow"].to_numpy(dtype=int)
    rep = lambda a: np.repeat(np.asarray(a, dtype=float)[:, None], horizon, axis=1)       # per series -> (n, H)
    per_day = lambda col: np.tile(fut_calendar[col].to_numpy(dtype=float), (n, 1))         # per target day -> (n, H)
    s = panel.series
    cols = {
        "category_code": rep(s["category_code"]), "store_format_code": rep(s["store_format_code"]),
        "floor_area_sqm": rep(s["floor_area_sqm"]), "list_price": rep(s["list_price"]),
        "is_perishable": rep(s["is_perishable"]),
        "mean_7": rep(mean_7), "mean_14": rep(mean_14), "mean_28": rep(mean_28), "mean_56": rep(mean_56),
        "zero_share_28": rep(zero_share), "days_since_last_sale": rep(since), "stockout_share_28": rep(stockout_share),
        "promo_days_28": rep(promo_days), "closing_stock_origin": rep(closing_o), "on_order_origin": rep(on_order_o),
        "cover_days_origin": rep(cover),
        "last_same_weekday": last_wd[:, target_dow], "mean_same_weekday_8w": mean_wd[:, target_dow],
        "horizon": np.tile(np.arange(1, horizon + 1, dtype=float), (n, 1)),
        "promo_flag": (np.asarray(fut_promo) > 0).astype(float), "promo_discount_pct": np.asarray(fut_promo, dtype=float),
    }
    for col in ("dow", "is_weekend", "month", "day_of_month", "is_ramadan", "ramadan_day", "is_eid_al_fitr",
                "is_eid_al_adha", "is_pre_eid_al_fitr", "is_pre_eid_al_adha"):
        cols[col] = per_day(col)
    return np.stack([cols[k] for k in FEATURE_COLUMNS], axis=-1).reshape(n * horizon, len(FEATURE_COLUMNS))


def training_frame(hist: Panel, horizon: int, policy: str = "exclude", step: int = 7, max_origins: int = 52,
                   min_history: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Training examples built ONLY from `hist` (history up to the forecast origin).

    Each past origin o with o + horizon inside `hist` gives n_series x horizon rows (features at o, target = units on
    day o + h). Under the "exclude" policy rows whose target day is censored (stockout) are dropped.
    """
    origins, o = [], hist.n_days - 1 - horizon
    while o >= min_history - 1 and len(origins) < max_origins:
        origins.append(o)
        o -= step
    xs, ys = [], []
    for o in reversed(origins):
        sl = slice(o + 1, o + 1 + horizon)
        x = features_at(hist, o, hist.calendar.iloc[sl].reset_index(drop=True), hist.promo_discount[:, sl], policy)
        target = hist.units[:, sl].reshape(-1)
        ok = ~np.isnan(target)
        if policy == "exclude":
            ok &= ~hist.censored[:, sl].reshape(-1)
        xs.append(x[ok])
        ys.append(target[ok])
    if not xs:
        return np.empty((0, len(FEATURE_COLUMNS))), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)
