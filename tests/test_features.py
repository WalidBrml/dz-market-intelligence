"""Milestone 3: series panel, features and leakage protection (no future data, no latent demand)."""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd

from dzmi.errors import DZMIError
from dzmi.features.build import CATEGORICAL_FEATURES, FEATURE_COLUMNS, features_at, training_frame
from dzmi.features.panel import Panel, load_panel, select_series
from dzmi.forecasting.backtest import run_backtest
from dzmi.paths import REPO_ROOT
from helpers import forecast_settings, loaded_sqlite, raises, shared_smoke_db, shared_smoke_panel, smoke

F = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


# ------------------------------------------------------------------------------------------ panel from the warehouse
def test_panel_matches_the_raw_source_tables():
    cfg, raw, dwh = smoke()
    p = shared_smoke_panel()
    inv = raw["inventory_daily"].merge(dwh["dim_store"][["store_id", "store_key"]], on="store_id") \
                                .merge(dwh["dim_product"][["product_id", "product_key"]], on="product_id")
    assert p.units.shape == (len(raw["store_assortment"]), 90) == (419, 90)
    assert not np.isnan(p.units).any() and not np.isnan(p.closing_stock).any()
    key = {(a, b): i for i, (a, b) in enumerate(zip(p.series["store_key"], p.series["product_key"]))}
    col = {d: i for i, d in enumerate(p.dates)}
    rows = np.array([key[(a, b)] for a, b in zip(inv["store_key"], inv["product_key"])])
    cols = np.array([col[d] for d in inv["inventory_date"]])
    assert np.array_equal(p.units[rows, cols], inv["units_sold"].to_numpy(dtype=float))
    assert np.array_equal(p.censored[rows, cols], inv["stockout_flag"].to_numpy(dtype=bool))   # stockout = censored day
    assert np.array_equal(p.closing_stock[rows, cols], inv["closing_stock"].to_numpy(dtype=float))
    assert p.censored.sum() == int(inv["stockout_flag"].sum()) == 794


def test_panel_promotions_and_calendar_match_the_warehouse():
    cfg, raw, dwh = smoke()
    p = shared_smoke_panel()
    pr = dwh["fact_promotions"]
    assert int((p.promo_discount > 0).sum()) == len(pr) == 2086
    assert abs(p.promo_discount.sum() - pd.to_numeric(pr["discount_pct"]).sum()) < 1e-6
    d = dwh["dim_date"].sort_values("date_key")
    assert (p.calendar["is_ramadan"].to_numpy() == d["is_ramadan"].astype(int).to_numpy()).all()
    assert (p.calendar["is_weekend"].to_numpy() == d["is_weekend"].astype(int).to_numpy()).all()
    assert (p.calendar["is_eid_al_fitr"].to_numpy() == d["is_eid_al_fitr"].astype(int).to_numpy()).all()
    assert (p.calendar["dow"].to_numpy() == pd.to_datetime(d["full_date"]).dt.dayofweek.to_numpy()).all()
    assert p.calendar["is_ramadan"].sum() > 0 and p.calendar["is_eid_al_fitr"].sum() >= 1


def test_series_sampling_is_deterministic_stratified_and_bounded():
    db = shared_smoke_db()
    a, b = select_series(db, 50, 7), select_series(db, 50, 7)
    assert a.equals(b) and len(a) == 50 and not a.duplicated(["store_key", "product_key"]).any()
    assert not a.equals(select_series(db, 50, 8))
    allv = select_series(db, 0)
    assert len(allv) == 419 and len(select_series(db, 10_000)) == 419
    decile = pd.qcut(allv["total_units"].rank(method="first"), 10, labels=False)
    top, bottom = allv[decile == 9], allv[decile == 0]
    merged = a.merge(allv.assign(dec=decile), on=["store_key", "product_key"])
    assert (merged["dec"] == 9).any() and (merged["dec"] == 0).any()                # rare and best-selling items present
    assert len(load_panel(db, 30, 1).series) == 30


def test_slice_and_future_enforce_the_calendar_bounds():
    p = shared_smoke_panel()
    raises(DZMIError, p.slice_to, 90)
    raises(DZMIError, p.future_known, 80, 14)               # 80 + 14 >= 90: targets must exist in the calendar
    assert p.future_known(75, 14).dates[-1] == p.dates[89]
    assert p.index_of("2025-02-01") == 0
    raises(DZMIError, p.index_of, "2024-01-01")


# ------------------------------------------------------------------------------------------ features: known answers
def _line_panel(values, censored=None, closing=10.0, on_order=3.0):
    v = np.asarray(values, dtype=float)[None, :]
    c = np.zeros(v.shape, dtype=bool)
    if censored is not None:
        c[0, list(censored)] = True
    return Panel.from_arrays(np.concatenate([v, np.zeros((1, 14))], axis=1),
                             censored=np.concatenate([c, np.zeros((1, 14), dtype=bool)], axis=1),
                             closing_stock=np.full((1, v.shape[1] + 14), closing), on_order=np.full((1, v.shape[1] + 14), on_order))


def _feats(panel, o, h=7, policy="exclude"):
    fut = panel.future_known(o, h)
    return features_at(panel, o, fut.calendar, fut.promo_discount, policy)


def test_history_features_known_answers():
    p = _line_panel(range(1, 36))                       # 35 days, Monday..Sunday x 5: value = day number, origin is a Sunday
    x = _feats(p, 34)
    first = x[0]                                         # target = next Monday, horizon step 1
    assert first[F["mean_7"]] == np.mean(range(29, 36)) and first[F["mean_14"]] == np.mean(range(22, 36))
    assert first[F["mean_28"]] == np.mean(range(8, 36)) and first[F["mean_56"]] == np.mean(range(1, 36))
    assert first[F["zero_share_28"]] == 0 and first[F["days_since_last_sale"]] == 0
    assert first[F["last_same_weekday"]] == 29            # last Monday on or before the origin
    assert first[F["mean_same_weekday_8w"]] == np.mean([1, 8, 15, 22, 29])
    assert first[F["horizon"]] == 1 and first[F["dow"]] == 0 and x[6][F["dow"]] == 6
    assert first[F["closing_stock_origin"]] == 10 and first[F["on_order_origin"]] == 3
    assert abs(first[F["cover_days_origin"]] - 10 / (np.mean(range(8, 36)) + 0.05)) < 1e-12
    assert x.shape == (7, len(FEATURE_COLUMNS))


def test_days_since_last_sale_zero_share_and_censoring_in_features():
    p = _line_panel([2, 3, 4, 5, 0, 0, 3, 0, 0, 0])
    x = _feats(p, 9)[0]
    assert x[F["days_since_last_sale"]] == 3 and abs(x[F["zero_share_28"]] - 0.5) < 1e-12
    pc = _line_panel(range(1, 36), censored=[34])        # the last day (value 35) ended in a stockout
    assert _feats(pc, 34, policy="exclude")[0][F["mean_7"]] == np.mean(range(29, 35))   # ignored under exclude
    assert _feats(pc, 34, policy="keep")[0][F["mean_7"]] == np.mean(range(29, 36))
    assert _feats(pc, 34, policy="exclude")[0][F["stockout_share_28"]] == 1 / 28         # but still described


def test_target_day_features_use_the_known_future_calendar_and_planned_promotions():
    p = shared_smoke_panel()
    o = 60
    fut = p.future_known(o, 14)
    x = features_at(p, o, fut.calendar, fut.promo_discount)
    n = p.n_series
    cube = x.reshape(n, 14, -1)
    assert (cube[:, :, F["horizon"]] == np.arange(1, 15)).all()
    assert (cube[:, :, F["dow"]] == p.calendar["dow"].to_numpy()[o + 1:o + 15]).all()
    assert (cube[:, :, F["promo_discount_pct"]] == p.promo_discount[:, o + 1:o + 15]).all()
    assert (cube[:, :, F["promo_flag"]] == (p.promo_discount[:, o + 1:o + 15] > 0)).all()
    assert cube[:, :, F["promo_flag"]].sum() > 0
    assert [FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))] == list(F)                       # stable order
    assert set(CATEGORICAL_FEATURES) <= set(FEATURE_COLUMNS)


# ------------------------------------------------------------------------------------------ leakage
def _scramble_after(panel, o, seed=0):
    rng = np.random.default_rng(seed)
    g = dataclasses.replace(panel, units=panel.units.copy(), censored=panel.censored.copy(),
                            closing_stock=panel.closing_stock.copy(), on_order=panel.on_order.copy(),
                            promo_discount=panel.promo_discount.copy())
    for a in (g.units, g.closing_stock, g.on_order):
        a[:, o + 1:] = rng.integers(0, 1000, a[:, o + 1:].shape)
    g.censored[:, o + 1:] = rng.random(g.censored[:, o + 1:].shape) < 0.5
    g.promo_discount[:, o + 1:] = rng.integers(0, 90, g.promo_discount[:, o + 1:].shape)
    return g


def test_features_do_not_change_when_the_future_is_overwritten():
    p = shared_smoke_panel()
    o = 55
    fut = p.future_known(o, 14)
    base = features_at(p, o, fut.calendar, fut.promo_discount)
    g = _scramble_after(p, o)
    np.testing.assert_array_equal(base, features_at(g, o, fut.calendar, fut.promo_discount))   # NaN-aware equality
    np.testing.assert_array_equal(base, features_at(p.slice_to(o), o, fut.calendar, fut.promo_discount))


def test_training_examples_use_only_the_history_before_the_origin():
    p = shared_smoke_panel()
    o = 70
    x1, y1 = training_frame(p.slice_to(o), 14)
    x2, y2 = training_frame(_scramble_after(p, o).slice_to(o), 14)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert len(y1) > 1000 and (y1 >= 0).all()


def test_training_targets_never_reach_past_the_origin_and_censored_targets_are_dropped():
    v = np.full((1, 60), 5.0)
    c = np.zeros((1, 60), dtype=bool)
    c[0, 45] = True
    p = Panel.from_arrays(v, censored=c)
    h = 7
    x_keep, y_keep = training_frame(p, h, policy="keep", step=7)
    x_ex, y_ex = training_frame(p, h, policy="exclude", step=7)
    assert len(y_keep) - len(y_ex) == 1                  # exactly the one censored target day (index 45) is dropped
    # the newest training origin leaves exactly h days inside the history: no target beyond the last history day
    assert len(y_keep) == h * len(range(59 - h, 27, -7))


# ------------------------------------------------------------------------------------------ latent demand
def test_training_code_never_references_latent_demand():
    root = REPO_ROOT / "src" / "dzmi"
    allowed = {root / "forecasting" / "diagnostics.py"}
    offenders = []
    for path in list((root / "features").glob("*.py")) + list((root / "forecasting").glob("*.py")):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        code = re.sub(r'""".*?"""', "", code, flags=re.S)          # docstrings may mention it to forbid it
        if re.search(r"sim_latent|latent_demand|latent_demand_units|unmet_demand", code):
            offenders.append(path.name)
    assert offenders == [], f"latent demand referenced outside diagnostics.py: {offenders}"
    importers = [p.name for p in root.rglob("*.py")
                 if re.search(r"^\s*(from|import)\s[^\n]*diagnostics", p.read_text(encoding="utf-8"), re.M)]
    assert set(importers) <= {"cli.py"}, importers        # only the CLI may switch the diagnostic on


def test_results_do_not_depend_on_the_latent_table():
    """Run the same backtest with and without ops.sim_latent_demand: every metric must be identical."""
    s = forecast_settings(models="naive,seasonal_naive,croston_sba,gbm", n_origins=1, max_series=60)
    db = loaded_sqlite()
    before = run_backtest(load_panel(db, s.max_series, s.seed), s, progress=lambda m: None).metrics
    db._h.execute("PRAGMA foreign_keys = OFF")
    db._h.execute("DROP TABLE ops.sim_latent_demand")
    assert db.query_df("SELECT name FROM ops.sqlite_master WHERE name = 'sim_latent_demand'").empty
    after = run_backtest(load_panel(db, s.max_series, s.seed), s, progress=lambda m: None).metrics
    pd.testing.assert_frame_equal(before, after)
    db.close()
