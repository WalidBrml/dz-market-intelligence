"""Milestone 3: forecasting models against hand-computed answers, plus model-level leakage tests."""
from __future__ import annotations

import dataclasses

import numpy as np

from dzmi.features.panel import Panel
from dzmi.forecasting.models import (MODEL_DESCRIPTIONS, croston_sba, forecast_model, moving_average, naive,
                                     seasonal_naive, ses_weekday)
from helpers import forecast_settings, shared_smoke_panel

MON = "2025-01-06"          # a Monday: day index 0 is Monday, index 6 is Sunday


def run(name, history, horizon=7, censored=None, policy="exclude", **settings):
    """Forecast `horizon` days after the last day of `history` (1 series) with any model, through the public path."""
    y = np.asarray(history, dtype=float)[None, :]
    units = np.concatenate([y, np.zeros((1, horizon))], axis=1)
    cens = np.zeros(units.shape, dtype=bool)
    if censored is not None:
        cens[0, list(censored)] = True
    panel = Panel.from_arrays(units, start=MON, censored=cens)
    o = y.shape[1] - 1
    s = forecast_settings(censor_policy=policy, horizon_days=horizon, **settings)
    return forecast_model(name, panel.slice_to(o), panel.future_known(o, horizon), s)[0]


# ------------------------------------------------------------------------------------------ naive family
def test_naive_repeats_last_value_and_skips_censored_day():
    assert run("naive", [1, 2, 3, 0], 3).tolist() == [0, 0, 0]
    assert run("naive", [1, 2, 3, 0], 3, censored=[3], policy="exclude").tolist() == [3, 3, 3]   # the zero was a stockout
    assert run("naive", [1, 2, 3, 0], 3, censored=[3], policy="keep").tolist() == [0, 0, 0]


def test_seasonal_naive_is_the_same_weekday_one_week_earlier():
    hist = list(range(1, 22))                      # 21 days: Mon..Sun, three weeks, last day Sunday = 21
    out = run("seasonal_naive", hist, 14)
    assert out.tolist() == list(range(15, 22)) * 2   # steps 1-7 and 8-14 both repeat the last week


def test_seasonal_naive_falls_back_to_older_weeks_when_the_day_is_censored():
    hist = list(range(1, 22))
    wed_last = 16                                   # index of week 3 Wednesday (value 17)
    assert run("seasonal_naive", hist, 7, censored=[wed_last], policy="exclude")[2] == 10   # week 2 Wednesday
    assert run("seasonal_naive", hist, 7, censored=[wed_last], policy="keep")[2] == 17
    # every Wednesday censored -> fall back to the mean of the last 28 usable days
    all_wed = [2, 9, 16]
    expected = np.mean([v for i, v in enumerate(hist) if i not in all_wed])
    assert abs(run("seasonal_naive", hist, 7, censored=all_wed, policy="exclude")[2] - expected) < 1e-12


def test_moving_average_ignores_censored_days_and_respects_the_window():
    assert abs(run("moving_average", [2, 4, 6, 100], 2, censored=[3], policy="exclude")[0] - 4.0) < 1e-12
    assert abs(run("moving_average", [2, 4, 6, 100], 2, policy="keep")[0] - 28.0) < 1e-12
    y = np.arange(1, 41, dtype=float)[None, :]
    assert abs(moving_average(y, 3, window=28)[0, 0] - np.mean(np.arange(13, 41))) < 1e-12


# ------------------------------------------------------------------------------------------ exponential smoothing
def test_ses_weekday_reproduces_a_constant_and_a_weekly_pattern():
    assert np.allclose(run("ses_weekday", [5.0] * 60, 7), 5.0)
    pattern = [1, 2, 3, 4, 5, 6, 7]
    out = run("ses_weekday", pattern * 9, 7)                          # history ends on Sunday, forecast starts Monday
    assert np.allclose(out, pattern, atol=1e-9)


def test_ses_weekday_follows_a_level_shift_and_is_never_negative():
    y = np.concatenate([np.full(60, 2.0), np.full(30, 8.0)])[None, :]
    dow = np.arange(90) % 7
    out = ses_weekday(y, dow, np.arange(7), alphas=(0.3,))
    assert (out > 6.0).all() and (out < 9.5).all()                    # moved most of the way from 2 to 8; the weekday
    assert abs(out.mean() - 8.0) < 1.0                                # index (estimated on a mixed window) adds a little spread
    assert (ses_weekday(np.zeros((2, 40)), np.arange(40) % 7, np.arange(7)) == 0).all()


# ------------------------------------------------------------------------------------------ Croston
def test_croston_sba_known_answers():
    assert abs(croston_sba(np.array([[0, 0, 4, 0, 4.0]]), 2)[0, 0] - 0.95 * 4 / 2.9) < 1e-12   # worked by hand
    assert abs(croston_sba(np.full((1, 30), 4.0), 3, alpha=0.1)[0, 0] - (1 - 0.05) * 4.0) < 1e-12   # demand every day
    assert croston_sba(np.zeros((1, 30)), 3).tolist() == [[0, 0, 0]]
    alternating = np.tile([4.0, 0.0], 100)[None, :]
    assert 1.8 < croston_sba(alternating, 1)[0, 0] < 2.0               # interval -> 2, size 4: about 0.95 * 4 / 2


def test_croston_skips_censored_days_instead_of_lengthening_the_interval():
    y = np.array([[4, np.nan, np.nan, 4, np.nan, 4, 4.0]])
    assert abs(croston_sba(y, 1)[0, 0] - 0.95 * 4.0) < 1e-12          # usable days all had demand: interval 1
    y2 = np.array([[4, 0, 0, 4, 0, 4, 4.0]])
    assert croston_sba(y2, 1)[0, 0] < 0.95 * 4.0                      # the same days as real zeros do lengthen it


# ------------------------------------------------------------------------------------------ all models
def test_every_model_returns_finite_nonnegative_forecasts_of_the_right_shape():
    panel = shared_smoke_panel()
    o = 60
    hist, fut = panel.slice_to(o), panel.future_known(o, 14)
    s = forecast_settings()
    assert set(s.models) == set(MODEL_DESCRIPTIONS)
    sample = dataclasses.replace(hist, series=hist.series.iloc[:60], units=hist.units[:60], censored=hist.censored[:60],
                                 closing_stock=hist.closing_stock[:60], on_order=hist.on_order[:60],
                                 promo_discount=hist.promo_discount[:60])
    fut60 = dataclasses.replace(fut, promo_discount=fut.promo_discount[:60])
    for name in s.models:
        pred = forecast_model(name, sample, fut60, s)
        assert pred.shape == (60, 14) and np.isfinite(pred).all() and (pred >= 0).all(), name


def test_models_handle_empty_and_all_censored_series():
    for name in ("naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba"):
        assert (run(name, [0.0] * 40, 7) == 0).all(), name
        assert (run(name, [3.0] * 40, 7, censored=range(40), policy="exclude") == 0).all(), name   # nothing usable


def test_gbm_is_reproducible_and_falls_back_to_zero_without_signal():
    panel = shared_smoke_panel()
    o = 60
    hist, fut = panel.slice_to(o), panel.future_known(o, 7)
    s = forecast_settings(horizon_days=7)
    sub = lambda a: a[:80]
    h = dataclasses.replace(hist, series=hist.series.iloc[:80], units=sub(hist.units), censored=sub(hist.censored),
                            closing_stock=sub(hist.closing_stock), on_order=sub(hist.on_order),
                            promo_discount=sub(hist.promo_discount))
    f = dataclasses.replace(fut, promo_discount=fut.promo_discount[:80])
    a, b = forecast_model("gbm", h, f, s), forecast_model("gbm", h, f, s)
    assert a.shape == (80, 7) and (a >= 0).all() and np.isfinite(a).all()
    assert np.allclose(a, b, rtol=0, atol=1e-9)                          # same seed, same machine, same numbers
    zero = dataclasses.replace(h, units=np.zeros_like(h.units))
    assert (forecast_model("gbm", zero, f, s) == 0).all()


# ------------------------------------------------------------------------------------------ leakage at model level
def test_no_model_can_see_the_future():
    """Overwrite everything after the origin with garbage: forecasts must not change (all six models)."""
    panel = shared_smoke_panel()
    o, h = 55, 14
    rng = np.random.default_rng(0)
    n = 40
    take = lambda a: a[:n].copy()
    base = dataclasses.replace(panel, series=panel.series.iloc[:n], units=take(panel.units), censored=take(panel.censored),
                               closing_stock=take(panel.closing_stock), on_order=take(panel.on_order),
                               promo_discount=take(panel.promo_discount))
    garbage = dataclasses.replace(
        base, units=base.units.copy(), censored=base.censored.copy(), closing_stock=base.closing_stock.copy(),
        on_order=base.on_order.copy())
    garbage.units[:, o + 1:] = rng.integers(0, 500, garbage.units[:, o + 1:].shape)
    garbage.censored[:, o + 1:] = rng.random(garbage.censored[:, o + 1:].shape) < 0.5
    garbage.closing_stock[:, o + 1:] = rng.integers(0, 999, garbage.closing_stock[:, o + 1:].shape)
    garbage.on_order[:, o + 1:] = rng.integers(0, 999, garbage.on_order[:, o + 1:].shape)
    s = forecast_settings(horizon_days=h)
    for name in s.models:
        a = forecast_model(name, base.slice_to(o), base.future_known(o, h), s)
        b = forecast_model(name, garbage.slice_to(o), garbage.future_known(o, h), s)
        assert np.array_equal(a, b), f"{name} used information from after the origin"


def test_history_and_future_objects_expose_no_leaking_fields():
    panel = shared_smoke_panel()
    hist, fut = panel.slice_to(40), panel.future_known(40, 14)
    assert hist.n_days == 41 and hist.dates[-1] == panel.dates[40] and hist.units.shape[1] == 41
    assert {f.name for f in dataclasses.fields(fut)} == {"dates", "calendar", "promo_discount"}   # no units, no stock
    assert fut.dates[0] == panel.dates[41] and len(fut.dates) == 14
