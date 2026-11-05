"""Backtest accuracy measures (ported DAX): additive sums, blanks, scope, skill, rank."""

import pandas as pd
import pytest

from dzmi_app import measures_forecast as mf


def rows(*items):
    return pd.DataFrame(items, columns=["n_obs", "sum_actual", "sum_pred", "sum_abs_error", "sum_sq_error"])


A = rows((10, 100.0, 110.0, 20.0, 60.0), (30, 200.0, 180.0, 90.0, 400.0))      # pooled: n=40


def test_sums_are_additive_and_blank_without_rows():
    s = mf.sums_of(A)
    assert (s.n_obs, s.actual, s.pred, s.abs_error, s.sq_error) == (40, 300.0, 290.0, 110.0, 460.0)
    assert mf.sums_of(A.iloc[0:0]) is None
    assert mf.observations(None) is None and mf.observations(s) == 40


def test_metrics_are_recomputed_from_the_sums():
    s = mf.sums_of(A)
    assert mf.mae(s) == pytest.approx(110 / 40)
    assert mf.rmse(s) == pytest.approx((460 / 40) ** 0.5)
    assert mf.wape(s) == pytest.approx(110 / 300)
    assert mf.bias(s) == pytest.approx((290 - 300) / 300)


def test_pooled_mae_is_weighted_never_an_average_of_averages():
    first, second = mf.mae(mf.sums_of(A.iloc[[0]])), mf.mae(mf.sums_of(A.iloc[[1]]))     # 2.0 and 3.0
    assert (first + second) / 2 == pytest.approx(2.5)
    assert mf.mae(mf.sums_of(A)) == pytest.approx(2.75)                                    # (20+90)/(10+30)
    assert mf.mae(mf.sums_of(A)) != pytest.approx((first + second) / 2)


def test_every_metric_is_blank_on_blank_sums():
    for fn in (mf.mae, mf.rmse, mf.wape, mf.bias):
        assert fn(None) is None


def test_zero_actual_demand_makes_wape_and_bias_blank_but_not_mae():
    s = mf.sums_of(rows((5, 0.0, 3.0, 3.0, 9.0)))
    assert mf.wape(s) is None and mf.bias(s) is None
    assert mf.mae(s) == pytest.approx(0.6) and mf.rmse(s) == pytest.approx((9 / 5) ** 0.5)


def test_bias_sign_positive_means_over_forecast():
    assert mf.bias(mf.sums_of(rows((1, 100.0, 120.0, 20.0, 400.0)))) == pytest.approx(0.2)
    assert mf.bias(mf.sums_of(rows((1, 100.0, 80.0, 20.0, 400.0)))) == pytest.approx(-0.2)


# ------------------------------------------------------------------ skill versus the benchmark


def test_skill_is_one_minus_mae_over_benchmark():
    assert mf.skill_vs_benchmark(2.5, 3.4) == pytest.approx(1 - 2.5 / 3.4)
    assert mf.skill_vs_benchmark(3.4, 3.4) == 0.0
    assert mf.skill_vs_benchmark(4.0, 3.0) < 0


def test_skill_is_blank_without_a_usable_benchmark():
    assert mf.skill_vs_benchmark(2.0, None) is None
    assert mf.skill_vs_benchmark(2.0, 0) is None


def test_skill_of_a_blank_mae_follows_dax_one_minus_blank():
    assert mf.skill_vs_benchmark(None, 3.0) == 1.0               # 1 - DIVIDE(BLANK, x) = 1 in DAX


# ------------------------------------------------------------------ scope and rank


@pytest.mark.parametrize("bases,levels,expected", [
    ((), (), ("observed", "series_day")),
    (("observed_uncensored",), (), ("observed_uncensored", "series_day")),
    ((), ("store_category_week",), ("observed", "store_category_week")),
    (("observed", "observed_uncensored"), ("series_day", "store_category_week"), ("observed", "series_day")),
    (("observed_uncensored",), ("store_category_week",), ("observed_uncensored", "store_category_week")),
])
def test_scope_is_the_single_selected_value_else_the_default(bases, levels, expected):
    assert mf.resolve_scope(bases, levels) == expected


def test_dense_rank_ascending_gives_ties_the_same_rank_and_no_gaps():
    population = [2.5, 2.6, 2.6, 3.4, None]
    assert [mf.dense_rank_ascending(v, population) for v in (2.5, 2.6, 3.4)] == [1, 2, 3]
    assert mf.dense_rank_ascending(None, population) is None


def test_rank_of_a_single_model_is_one():
    assert mf.dense_rank_ascending(5.0, [5.0]) == 1
