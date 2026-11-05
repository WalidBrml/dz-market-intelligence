"""Backtest accuracy measures, ported from _Measures.tmdl. Everything is recomputed from additive
sums (never an average of averages):

    Forecast Observations = SUM ( n_obs )                 Forecast MAE  = DIVIDE ( sum_abs_error, n_obs )
    Forecast RMSE = SQRT ( DIVIDE ( sum_sq_error, n_obs ) )   Forecast WAPE = DIVIDE ( sum_abs_error, sum_actual )
    Forecast Bias = DIVIDE ( sum_pred - sum_actual, sum_actual )          (+ = over-forecast)
    Benchmark MAE = Forecast MAE of the seasonal_naive model            (see forecasting.py for scope)
    Skill vs Benchmark = IF ( NOT ISBLANK ( [Benchmark MAE] ) && [Benchmark MAE] <> 0,
                              1 - DIVIDE ( [Forecast MAE], [Benchmark MAE] ) )
    MAE Rank = RANKX ( ALLSELECTED ( dim_model[model_label] ), [Forecast MAE], , ASC, DENSE )

Scope: the level / basis are the single value selected in their slicer, else series_day / observed.
``None`` stands for DAX BLANK(). A zero denominator (for example no actual demand) gives BLANK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .measures_sales import divide

DEFAULT_LEVEL = "series_day"
DEFAULT_BASIS = "observed"
BENCHMARK_MODEL = "seasonal_naive"


@dataclass(frozen=True)
class Sums:
    n_obs: int
    actual: float
    pred: float
    abs_error: float
    sq_error: float


def sums_of(rows: pd.DataFrame) -> Sums | None:
    """SUM over the additive columns; BLANK (None) when there are no rows."""
    if len(rows) == 0:
        return None
    return Sums(int(rows["n_obs"].sum()), float(rows["sum_actual"].sum()), float(rows["sum_pred"].sum()),
                float(rows["sum_abs_error"].sum()), float(rows["sum_sq_error"].sum()))


def observations(s: Sums | None):
    return None if s is None else s.n_obs


def mae(s: Sums | None):
    return None if s is None else divide(s.abs_error, s.n_obs)


def rmse(s: Sums | None):
    value = None if s is None else divide(s.sq_error, s.n_obs)
    return None if value is None else math.sqrt(value)


def wape(s: Sums | None):
    return None if s is None else divide(s.abs_error, s.actual)


def bias(s: Sums | None):
    return None if s is None else divide(s.pred - s.actual, s.actual)


def skill_vs_benchmark(model_mae, benchmark_mae):
    """DAX evaluates ``1 - DIVIDE(BLANK, x)`` as 1 (BLANK counts as 0): reproduced literally."""
    if benchmark_mae is None or benchmark_mae == 0:
        return None
    ratio = divide(model_mae, benchmark_mae)
    return 1 - (0.0 if ratio is None else ratio)


def resolve_scope(bases: tuple[str, ...], levels: tuple[str, ...]) -> tuple[str, str]:
    """SELECTEDVALUE with defaults: exactly one selected value wins, anything else uses the default."""
    basis = bases[0] if len(bases) == 1 else DEFAULT_BASIS
    level = levels[0] if len(levels) == 1 else DEFAULT_LEVEL
    return basis, level


def dense_rank_ascending(value, population) -> int | None:
    """RANKX ( ..., ASC, DENSE ): 1 + number of distinct non-blank values below ``value``."""
    if value is None:
        return None
    distinct = {v for v in population if v is not None}
    return 1 + sum(1 for v in distinct if v < value)
