"""Warning-free NaN-aware statistics on (series x days) arrays. NaN means "not usable" (missing or censored)."""
from __future__ import annotations

import numpy as np


def nanmean_rows(a: np.ndarray) -> np.ndarray:
    """Row means ignoring NaN; NaN for rows with no valid value (no RuntimeWarning, unlike np.nanmean)."""
    valid = ~np.isnan(a)
    cnt = valid.sum(axis=1)
    total = np.where(valid, a, 0.0).sum(axis=1)
    return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)


def last_k(a: np.ndarray, k: int) -> np.ndarray:
    """The last k columns (all of them if fewer exist)."""
    return a[:, max(0, a.shape[1] - k):]


def nanmean_last(a: np.ndarray, k: int) -> np.ndarray:
    return nanmean_rows(last_k(a, k))
