"""Page filters (pure, no Streamlit).

Decision D1: every filter defaults to "All". An empty tuple means All; the date range defaults
to the full loaded range. Each page owns its own filter object (no shared state).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd


def _as_date(value) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    return value


def normalize_date_range(value, lo: date, hi: date) -> tuple[date, date]:
    """Turn whatever a date-range widget returns into a valid ``(start, end)`` inside [lo, hi].

    * a full range (start, end)           -> clamped to the data range, swapped if reversed
    * a single date, or a 1-tuple (the user has picked only the first date so far) -> that day
    * nothing / None                       -> the whole range (All)
    """
    if value is None:
        return lo, hi
    if isinstance(value, (list, tuple)):
        items = [_as_date(v) for v in value if v is not None]
    else:
        items = [_as_date(value)]
    if not items:
        return lo, hi
    start, end = items[0], items[-1]
    if start > end:
        start, end = end, start
    start, end = max(start, lo), min(end, hi)
    if start > end:                     # selection entirely outside the data
        return lo, hi
    return start, end


@dataclass(frozen=True)
class OverviewFilters:
    """Executive Overview slicers: Date range, Store, Category, Store format."""

    date_start: date | None = None      # None = from the first loaded day
    date_end: date | None = None        # None = to the last loaded day
    stores: tuple[str, ...] = ()        # empty = All
    categories: tuple[str, ...] = ()
    store_formats: tuple[str, ...] = ()

    def is_default(self, lo: date, hi: date) -> bool:
        return (
            self.date_start in (None, lo) and self.date_end in (None, hi)
            and not (self.stores or self.categories or self.store_formats)
        )


def as_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else ()
