"""Filter objects and date-range normalisation (decisions D1 and V4)."""

from datetime import date, datetime

import pandas as pd
import pytest

from dzmi_app.filters import OverviewFilters, as_tuple, normalize_date_range

LO, HI = date(2025, 2, 1), date(2025, 5, 1)


def test_filters_default_to_all():
    f = OverviewFilters()
    assert f.date_start is None and f.date_end is None
    assert f.stores == () and f.categories == () and f.store_formats == ()
    assert f.is_default(LO, HI)


def test_full_range_counts_as_default_but_any_selection_does_not():
    assert OverviewFilters(date_start=LO, date_end=HI).is_default(LO, HI)
    assert not OverviewFilters(date_start=date(2025, 3, 1), date_end=HI).is_default(LO, HI)
    assert not OverviewFilters(date_start=LO, date_end=date(2025, 4, 30)).is_default(LO, HI)
    assert not OverviewFilters(stores=("Alger Centre",)).is_default(LO, HI)
    assert not OverviewFilters(categories=("Dairy & Eggs",)).is_default(LO, HI)
    assert not OverviewFilters(store_formats=("Supermarket",)).is_default(LO, HI)


def test_filters_are_immutable():
    f = OverviewFilters()
    with pytest.raises(Exception):
        f.stores = ("x",)


def test_as_tuple():
    assert as_tuple(None) == () and as_tuple([]) == ()
    assert as_tuple(["a", "b"]) == ("a", "b")


@pytest.mark.parametrize("value,expected", [
    ((LO, HI), (LO, HI)),                                           # full range
    ((date(2025, 3, 1), date(2025, 3, 29)), (date(2025, 3, 1), date(2025, 3, 29))),
    ((date(2025, 3, 29), date(2025, 3, 1)), (date(2025, 3, 1), date(2025, 3, 29))),   # reversed
    ((date(2025, 1, 1), date(2025, 3, 1)), (LO, date(2025, 3, 1))),                    # clamped low
    ((date(2025, 4, 1), date(2025, 12, 31)), (date(2025, 4, 1), HI)),                  # clamped high
    ((date(2025, 3, 10),), (date(2025, 3, 10), date(2025, 3, 10))),   # first click of a range
    (date(2025, 3, 10), (date(2025, 3, 10), date(2025, 3, 10))),      # a single date
    (None, (LO, HI)),
    ((), (LO, HI)),
    ([], (LO, HI)),
    ((date(2024, 1, 1), date(2024, 12, 31)), (LO, HI)),               # entirely outside the data
])
def test_normalize_date_range(value, expected):
    assert normalize_date_range(value, LO, HI) == expected


def test_normalize_accepts_timestamps_and_datetimes():
    assert normalize_date_range((pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-05")), LO, HI) \
        == (date(2025, 3, 1), date(2025, 3, 5))
    assert normalize_date_range((datetime(2025, 3, 1, 15, 30), datetime(2025, 3, 5)), LO, HI) \
        == (date(2025, 3, 1), date(2025, 3, 5))


def test_normalize_returns_plain_dates():
    start, end = normalize_date_range((pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-05")), LO, HI)
    assert type(start) is date and type(end) is date
