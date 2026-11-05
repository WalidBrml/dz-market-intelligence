"""Display formatting mirrors the Power BI format strings used by the Page 1 cards."""

import numpy as np
import pytest

from dzmi_app.formatting import (
    BLANK, fmt_count, fmt_millions, fmt_number, fmt_percent, fmt_thousands, is_blank,
)


@pytest.mark.parametrize("value,decimals,expected", [
    (40389951.0, 1, "40.4M"),        # KPI Net Sales: #,0.0,,"M"
    (6842371.34, 1, "6.8M"),         # KPI Gross Profit
    (1608182.54, 2, "1.61M"),        # KPI Stock Value At Cost: #,0.00,,"M"
    (0, 1, "0.0M"),
    (1_250_000, 1, "1.3M"),          # half away from zero, not banker's rounding
    (1_050_000, 1, "1.1M"),
    (-2_500_000, 1, "-2.5M"),
    (1_234_567_890, 1, "1,234.6M"),
])
def test_millions(value, decimals, expected):
    assert fmt_millions(value, decimals) == expected


@pytest.mark.parametrize("value,decimals,expected", [
    (165893, 1, "165.9K"),           # KPI Units Sold: #,0.0,"K"
    (1_250, 1, "1.3K"),
    (999, 1, "1.0K"),
    (0, 1, "0.0K"),
])
def test_thousands(value, decimals, expected):
    assert fmt_thousands(value, decimals) == expected


@pytest.mark.parametrize("value,decimals,expected", [
    (6992.720048476454, 2, "6,992.72"),   # Average Basket Value: #,0.00
    (1234.5, 0, "1,235"),
    (2.5, 0, "3"),
    (0.5, 0, "1"),
    (1234567.891, 2, "1,234,567.89"),
    (0, 2, "0.00"),
])
def test_number(value, decimals, expected):
    assert fmt_number(value, decimals) == expected


@pytest.mark.parametrize("value,decimals,expected", [
    (0.16940776531271354, 1, "16.9%"),    # Gross Margin %: 0.0%
    (0.02105542296473084, 2, "2.11%"),    # Stockout Rate: 0.00%
    (0.0, 2, "0.00%"),
    (0.5, 1, "50.0%"),
    (1.0, 0, "100%"),
])
def test_percent(value, decimals, expected):
    assert fmt_percent(value, decimals) == expected


@pytest.mark.parametrize("blank", [None, float("nan"), np.nan])
@pytest.mark.parametrize("fn", [fmt_number, fmt_millions, fmt_thousands, fmt_percent])
def test_blank_values_show_a_dash(fn, blank):
    assert is_blank(blank)
    assert fn(blank) == BLANK == "\u2014"


def test_counts_show_zero_when_blank_and_accept_numpy_integers():
    assert fmt_count(None) == "0"
    assert fmt_count(float("nan")) == "0"
    assert fmt_count(1234) == "1,234"
    assert fmt_count(np.int64(83)) == "83"


def test_zero_is_not_blank():
    assert not is_blank(0) and not is_blank(0.0)
    assert fmt_millions(0) != BLANK
