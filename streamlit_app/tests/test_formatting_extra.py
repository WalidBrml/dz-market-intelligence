"""Display formats of pages 3-6 (format strings of the semantic model)."""

import pytest

from dzmi_app.formatting_extra import fmt_decimal, fmt_percent_spaced, fmt_signed_percent, fmt_text_or_blank


@pytest.mark.parametrize("value,decimals,expected", [
    (2.50937, 3, "2.509"), (31.5069, 3, "31.507"), (0.0, 3, "0.000"), (1.0005, 3, "1.001"),
    (-7.1e-15, 6, "0.000000"), (-0.25, 3, "-0.250"), (1234.5678, 3, "1234.568"),
])
def test_decimal_has_no_thousands_separator_and_no_negative_zero(value, decimals, expected):
    assert fmt_decimal(value, decimals) == expected


@pytest.mark.parametrize("value,expected", [
    (0.2642, "+26.4%"), (-0.0679, "-6.8%"), (0.0, "0.0%"), (0.00001, "0.0%"), (-0.00001, "0.0%"), (0.094, "+9.4%"),
])
def test_signed_percent_matches_plus_minus_zero_format(value, expected):
    assert fmt_signed_percent(value, 1) == expected


@pytest.mark.parametrize("value,expected", [(0.6682577565632458, "66.8 %"), (0.0, "0.0 %"), (1.0, "100.0 %")])
def test_percent_with_a_space_like_the_at_risk_measure(value, expected):
    assert fmt_percent_spaced(value, 1) == expected


@pytest.mark.parametrize("fn", [fmt_decimal, fmt_signed_percent, fmt_percent_spaced, fmt_text_or_blank])
@pytest.mark.parametrize("blank", [None, float("nan")])
def test_blank_is_an_empty_cell(fn, blank):
    assert fn(blank) == ""


def test_text_passes_through():
    assert fmt_text_or_blank("OK") == "OK" and fmt_text_or_blank(0) == "0"
