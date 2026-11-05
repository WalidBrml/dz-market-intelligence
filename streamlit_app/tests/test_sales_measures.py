"""Ported Sales Intelligence measures on hand-made frames: definitions, blanks, DATEADD and MoM."""

from datetime import date

import pandas as pd
import pytest

from dzmi_app import measures_sales as base
from dzmi_app import measures_sales_intelligence as si


def lines(rows):
    return pd.DataFrame(rows, columns=["date_key", "promotion_key", "quantity", "net_sales",
                                       "gross_profit", "category_name"]).astype(
        {"promotion_key": "Int64"})


@pytest.fixture(scope="module")
def sales_context():
    from dzmi_app.sales_intelligence import load_sales_context
    return load_sales_context()


SAMPLE = lines([
    (20250201, pd.NA, 2, 100.0, 20.0, "A"),
    (20250201, 1, 1, 50.0, 10.0, "A"),
    (20250202, pd.NA, 4, 200.0, 30.0, "B"),
    (20250203, 2, 3, 150.0, -5.0, "B"),
])


# ------------------------------------------------------------------ scalar measures


def test_average_selling_price_is_net_sales_over_units():
    assert si.average_selling_price(SAMPLE) == pytest.approx(500.0 / 10)


def test_promo_net_sales_counts_only_lines_with_a_promotion_key():
    assert si.promo_net_sales(SAMPLE) == 200.0
    assert si.promo_sales_share(SAMPLE) == pytest.approx(200.0 / 500.0)


def test_promo_measures_are_blank_without_promotional_lines():
    no_promo = SAMPLE[SAMPLE["promotion_key"].isna()]
    assert si.promo_net_sales(no_promo) is None
    assert si.promo_sales_share(no_promo) is None          # DIVIDE(BLANK, x) is BLANK, not 0%


def test_sales_days_counts_distinct_dates_not_lines():
    assert si.sales_days(SAMPLE) == 3
    assert si.average_daily_net_sales(SAMPLE) == pytest.approx(500.0 / 3)


def test_net_sales_per_sqm_uses_the_floor_area_of_the_context():
    assert si.net_sales_per_sqm(SAMPLE, 250) == pytest.approx(2.0)
    assert si.net_sales_per_sqm(SAMPLE, 0) is None
    assert si.net_sales_per_sqm(SAMPLE, None) is None


def test_every_measure_is_blank_on_an_empty_context():
    empty = SAMPLE.iloc[0:0]
    assert si.average_selling_price(empty) is None
    assert si.promo_net_sales(empty) is None
    assert si.promo_sales_share(empty) is None
    assert si.sales_days(empty) == 0
    assert si.average_daily_net_sales(empty) is None
    assert si.net_sales_per_sqm(empty, 100) is None


# ------------------------------------------------------------------ grouped measures


def test_measures_by_equals_the_scalar_measures_per_group():
    out = si.measures_by(SAMPLE, ["category_name"]).set_index("category_name")
    for cat, group in SAMPLE.groupby("category_name"):
        row = out.loc[cat]
        assert row["net_sales"] == base.net_sales(group)
        assert row["gross_profit"] == base.gross_profit(group)
        assert row["units_sold"] == base.units_sold(group)
        assert row["gross_margin"] == pytest.approx(base.gross_margin(group))
        assert row["avg_selling_price"] == pytest.approx(si.average_selling_price(group))
        assert row["sales_days"] == si.sales_days(group)
        assert row["avg_daily_net_sales"] == pytest.approx(si.average_daily_net_sales(group))


def test_measures_by_keeps_promo_share_blank_for_groups_without_promotions():
    out = si.measures_by(SAMPLE, ["category_name"]).set_index("category_name")
    assert out.loc["A", "promo_net_sales"] == 50.0 and out.loc["A", "promo_share"] == pytest.approx(50 / 150)
    assert out.loc["B", "promo_share"] == pytest.approx(150 / 350)
    only_plain = SAMPLE[SAMPLE["promotion_key"].isna()]
    assert si.measures_by(only_plain, ["category_name"])["promo_share"].isna().all()


def test_measures_by_gross_margin_is_blank_when_net_sales_sum_to_zero():
    zero = lines([(20250201, pd.NA, 1, 10.0, 1.0, "A"), (20250201, pd.NA, 1, -10.0, 1.0, "A")])
    assert si.measures_by(zero, ["category_name"])["gross_margin"].isna().all()


def test_measures_by_does_not_modify_its_input():
    before = SAMPLE.copy(deep=True)
    si.measures_by(SAMPLE, ["category_name"])
    pd.testing.assert_frame_equal(SAMPLE, before)


def test_measures_by_on_the_real_data_equals_scalar_measures(sales_context):
    sales = sales_context.sales
    out = si.measures_by(sales, ["category_name"]).set_index("category_name")
    assert len(out) == 14
    for cat, group in sales.groupby("category_name"):
        assert out.loc[cat, "net_sales"] == pytest.approx(base.net_sales(group), abs=0.01)
        assert out.loc[cat, "gross_margin"] == pytest.approx(base.gross_margin(group), rel=1e-12)
        assert out.loc[cat, "units_sold"] == base.units_sold(group)
        promo = si.promo_sales_share(group)
        if promo is None:
            assert pd.isna(out.loc[cat, "promo_share"])
        else:
            assert out.loc[cat, "promo_share"] == pytest.approx(promo, rel=1e-12)


# ------------------------------------------------------------------ DATEADD ( ..., -1, MONTH )


@pytest.mark.parametrize("start,end,expected", [
    (date(2025, 3, 1), date(2025, 3, 31), (date(2025, 2, 1), date(2025, 2, 28))),
    (date(2025, 4, 1), date(2025, 4, 30), (date(2025, 3, 1), date(2025, 3, 31))),   # month-end rule
    (date(2025, 2, 1), date(2025, 2, 28), (date(2025, 1, 1), date(2025, 1, 31))),
    (date(2025, 1, 1), date(2025, 1, 31), (date(2024, 12, 1), date(2024, 12, 31))),  # year wrap
    (date(2024, 3, 1), date(2024, 3, 31), (date(2024, 2, 1), date(2024, 2, 29))),   # leap year
    (date(2025, 3, 10), date(2025, 3, 20), (date(2025, 2, 10), date(2025, 2, 20))),
    (date(2025, 3, 29), date(2025, 3, 30), (date(2025, 2, 28), date(2025, 2, 28))),  # clamped
    (date(2025, 5, 1), date(2025, 5, 1), (date(2025, 4, 1), date(2025, 4, 1))),
    (date(2025, 2, 1), date(2025, 4, 30), (date(2025, 1, 1), date(2025, 3, 31))),   # several months
])
def test_dateadd_back_one_month(start, end, expected):
    assert si.dateadd_back_one_month(start, end) == expected


# ------------------------------------------------------------------ Net Sales MoM %


def calendar_frames(complete_months=("2025-02", "2025-03", "2025-04"), net_per_day=None):
    """dim_date rows 2025-02-01..2025-05-01 plus one sales line per day (net = 1 unless overridden)."""
    days = pd.date_range("2025-02-01", "2025-05-01")
    dates = pd.DataFrame({"full_date": days,
                          "is_complete_month": [int(d.strftime("%Y-%m") in complete_months) for d in days]})
    net = net_per_day or {}
    sales = pd.DataFrame({"full_date": days,
                          "net_sales": [net.get(d.strftime("%Y-%m"), 1.0) for d in days]})
    return dates, sales


def test_mom_compares_a_month_with_the_complete_previous_month():
    dates, sales = calendar_frames()
    assert si.net_sales_mom(dates, sales, 31.0, date(2025, 3, 1), date(2025, 3, 31)) \
        == pytest.approx((31 - 28) / 28)
    assert si.net_sales_mom(dates, sales, 30.0, date(2025, 4, 1), date(2025, 4, 30)) \
        == pytest.approx((30 - 31) / 31)                    # previous period is all of March


def test_mom_is_blank_when_the_current_month_is_incomplete():
    dates, sales = calendar_frames()
    assert si.net_sales_mom(dates, sales, 1.0, date(2025, 5, 1), date(2025, 5, 1)) is None


def test_mom_is_blank_when_the_previous_month_is_incomplete():
    dates, sales = calendar_frames(complete_months=("2025-02", "2025-04"))      # March flagged incomplete
    assert si.net_sales_mom(dates, sales, 30.0, date(2025, 4, 1), date(2025, 4, 30)) is None


def test_mom_is_blank_without_a_previous_month_in_the_data():
    dates, sales = calendar_frames()
    assert si.net_sales_mom(dates, sales, 28.0, date(2025, 2, 1), date(2025, 2, 28)) is None


def test_mom_is_blank_when_the_previous_month_has_no_sales_or_zero_sales():
    dates, sales = calendar_frames()
    no_feb = sales[sales["full_date"] >= "2025-03-01"]
    assert si.net_sales_mom(dates, no_feb, 31.0, date(2025, 3, 1), date(2025, 3, 31)) is None
    dates, zero_feb = calendar_frames(net_per_day={"2025-02": 0.0})
    assert si.net_sales_mom(dates, zero_feb, 31.0, date(2025, 3, 1), date(2025, 3, 31)) is None


def test_mom_treats_a_blank_current_month_as_zero_like_dax():
    dates, sales = calendar_frames()
    assert si.net_sales_mom(dates, sales, None, date(2025, 3, 1), date(2025, 3, 31)) == pytest.approx(-1.0)


def test_mom_prior_month_is_not_limited_by_the_selected_date_range():
    dates, sales = calendar_frames()
    # Only March 10-20 selected: the comparison period is Feb 10-20, taken from the unfiltered sales.
    assert si.net_sales_mom(dates, sales, 11.0, date(2025, 3, 10), date(2025, 3, 20)) \
        == pytest.approx((11 - 11) / 11)


def test_mom_ignores_missing_dates_of_the_calendar():
    dates, sales = calendar_frames()
    assert si.net_sales_mom(dates.iloc[0:0], sales, 31.0, date(2025, 3, 1), date(2025, 3, 31)) is None


def test_sort_by_mom_descending_puts_blanks_last_and_keeps_month_order_for_ties():
    frame = pd.DataFrame({"year_month_label": ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
                          "mom": [None, 0.5, -0.2, None, 0.1]})
    out = si.sort_by_mom_desc(frame)
    assert list(out["year_month_label"]) == ["2025-03", "2025-06", "2025-04", "2025-02", "2025-05"]
