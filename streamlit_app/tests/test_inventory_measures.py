"""Inventory Intelligence measures (ported DAX) on hand-made frames."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dzmi_app import measures_inventory_intelligence as ii


def inv(rows):
    return pd.DataFrame(rows, columns=["full_date", "date_key", "category_name", "store_name",
                                       "closing_stock", "units_sold"]).astype({"full_date": "datetime64[ns]"})


def D(day):
    return pd.Timestamp(2025, 2, day)


FRAME = inv([
    (D(1), 20250201, "A", "S1", 0, 3), (D(1), 20250201, "A", "S2", 5, 1),
    (D(2), 20250202, "A", "S1", 4, 2), (D(2), 20250202, "B", "S1", 0, 0),
])


# ------------------------------------------------------------------ availability and stockout


def test_availability_is_one_minus_stockout_rate_in_the_same_context():
    assert ii.availability_rate(FRAME) == pytest.approx(0.5)
    from dzmi_app.measures_inventory import stockout_rate
    assert ii.availability_rate(FRAME) + stockout_rate(FRAME) == pytest.approx(1.0)


def test_availability_is_blank_without_sku_days_and_one_without_stockouts():
    assert ii.availability_rate(FRAME.iloc[0:0]) is None
    assert ii.availability_rate(FRAME[FRAME["closing_stock"] > 0]) == 1.0


# ------------------------------------------------------------------ closing stock: one date, never summed over dates


def test_closing_stock_reads_only_the_last_date():
    assert ii.closing_stock_units(FRAME, date(2025, 2, 2)) == 4
    assert ii.closing_stock_units(FRAME, date(2025, 2, 1)) == 5
    assert ii.closing_stock_units(FRAME, date(2025, 2, 2)) != int(FRAME["closing_stock"].sum())


def test_closing_stock_is_blank_when_the_last_date_has_no_rows_or_is_missing():
    assert ii.closing_stock_units(FRAME, date(2025, 3, 1)) is None
    assert ii.closing_stock_units(FRAME, None) is None
    assert ii.closing_stock_units(FRAME.iloc[0:0], date(2025, 2, 1)) is None


def test_a_zero_closing_stock_is_zero_not_blank():
    only_zero = FRAME[(FRAME["store_name"] == "S1") & (FRAME["full_date"] == D(1))]
    assert ii.closing_stock_units(only_zero, date(2025, 2, 1)) == 0


# ------------------------------------------------------------------ 14-day average and days of cover


def window_frame():
    days = pd.date_range("2025-02-01", "2025-02-20")
    return pd.DataFrame({"full_date": days, "date_key": [int(d.strftime("%Y%m%d")) for d in days],
                         "units_sold": range(1, 21), "closing_stock": 1})


def test_average_daily_units_uses_exactly_the_14_days_ending_on_the_last_date():
    f = window_frame()
    # days 7..20 (14 days): sum = 7+...+20 = 189, 14 distinct dates
    assert ii.avg_daily_units_14d(f, date(2025, 2, 20)) == pytest.approx(189 / 14)


def test_the_window_is_shorter_at_the_start_of_the_data():
    f = window_frame()
    assert ii.avg_daily_units_14d(f, date(2025, 2, 5)) == pytest.approx((1 + 2 + 3 + 4 + 5) / 5)


def test_the_window_is_not_limited_by_the_start_of_the_date_range():
    # The frame handed over is already free of the date filter (DATESINPERIOD replaces it).
    assert ii.avg_daily_units_14d(window_frame(), date(2025, 2, 20)) == pytest.approx(189 / 14)


def test_average_is_blank_without_rows_in_the_window():
    assert ii.avg_daily_units_14d(window_frame(), date(2025, 4, 1)) is None
    assert ii.avg_daily_units_14d(window_frame(), None) is None


def test_days_of_cover_is_closing_stock_over_average_daily_units():
    assert ii.days_of_cover(100, 20.0) == pytest.approx(5.0)
    assert ii.days_of_cover(0, 20.0) == 0.0                  # zero stock is zero days, not blank
    assert ii.days_of_cover(100, 0.0) is None                # no sales: undefined, not infinite
    assert ii.days_of_cover(None, 20.0) is None
    assert ii.days_of_cover(100, None) is None


# ------------------------------------------------------------------ risk snapshot measures


def risk(tiers, covers=None):
    return pd.DataFrame({"risk_tier": tiers, "days_of_cover": covers if covers is not None else [1.0] * len(tiers)})


def test_at_risk_share_counts_stockout_critical_and_low_only():
    r = risk(["Stockout", "Critical", "Low", "Healthy", "Healthy", "Overstock", "No recent sales", "Low"])
    assert ii.at_risk_sku_share(r) == pytest.approx(4 / 8)


def test_at_risk_share_is_blank_for_an_empty_snapshot_and_zero_without_risky_skus():
    assert ii.at_risk_sku_share(risk([])) is None
    assert ii.at_risk_sku_share(risk(["Healthy", "Healthy"])) == 0.0


def test_median_days_of_cover_counts_zero_and_skips_blanks():
    r = risk(["Stockout", "Low", "Healthy", "Healthy"], [0.0, 4.0, np.nan, 10.0])
    assert ii.median_sku_days_of_cover(r) == pytest.approx(4.0)          # median of 0, 4, 10
    assert ii.median_sku_days_of_cover(risk(["Low"], [np.nan])) is None
    assert ii.median_sku_days_of_cover(risk([])) is None


def test_median_differs_from_the_aggregate_ratio_by_construction():
    r = risk(["a", "b", "c"], [1.0, 2.0, 30.0])
    assert ii.median_sku_days_of_cover(r) == 2.0
    assert ii.days_of_cover(33, 3.0) == pytest.approx(11.0)


# ------------------------------------------------------------------ daily series and category x store matrix


def test_daily_rate_is_a_ratio_of_counts_per_date_sorted_ascending():
    out = ii.stockout_rate_by_day(FRAME.sample(frac=1.0, random_state=1))
    assert list(out["full_date"]) == [D(1), D(2)]
    assert list(out["stockout_rate"]) == pytest.approx([0.5, 0.5])
    assert ii.stockout_rate_by_day(FRAME.iloc[0:0]).empty


def test_matrix_cells_are_blank_when_a_pair_has_no_sku_days():
    m = ii.stockout_rate_matrix(FRAME)
    assert list(m.index) == ["A", "B"] and list(m.columns) == ["S1", "S2"]
    assert m.loc["A", "S1"] == pytest.approx(0.5) and m.loc["A", "S2"] == 0.0
    assert m.loc["B", "S1"] == 1.0
    assert np.isnan(m.loc["B", "S2"])                          # no pair: blank, not 0


def test_measures_do_not_modify_their_input():
    before = FRAME.copy(deep=True)
    ii.availability_rate(FRAME); ii.closing_stock_units(FRAME, date(2025, 2, 2))
    ii.avg_daily_units_14d(FRAME, date(2025, 2, 2)); ii.stockout_rate_by_day(FRAME); ii.stockout_rate_matrix(FRAME)
    pd.testing.assert_frame_equal(FRAME, before)
