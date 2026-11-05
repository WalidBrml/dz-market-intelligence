"""Sales measures (ported DAX) on small hand-made frames and on the real data."""

import pandas as pd
import pytest

from dzmi_app import measures_sales as m
import oracle_overview as ov


def frame(rows):
    return pd.DataFrame(rows, columns=["transaction_id", "net_sales", "gross_profit", "quantity",
                                       "store_name", "product_name", "year_month_label"])


SAMPLE = frame([
    (1, 100.0, 20.0, 2, "A", "p1", "2025-02"),
    (1, 50.0, 10.0, 1, "A", "p2", "2025-02"),     # same basket, second line
    (2, 200.0, 30.0, 4, "B", "p1", "2025-03"),
    (3, 150.0, -5.0, 3, "B", "p3", "2025-03"),
])


# ------------------------------------------------------------------ DIVIDE and blanks


@pytest.mark.parametrize("n,d,expected", [
    (10, 4, 2.5), (0, 5, 0.0), (10, 0, None), (10, None, None), (None, 5, None), (None, None, None),
])
def test_divide_mirrors_dax_divide(n, d, expected):
    assert m.divide(n, d) == expected


def test_sums_are_blank_when_there_are_no_rows():
    empty = SAMPLE.iloc[0:0]
    assert m.net_sales(empty) is None
    assert m.gross_profit(empty) is None
    assert m.units_sold(empty) is None
    assert m.gross_margin(empty) is None
    assert m.average_basket_value(empty) is None
    assert m.transactions(empty) == 0


# ------------------------------------------------------------------ definitions


def test_basic_measures_on_a_known_sample():
    assert m.net_sales(SAMPLE) == 500.0
    assert m.gross_profit(SAMPLE) == 55.0
    assert m.units_sold(SAMPLE) == 10
    assert m.transactions(SAMPLE) == 3                       # distinct baskets, not lines
    assert m.gross_margin(SAMPLE) == pytest.approx(55.0 / 500.0)
    assert m.average_basket_value(SAMPLE) == pytest.approx(500.0 / 3)


def test_negative_gross_profit_lines_are_kept():
    assert m.gross_profit(SAMPLE[SAMPLE["transaction_id"] == 3]) == -5.0
    assert m.gross_margin(SAMPLE[SAMPLE["transaction_id"] == 3]) == pytest.approx(-5.0 / 150.0)


def test_transactions_are_not_additive_across_groups():
    # A basket is counted in every group it touches, so group counts do not add up to the total.
    a = SAMPLE[SAMPLE["product_name"] == "p1"]
    b = SAMPLE[SAMPLE["product_name"] == "p2"]
    assert m.transactions(a) + m.transactions(b) == 3
    assert m.transactions(SAMPLE[SAMPLE["product_name"].isin(["p1", "p2"])]) == 2


def test_monthly_net_sales_is_ascending_and_skips_empty_months():
    out = m.monthly_net_sales(SAMPLE)
    assert list(out["year_month_label"]) == ["2025-02", "2025-03"]
    assert list(out["net_sales"]) == [150.0, 350.0]
    shuffled = SAMPLE.sample(frac=1.0, random_state=3)
    assert list(m.monthly_net_sales(shuffled)["year_month_label"]) == ["2025-02", "2025-03"]


def test_net_sales_by_store_is_descending_with_name_tie_break():
    tie = frame([(1, 10.0, 1.0, 1, "Zed", "p", "2025-02"), (2, 10.0, 1.0, 1, "Abe", "p", "2025-02"),
                 (3, 99.0, 1.0, 1, "Mid", "p", "2025-02")])
    out = m.net_sales_by_store(tie)
    assert list(out["store_name"]) == ["Mid", "Abe", "Zed"]


def test_top_products_returns_the_n_best_in_order():
    rows = [(i, float(i * 10), 1.0, 1, "S", f"p{i:02d}", "2025-02") for i in range(1, 16)]
    out = m.top_products_by_net_sales(frame(rows), 10)
    assert len(out) == 10
    assert list(out["product_name"]) == [f"p{i:02d}" for i in range(15, 5, -1)]
    assert out["net_sales"].is_monotonic_decreasing


def test_top_products_keeps_every_row_tied_with_the_tenth_like_topn():
    rows = [(i, float(i * 10), 1.0, 1, "S", f"p{i:02d}", "2025-02") for i in range(1, 16)]
    rows.append((99, 60.0, 1.0, 1, "S", "tie", "2025-02"))        # ties product p06 at 10th place
    out = m.top_products_by_net_sales(frame(rows), 10)
    assert len(out) == 11
    assert {"p06", "tie"} <= set(out["product_name"])


def test_top_products_with_fewer_than_n_products_returns_them_all():
    out = m.top_products_by_net_sales(SAMPLE, 10)
    assert set(out["product_name"]) == {"p1", "p2", "p3"}
    assert out.iloc[0]["product_name"] == "p1" and out.iloc[0]["net_sales"] == 300.0


def test_measures_do_not_modify_their_input():
    before = SAMPLE.copy(deep=True)
    for fn in (m.net_sales, m.gross_profit, m.units_sold, m.transactions, m.gross_margin,
               m.average_basket_value, m.monthly_net_sales, m.net_sales_by_store,
               m.top_products_by_net_sales):
        fn(SAMPLE)
    pd.testing.assert_frame_equal(SAMPLE, before)


# ------------------------------------------------------------------ real data (audit values)


@pytest.fixture(scope="module")
def sales(overview_context):
    return overview_context.sales


def test_audit_totals(sales):
    e = ov.ORACLES["default"]
    assert m.net_sales(sales) == pytest.approx(e["net_sales"], abs=0.01)
    assert m.net_sales(sales) == pytest.approx(40_389_951.00, abs=0.01)
    assert m.gross_profit(sales) == pytest.approx(6_842_371.34, abs=0.01)
    assert m.units_sold(sales) == 165_893
    assert m.transactions(sales) == 5_776
    assert m.gross_margin(sales) == pytest.approx(6_842_371.34 / 40_389_951.00, rel=1e-12)
    assert m.average_basket_value(sales) == pytest.approx(6992.72, abs=0.005)


def test_audit_monthly_net_sales(sales):
    out = m.monthly_net_sales(sales)
    assert list(out["year_month_label"]) == ["2025-02", "2025-03", "2025-04", "2025-05"]
    assert list(out["net_sales"]) == pytest.approx(
        [10_803_630.50, 16_978_045.50, 12_188_345.00, 419_930.00], abs=0.01)


def test_audit_top_product(sales):
    top = m.top_products_by_net_sales(sales, 10)
    assert len(top) == 10
    assert top.iloc[0]["product_name"] == "Oranges 2 kg (Brand B)"
    assert top.iloc[0]["net_sales"] == pytest.approx(3_802_490.0, abs=0.01)
