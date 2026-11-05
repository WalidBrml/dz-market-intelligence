"""Inventory measures (ported DAX) incl. the V2 correction of the tier counts."""

import pandas as pd
import pytest

from dzmi_app import measures_inventory as m
import oracle_overview as ov


def inv(rows):
    return pd.DataFrame(rows, columns=["store_name", "closing_stock"])


def risk(tiers, values=None):
    values = values or [10.0] * len(tiers)
    return pd.DataFrame({"risk_tier": tiers, "stock_value_at_current_cost": values})


def test_stockout_is_defined_by_zero_closing_stock_only():
    # No stockout_flag column is needed (nor read): the definition is closing_stock = 0.
    f = inv([("A", 0), ("A", 5), ("A", 0), ("B", 3)])
    assert m.sku_days(f) == 4
    assert m.stockout_sku_days(f) == 2
    assert m.stockout_rate(f) == pytest.approx(0.5)


def test_stockout_rate_is_blank_without_sku_days_and_zero_without_stockouts():
    assert m.stockout_rate(inv([])) is None
    assert m.stockout_rate(inv([("A", 4), ("A", 9)])) == 0.0


def test_stockout_rate_by_store_orders_by_rate_then_name():
    f = inv([("Zed", 0), ("Zed", 1), ("Abe", 0), ("Abe", 2), ("Mid", 0), ("Mid", 0)])
    out = m.stockout_rate_by_store(f)
    assert list(out["store_name"]) == ["Mid", "Abe", "Zed"]       # 1.0, then 0.5 tie by name
    assert list(out["stockout_rate"]) == [1.0, 0.5, 0.5]


def test_rate_by_store_is_a_ratio_of_sums_not_a_mean_of_rates():
    f = inv([("A", 0)] + [("B", 1)] * 3 + [("B", 0)])
    assert m.stockout_rate(f) == pytest.approx(2 / 5)
    assert dict(zip(*[m.stockout_rate_by_store(f)[c] for c in ("store_name", "stockout_rate")])) \
        == {"A": 1.0, "B": 0.25}


def test_tier_counts_cover_all_six_tiers_in_order_with_zeros():
    out = m.risk_tier_counts(risk(["Low", "Low", "Critical", "Healthy"]))
    assert list(out["risk_tier"]) == ["Stockout", "Critical", "Low", "Healthy", "Overstock", "No recent sales"]
    assert list(out["skus"]) == [0, 1, 2, 1, 0, 0]
    assert list(out["risk_tier_order"]) == [1, 2, 3, 4, 5, 6]
    assert m.TIERS == tuple(out["risk_tier"])


def test_tier_counts_of_an_empty_selection_are_six_zero_rows():
    out = m.risk_tier_counts(risk([]))
    assert len(out) == 6 and out["skus"].sum() == 0


def test_tier_count_and_tracked_skus():
    r = risk(["Stockout", "Critical", "Critical", "Healthy"])
    assert m.tier_count(r, "Critical") == 2
    assert m.tier_count(r, "Low") == 0
    assert m.tracked_skus(r) == 4


def test_stock_value_sums_and_is_blank_when_empty():
    assert m.stock_value_at_cost(risk(["Low", "Low"], [1.5, 2.25])) == pytest.approx(3.75)
    assert m.stock_value_at_cost(risk([])) is None


def test_v2_tier_counts_are_not_flat():
    """Power BI's Tracked SKUs (REMOVEFILTERS on risk_tier) drew the same total on every bar."""
    out = m.risk_tier_counts(risk(["Stockout"] * 4 + ["Critical"] * 83 + ["Low"] * 193 + ["Healthy"] * 139))
    assert list(out["skus"]) == [4, 83, 193, 139, 0, 0]
    assert out["skus"].nunique() > 1
    assert out["skus"].sum() == m.tracked_skus(risk(["x"] * 419))


# ------------------------------------------------------------------ real data (audit values)


def test_audit_stockout_figures(overview_context):
    f = overview_context.inventory
    assert m.sku_days(f) == 37_710
    assert m.stockout_sku_days(f) == 794
    assert m.stockout_rate(f) == pytest.approx(794 / 37_710, rel=1e-12)


def test_audit_tier_counts_and_stock_value(overview_context):
    r = overview_context.risk
    assert dict(zip(m.risk_tier_counts(r)["risk_tier"], m.risk_tier_counts(r)["skus"])) == {
        "Stockout": 4, "Critical": 83, "Low": 193, "Healthy": 139, "Overstock": 0, "No recent sales": 0}
    assert m.tracked_skus(r) == 419
    assert m.stock_value_at_cost(r) == pytest.approx(1_608_182.54, abs=0.01)


def test_stockout_rate_by_store_matches_the_independent_oracle(overview_context):
    out = m.stockout_rate_by_store(overview_context.inventory)
    e = ov.ORACLES["default"]["stockout_by_store"]
    assert list(out["store_name"]) == [name for name, _ in e]
    assert list(out["stockout_rate"]) == pytest.approx([v for _, v in e], rel=1e-12)
