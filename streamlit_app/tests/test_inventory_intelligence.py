"""Inventory Intelligence calculations. Every scenario is compared with oracle_inventory.py, produced by
plain SQL on the CSVs (tests/oracle_generator_inventory.py), not by dzmi_app."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import oracle_inventory as ov
from dzmi_app.inventory_intelligence import (
    InventoryFilters, build_inventory_context, compute_inventory_intelligence, filter_inventory,
    filter_risk, last_date_in_selection, load_inventory_context, risk_list,
)

NAMES = list(ov.SCENARIOS)
REL = 1e-9


@pytest.fixture(scope="module")
def inventory_context():
    return load_inventory_context()


def filters_for(name):
    sc = ov.SCENARIOS[name]
    return InventoryFilters(
        date_start=date.fromisoformat(sc["start"]) if "start" in sc else None,
        date_end=date.fromisoformat(sc["end"]) if "end" in sc else None,
        stores=tuple(sc.get("stores", ())), categories=tuple(sc.get("categories", ())),
        products=tuple(sc.get("products", ())), tiers=tuple(sc.get("tiers", ())))


@pytest.fixture(scope="module")
def results(inventory_context):
    return {n: compute_inventory_intelligence(inventory_context, filters_for(n)) for n in NAMES}


def close(actual, expected, **tol):
    tol = tol or {"rel": REL}
    if expected is None:
        assert actual is None or pd.isna(actual)
    else:
        assert actual == pytest.approx(expected, **tol)


# ------------------------------------------------------------------ context and relationships


def test_context_rows_and_no_sales_columns(inventory_context):
    c = inventory_context
    assert len(c.inventory) == 37_710 and len(c.risk) == 419
    assert not set(c.inventory.columns) & {"quantity", "net_sales", "gross_profit", "transaction_id"}
    assert not c.inventory[["full_date", "store_name", "category_name", "product_name"]].isna().any().any()
    assert not c.risk[["store_name", "category_name", "product_name"]].isna().any().any()


def test_slicer_options(inventory_context):
    c = inventory_context
    assert c.stores == ("Alger Centre", "Blida Ville", "Constantine Centre", "Oran Es Senia", "S\u00e9tif El Hidhab")
    assert len(c.categories) == 14 and len(c.products) == 100 and list(c.products) == sorted(c.products)
    assert c.tiers == ("Stockout", "Critical", "Low", "Healthy")           # tiers present, in risk_tier_order
    assert c.snapshot_date == date(2025, 5, 1)
    assert c.coverage_text == "Synthetic data loaded: 2025-02-01 to 2025-05-01"


def test_context_is_cached_and_rebuildable(dataset, inventory_context):
    assert load_inventory_context() is inventory_context
    assert len(build_inventory_context(dataset).inventory) == 37_710


def test_store_filter_reaches_inventory_and_risk(inventory_context):
    f = InventoryFilters(stores=("Alger Centre",))
    assert set(filter_inventory(inventory_context, f)["store_name"]) == {"Alger Centre"}
    assert set(filter_risk(inventory_context, f)["store_name"]) == {"Alger Centre"}


def test_category_and_product_filters_reach_inventory_and_risk(inventory_context):
    f = InventoryFilters(categories=("Dairy & Eggs",), products=("Oranges 2 kg (Brand B)",))
    assert len(filter_inventory(inventory_context, f)) == 0                  # AND: the orange is not dairy
    f = InventoryFilters(products=("Oranges 2 kg (Brand B)",))
    assert set(filter_risk(inventory_context, f)["product_name"]) == {"Oranges 2 kg (Brand B)"}
    assert len(filter_risk(inventory_context, f)) == 5


def test_the_date_range_never_reaches_the_risk_snapshot(inventory_context):
    f = InventoryFilters(date_start=date(2025, 3, 1), date_end=date(2025, 3, 2))
    assert len(filter_risk(inventory_context, f)) == 419
    assert filter_inventory(inventory_context, f)["full_date"].nunique() == 2


def test_the_tier_filter_applies_only_when_asked_for_the_sku_list(inventory_context):
    f = InventoryFilters(tiers=("Critical",))
    assert len(filter_risk(inventory_context, f)) == 419
    assert len(filter_risk(inventory_context, f, with_tiers=True)) == 83


def test_last_date_of_the_selection(inventory_context):
    c = inventory_context
    assert last_date_in_selection(c, InventoryFilters()) == date(2025, 5, 1)
    assert last_date_in_selection(c, InventoryFilters(date_start=date(2025, 3, 1), date_end=date(2025, 3, 29))) == date(2025, 3, 29)
    assert last_date_in_selection(c, InventoryFilters(date_start=date(2025, 6, 1), date_end=date(2025, 6, 2))) is None


# ------------------------------------------------------------------ scenarios vs the independent oracle


@pytest.mark.parametrize("name", NAMES)
def test_rates_and_counts(results, name):
    r, e = results[name], ov.ORACLES[name]
    close(r.stockout_rate, e["stockout_rate"])
    close(r.availability_rate, e["availability_rate"])
    assert r.sku_days == e["sku_days"]
    if e["stockout_sku_days"] is not None:
        assert r.stockout_sku_days == e["stockout_sku_days"]


@pytest.mark.parametrize("name", NAMES)
def test_closing_stock_and_days_of_cover(results, name):
    r, e = results[name], ov.ORACLES[name]
    assert (str(r.last_date) if r.last_date else None) == e["last_date"]
    assert r.closing_stock == e["closing_stock"]
    close(r.avg_daily_units_14d, e["avg_daily_units_14d"])
    close(r.overall_days_of_cover, e["days_of_cover"])
    close(r.median_days_of_cover, e["median_days_of_cover"])


@pytest.mark.parametrize("name", NAMES)
def test_snapshot_measures(results, name):
    r, e = results[name], ov.ORACLES[name]
    close(r.at_risk_share, e["at_risk_share"])
    assert r.tracked_skus == e["tracked_skus"]
    assert {t: int(n) for t, n in zip(r.tiers["risk_tier"], r.tiers["skus"]) if n} == e["tiers"]
    assert list(r.tiers["risk_tier"]) == ["Stockout", "Critical", "Low", "Healthy", "Overstock", "No recent sales"]


@pytest.mark.parametrize("name", NAMES)
def test_daily_series(results, name):
    d, e = results[name].daily, ov.ORACLES[name]["daily"]
    assert list(d["full_date"].dt.strftime("%Y-%m-%d")) == [x[0] for x in e]
    assert list(d["stockout_rate"]) == pytest.approx([x[1] for x in e], rel=REL)


@pytest.mark.parametrize("name", NAMES)
def test_category_by_store_matrix(results, name):
    h, e = results[name].heatmap, ov.ORACLES[name]["heatmap"]
    got = [(c, s, float(h.loc[c, s])) for c in h.index for s in h.columns if not np.isnan(h.loc[c, s])]
    assert [(a, b) for a, b, _ in got] == [(a, b) for a, b, _ in e]
    assert [v for _, _, v in got] == pytest.approx([v for _, _, v in e], rel=REL)


@pytest.mark.parametrize("name", NAMES)
def test_sku_risk_list(results, name):
    t, e = results[name].risk_table, ov.ORACLES[name]["risk_table"]
    assert len(t) == len(e)
    for row, x in zip(t.itertuples(index=False), e):
        assert (row.risk_tier, row.store_code, row.sku, row.product_name, int(row.closing_stock)) == x[:5]
        close(None if pd.isna(row.avg_daily_units_14d) else row.avg_daily_units_14d, x[5])
        close(None if pd.isna(row.days_of_cover) else row.days_of_cover, x[6])
        close(row.stock_value_at_current_cost, x[7], abs=0.01)


# ------------------------------------------------------------------ audit values and business rules


def test_audit_default_values(results):
    r = results["default"]
    assert r.stockout_sku_days == 794 and r.sku_days == 37_710
    assert r.stockout_rate == pytest.approx(794 / 37_710, rel=1e-12)
    assert r.closing_stock == 7_291
    assert r.overall_days_of_cover == pytest.approx(4.3031, abs=5e-5)
    assert r.median_days_of_cover == pytest.approx(5.3529, abs=5e-5)
    assert r.at_risk_share == pytest.approx(280 / 419, rel=1e-12)
    assert list(r.tiers["skus"]) == [4, 83, 193, 139, 0, 0]


def test_both_days_of_cover_definitions_are_exposed_and_differ(results):
    r = results["default"]
    assert r.overall_days_of_cover is not None and r.median_days_of_cover is not None
    assert r.overall_days_of_cover != pytest.approx(r.median_days_of_cover, abs=0.5)


def test_closing_stock_is_one_day_never_a_sum_over_dates(inventory_context, results):
    total_over_all_dates = int(inventory_context.inventory["closing_stock"].sum())
    assert results["default"].closing_stock == 7_291 < total_over_all_dates
    assert results["ramadan_window"].closing_stock == 6_313              # 2025-03-29 only
    assert results["first_days"].closing_stock == 7_427                  # 2025-02-05 only


def test_snapshot_measures_ignore_the_date_range(results):
    base = results["default"]
    for name in ("ramadan_window", "last_day_only", "first_days"):
        r = results[name]
        pd.testing.assert_frame_equal(r.tiers, base.tiers)
        pd.testing.assert_frame_equal(r.risk_table, base.risk_table)
        assert r.median_days_of_cover == base.median_days_of_cover and r.at_risk_share == base.at_risk_share


def test_date_range_does_change_the_rates_and_the_cover_ratio(results):
    assert results["ramadan_window"].stockout_rate > results["default"].stockout_rate
    assert results["last_day_only"].stockout_rate < results["default"].stockout_rate
    assert results["ramadan_window"].overall_days_of_cover != results["default"].overall_days_of_cover


def test_the_14_day_window_is_shorter_at_the_start_of_the_data(results):
    # Feb 1-5: only five calendar days exist before the last date, so only those enter the average.
    e = ov.ORACLES["first_days"]
    assert results["first_days"].avg_daily_units_14d == pytest.approx(e["avg_daily_units_14d"], rel=REL)


def test_the_tier_filter_changes_only_the_sku_list(results):
    f, d = results["tier_filter"], results["default"]
    assert len(f.risk_table) == 276 and set(f.risk_table["risk_tier"]) == {"Critical", "Low"}
    assert f.tracked_skus == d.tracked_skus and f.at_risk_share == d.at_risk_share
    pd.testing.assert_frame_equal(f.tiers, d.tiers)
    assert f.stockout_rate == d.stockout_rate


def test_availability_plus_stockout_rate_is_one_with_the_same_filters(results):
    for name, r in results.items():
        if r.sku_days:
            assert r.availability_rate + r.stockout_rate == pytest.approx(1.0, abs=1e-12), name


def test_sku_list_is_sorted_by_tier_then_days_of_cover(results):
    t = results["default"].risk_table
    order = {"Stockout": 1, "Critical": 2, "Low": 3, "Healthy": 4}
    assert list(t["risk_tier"].map(order)) == sorted(t["risk_tier"].map(order))
    assert t.iloc[0]["risk_tier"] == "Stockout"
    for _, g in t.groupby("risk_tier", sort=False):
        assert g["days_of_cover"].is_monotonic_increasing


def test_sku_list_puts_blank_cover_last_within_a_tier():
    frame = pd.DataFrame({
        "risk_tier": ["Low", "Low", "Low"], "risk_tier_order": [3, 3, 3], "store_code": ["A", "B", "C"],
        "sku": ["s1", "s2", "s3"], "product_name": ["p"] * 3, "closing_stock": [1, 2, 3],
        "avg_daily_units_14d": [1.0, np.nan, 1.0], "days_of_cover": [5.0, np.nan, 2.0],
        "stock_value_at_current_cost": [1.0, 2.0, 3.0]})
    assert list(risk_list(frame)["sku"]) == ["s3", "s1", "s2"]


def test_sums_are_consistent_across_the_visuals(inventory_context, results):
    for name, r in results.items():
        if r.sku_days == 0:
            continue
        assert int(r.tiers["skus"].sum()) == r.tracked_skus
        sku_days_by_date = filter_inventory(inventory_context, filters_for(name)).groupby("full_date").size()
        stockout_total = (r.daily.set_index("full_date")["stockout_rate"] * sku_days_by_date).sum()
        assert stockout_total == pytest.approx(r.stockout_sku_days, abs=1e-6), name


def test_executive_overview_stock_figures_agree_with_this_page(overview_context, results):
    from dzmi_app.filters import OverviewFilters
    from dzmi_app.overview import compute_overview
    k = compute_overview(overview_context, OverviewFilters()).kpis
    r = results["default"]
    assert k.stockout_rate == pytest.approx(r.stockout_rate, rel=1e-12)
    assert (k.stockout_now, k.critical, k.low) == tuple(int(x) for x in r.tiers["skus"][:3])


# ------------------------------------------------------------------ empty and missing values


def test_empty_selection_blank_versus_zero(results):
    r = results["empty_selection"]
    assert r.sku_days == 0 and r.stockout_sku_days == 0 and r.tracked_skus == 0     # counts are zero ...
    for blank in (r.stockout_rate, r.availability_rate, r.closing_stock, r.avg_daily_units_14d,
                  r.overall_days_of_cover, r.median_days_of_cover, r.at_risk_share):
        assert blank is None                                                       # ... ratios are blank
    assert r.last_date == date(2025, 5, 1)       # the date range itself is not empty
    assert r.daily.empty and r.heatmap.empty and r.risk_table.empty
    assert list(r.tiers["skus"]) == [0, 0, 0, 0, 0, 0]


def test_computing_does_not_modify_the_shared_frames(inventory_context):
    def fingerprint():
        return (int(pd.util.hash_pandas_object(inventory_context.inventory, index=True).sum()),
                int(pd.util.hash_pandas_object(inventory_context.risk[["risk_tier", "days_of_cover"]], index=True).sum()))
    before = fingerprint()
    for n in NAMES:
        compute_inventory_intelligence(inventory_context, filters_for(n))
    assert fingerprint() == before
