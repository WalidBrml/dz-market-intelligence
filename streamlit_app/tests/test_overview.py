"""Executive Overview calculations: every scenario is checked against oracle_overview.py,
which was produced by plain SQL on the CSVs (tests/oracle_generator_b1.py), not by dzmi_app."""

from datetime import date

import pandas as pd
import pytest

import oracle_overview as ov
from dzmi_app import measures_inventory as inv_m
from dzmi_app.filters import OverviewFilters
from dzmi_app.overview import (
    build_overview_context, compute_overview, filter_inventory, filter_risk, filter_sales,
    load_overview_context,
)

SCENARIO_NAMES = list(ov.SCENARIOS)


def filters_for(name: str) -> OverviewFilters:
    sc = ov.SCENARIOS[name]
    return OverviewFilters(
        date_start=date.fromisoformat(sc["start"]) if "start" in sc else None,
        date_end=date.fromisoformat(sc["end"]) if "end" in sc else None,
        stores=tuple(sc.get("stores", ())),
        categories=tuple(sc.get("categories", ())),
        store_formats=tuple(sc.get("formats", ())),
    )


def close(actual, expected, **tol):
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, **tol)


# ------------------------------------------------------------------ context


def test_context_has_one_row_per_fact_row(overview_context):
    assert len(overview_context.sales) == 48_216
    assert len(overview_context.inventory) == 37_710
    assert len(overview_context.risk) == 419


def test_context_joins_leave_no_missing_attributes(overview_context):
    c = overview_context
    for frame, cols in ((c.sales, ["full_date", "year_month_label", "store_name", "store_format",
                                   "product_name", "category_name"]),
                        (c.inventory, ["full_date", "store_name", "store_format", "category_name"]),
                        (c.risk, ["store_name", "store_format", "category_name"])):
        assert not frame[cols].isna().any().any()


def test_slicer_options_are_sorted_and_complete(overview_context):
    c = overview_context
    assert c.stores == ("Alger Centre", "Blida Ville", "Constantine Centre", "Oran Es Senia",
                        "S\u00e9tif El Hidhab")
    assert c.store_formats == ("Hypermarket", "Supermarket")
    assert len(c.categories) == 14 and list(c.categories) == sorted(c.categories)
    assert "Bakery & Pastry" in c.categories


def test_data_coverage_ignores_filters_and_uses_the_original_wording(overview_context):
    assert overview_context.coverage_text == "Synthetic data loaded: 2025-02-01 to 2025-05-01"
    assert (overview_context.date_min, overview_context.date_max) == (date(2025, 2, 1), date(2025, 5, 1))


def test_snapshot_date_is_the_last_loaded_day(overview_context):
    assert overview_context.snapshot_date == date(2025, 5, 1)


def test_context_is_cached(overview_context):
    assert load_overview_context() is overview_context


def test_context_can_be_rebuilt_from_the_dataset(dataset, overview_context):
    rebuilt = build_overview_context(dataset)
    assert rebuilt.coverage_text == overview_context.coverage_text
    assert len(rebuilt.sales) == len(overview_context.sales)


# ------------------------------------------------------------------ scenarios vs independent oracles


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_kpis_match_the_oracle(overview_context, name):
    k, e = compute_overview(overview_context, filters_for(name)).kpis, ov.ORACLES[name]
    close(k.net_sales, e["net_sales"], abs=0.01)
    close(k.gross_profit, e["gross_profit"], abs=0.01)
    assert k.units_sold == e["units_sold"]
    close(k.gross_margin, e["gross_margin"], rel=1e-12)
    close(k.average_basket, e["average_basket"], rel=1e-12)
    close(k.stockout_rate, e["stockout_rate"], rel=1e-12)
    close(k.stock_value, e["stock_value"], abs=0.01)
    assert (k.stockout_now, k.critical, k.low) == (
        e["tiers"].get("Stockout", 0), e["tiers"].get("Critical", 0), e["tiers"].get("Low", 0))


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_row_counts_match_the_oracle(overview_context, name):
    r, e = compute_overview(overview_context, filters_for(name)), ov.ORACLES[name]
    assert r.sales_rows == e["sales_rows"]
    assert r.inventory_rows == e["sku_days"]
    assert r.risk_rows == e["tracked_skus"]


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_charts_match_the_oracle(overview_context, name):
    r, e = compute_overview(overview_context, filters_for(name)), ov.ORACLES[name]
    assert list(r.monthly["year_month_label"]) == [m for m, _ in e["monthly"]]
    assert list(r.monthly["net_sales"]) == pytest.approx([v for _, v in e["monthly"]], abs=0.01)
    assert list(r.by_store["store_name"]) == [s for s, _ in e["by_store"]]
    assert list(r.by_store["net_sales"]) == pytest.approx([v for _, v in e["by_store"]], abs=0.01)
    assert list(r.top_products["product_name"]) == [p for p, _ in e["top10"]]
    assert list(r.top_products["net_sales"]) == pytest.approx([v for _, v in e["top10"]], abs=0.01)
    assert list(r.stockout_by_store["store_name"]) == [s for s, _ in e["stockout_by_store"]]
    assert list(r.stockout_by_store["stockout_rate"]) == pytest.approx(
        [v for _, v in e["stockout_by_store"]], rel=1e-12)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_tier_chart_matches_the_oracle_and_always_has_six_rows(overview_context, name):
    r, e = compute_overview(overview_context, filters_for(name)), ov.ORACLES[name]
    assert list(r.tiers["risk_tier"]) == list(inv_m.TIERS)
    assert {t: int(n) for t, n in zip(r.tiers["risk_tier"], r.tiers["skus"]) if n} == e["tiers"]
    assert int(r.tiers["skus"].sum()) == e["tracked_skus"]


# ------------------------------------------------------------------ defaults (decision D1)


def test_default_filters_equal_an_explicit_full_range(overview_context):
    implicit = compute_overview(overview_context, OverviewFilters())
    explicit = compute_overview(overview_context, OverviewFilters(
        date_start=overview_context.date_min, date_end=overview_context.date_max))
    assert implicit.kpis == explicit.kpis
    pd.testing.assert_frame_equal(implicit.monthly, explicit.monthly)
    pd.testing.assert_frame_equal(implicit.top_products, explicit.top_products)


def test_audit_default_values(overview_context):
    k = compute_overview(overview_context, OverviewFilters()).kpis
    assert k.net_sales == pytest.approx(40_389_951.00, abs=0.01)
    assert k.gross_profit == pytest.approx(6_842_371.34, abs=0.01)
    assert k.units_sold == 165_893
    assert k.average_basket == pytest.approx(6_992.72, abs=0.005)
    assert k.stockout_rate == pytest.approx(794 / 37_710, rel=1e-12)
    assert (k.stockout_now, k.critical, k.low) == (4, 83, 193)
    assert k.stock_value == pytest.approx(1_608_182.54, abs=0.01)


# ------------------------------------------------------------------ filter semantics and grain rules


def test_date_range_does_not_reach_the_risk_snapshot(overview_context):
    base = compute_overview(overview_context, OverviewFilters())
    for name in ("ramadan_window", "last_day_only"):
        r = compute_overview(overview_context, filters_for(name))
        pd.testing.assert_frame_equal(r.tiers, base.tiers)
        assert (r.kpis.stockout_now, r.kpis.critical, r.kpis.low, r.kpis.stock_value) == (
            base.kpis.stockout_now, base.kpis.critical, base.kpis.low, base.kpis.stock_value)
        assert r.risk_rows == 419


def test_date_range_does_reach_sales_and_the_stockout_rate(overview_context):
    base = compute_overview(overview_context, OverviewFilters())
    ramadan = compute_overview(overview_context, filters_for("ramadan_window"))
    assert ramadan.kpis.net_sales < base.kpis.net_sales
    assert ramadan.kpis.stockout_rate > base.kpis.stockout_rate        # 3.99% vs 2.11%
    assert list(ramadan.monthly["year_month_label"]) == ["2025-03"]


def test_store_filter_reaches_sales_inventory_and_risk(overview_context):
    f = OverviewFilters(stores=("Alger Centre",))
    assert set(filter_sales(overview_context, f)["store_name"]) == {"Alger Centre"}
    assert set(filter_inventory(overview_context, f)["store_name"]) == {"Alger Centre"}
    assert set(filter_risk(overview_context, f)["store_name"]) == {"Alger Centre"}


def test_category_filter_reaches_sales_inventory_and_risk(overview_context):
    f = OverviewFilters(categories=("Bakery & Pastry",))
    for frame in (filter_sales(overview_context, f), filter_inventory(overview_context, f),
                  filter_risk(overview_context, f)):
        assert set(frame["category_name"]) == {"Bakery & Pastry"}


def test_store_format_equals_the_stores_of_that_format(overview_context):
    by_format = compute_overview(overview_context, OverviewFilters(store_formats=("Hypermarket",)))
    by_store = compute_overview(overview_context, OverviewFilters(stores=("Alger Centre", "Oran Es Senia")))
    assert by_format.kpis == by_store.kpis
    pd.testing.assert_frame_equal(by_format.tiers, by_store.tiers)


def test_filters_combine_with_and(overview_context):
    r = compute_overview(overview_context, filters_for("two_stores_one_format"))
    assert list(r.by_store["store_name"]) == ["Blida Ville"]            # Alger Centre is a Hypermarket


def test_sums_are_consistent_across_visuals(overview_context):
    for name in SCENARIO_NAMES:
        r = compute_overview(overview_context, filters_for(name))
        if r.sales_rows == 0:
            continue
        assert r.monthly["net_sales"].sum() == pytest.approx(r.kpis.net_sales, abs=0.01)
        assert r.by_store["net_sales"].sum() == pytest.approx(r.kpis.net_sales, abs=0.01)
        assert r.top_products["net_sales"].sum() <= r.kpis.net_sales + 0.01
        assert int(r.tiers["skus"].sum()) == r.risk_rows


def test_top_products_never_exceeds_ten_without_ties(overview_context):
    for name in SCENARIO_NAMES:
        assert len(compute_overview(overview_context, filters_for(name)).top_products) <= 10


def test_v2_tier_bars_are_not_flat_and_stockout_cards_agree_with_the_chart(overview_context):
    r = compute_overview(overview_context, OverviewFilters())
    assert list(r.tiers["skus"]) == [4, 83, 193, 139, 0, 0]
    assert r.tiers["skus"].nunique() > 1
    by_tier = dict(zip(r.tiers["risk_tier"], r.tiers["skus"]))
    assert (r.kpis.stockout_now, r.kpis.critical, r.kpis.low) == (
        by_tier["Stockout"], by_tier["Critical"], by_tier["Low"])


def test_stockout_figures_never_use_the_inventory_snapshot_alone(overview_context):
    """The card is SKU-days over the date range; the tier cards are a snapshot (grain rule)."""
    r = compute_overview(overview_context, OverviewFilters())
    assert r.inventory_rows == 37_710 and r.risk_rows == 419
    assert inv_m.stockout_sku_days(filter_inventory(overview_context, OverviewFilters())) == 794


# ------------------------------------------------------------------ empty selection


def test_empty_selection_gives_blanks_zero_counts_and_empty_charts(overview_context):
    r = compute_overview(overview_context, filters_for("empty_selection"))
    k = r.kpis
    assert r.sales_rows == r.inventory_rows == r.risk_rows == 0
    assert k.net_sales is None and k.gross_profit is None and k.gross_margin is None
    assert k.units_sold is None and k.average_basket is None and k.stockout_rate is None
    assert k.stock_value is None
    assert (k.stockout_now, k.critical, k.low) == (0, 0, 0)
    assert len(r.monthly) == len(r.by_store) == len(r.top_products) == len(r.stockout_by_store) == 0
    assert list(r.tiers["skus"]) == [0, 0, 0, 0, 0, 0]


# ------------------------------------------------------------------ read-only behaviour


def test_computing_does_not_modify_the_shared_frames(overview_context):
    def fingerprint(c):
        return tuple(int(pd.util.hash_pandas_object(f, index=True).sum())
                     for f in (c.sales, c.inventory, c.risk))
    before = fingerprint(overview_context)
    for name in SCENARIO_NAMES:
        compute_overview(overview_context, filters_for(name))
    assert fingerprint(overview_context) == before
