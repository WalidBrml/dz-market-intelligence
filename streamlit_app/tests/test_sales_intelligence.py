"""Sales Intelligence calculations. Every scenario is compared with oracle_sales_intelligence.py,
produced by plain SQL on the CSVs (tests/oracle_generator_b2.py), not by dzmi_app."""

from datetime import date

import pandas as pd
import pytest

import oracle_sales_intelligence as ov
from dzmi_app import measures_sales as base
from dzmi_app.sales_intelligence import (
    BLANK_HOUSEHOLD, SalesFilters, build_sales_context, compute_sales_intelligence, filter_sales,
    filter_sales_without_dates, load_sales_context,
)

@pytest.fixture(scope="module")
def sales_context():
    from dzmi_app.sales_intelligence import load_sales_context
    return load_sales_context()


NAMES = list(ov.SCENARIOS)
REL = 1e-9


def filters_for(name: str) -> SalesFilters:
    sc = ov.SCENARIOS[name]
    return SalesFilters(
        date_start=date.fromisoformat(sc["start"]) if "start" in sc else None,
        date_end=date.fromisoformat(sc["end"]) if "end" in sc else None,
        wilayas=tuple(sc.get("wilayas", ())), categories=tuple(sc.get("categories", ())),
        brands=tuple(sc.get("brands", ())))


@pytest.fixture(scope="module")
def results(sales_context):
    return {n: compute_sales_intelligence(sales_context, filters_for(n)) for n in NAMES}


def close(actual, expected, **tol):
    tol = tol or {"rel": REL}
    if expected is None:
        assert actual is None or pd.isna(actual)
    else:
        assert actual == pytest.approx(expected, **tol)


def pairs(frame, label, value):
    return list(frame[label]), [float(v) for v in frame[value]]


# ------------------------------------------------------------------ context and relationships


def test_context_has_one_row_per_fact_row_and_no_inventory_columns(sales_context):
    cols = set(sales_context.sales.columns)
    assert len(sales_context.sales) == 48_216
    assert not cols & {"closing_stock", "opening_stock", "stockout_flag", "received_qty"}


def test_context_joins_do_not_lose_dimension_attributes(sales_context):
    s = sales_context.sales
    for col in ("full_date", "year_month_label", "day_name", "store_name", "wilaya", "floor_area_sqm",
                "product_name", "category_name", "brand"):
        assert not s[col].isna().any(), col


def test_customer_attributes_are_missing_exactly_where_the_customer_key_is(sales_context):
    s = sales_context.sales
    assert s["household_type"].isna().sum() == s["customer_key"].isna().sum() == 31_262
    assert s["promotion_key"].isna().sum() == 45_288


def test_slicer_options(sales_context):
    c = sales_context
    assert c.wilayas == ("S\u00e9tif", "Oran", "Constantine", "Blida", "Alger")   # descending, as in the report
    assert len(c.categories) == 14 and list(c.categories) == sorted(c.categories)
    assert c.brands == tuple(f"Brand {x}" for x in "ABCDEFGHIJ")
    assert c.coverage_text == "Synthetic data loaded: 2025-02-01 to 2025-05-01"
    assert (c.date_min, c.date_max) == (date(2025, 2, 1), date(2025, 5, 1))


def test_context_is_cached_and_rebuildable(dataset, sales_context):
    assert load_sales_context() is sales_context
    assert len(build_sales_context(dataset).sales) == len(sales_context.sales)


def test_wilaya_slicer_filters_the_store_wilaya_not_the_customer_wilaya(sales_context):
    f = SalesFilters(wilayas=("Alger",))
    out = filter_sales(sales_context, f)
    assert set(out["wilaya"]) == {"Alger"} and set(out["store_name"]) == {"Alger Centre"}
    assert len(out) == 11_374


def test_category_and_brand_slicers_filter_through_the_product(sales_context):
    out = filter_sales(sales_context, SalesFilters(categories=("Dairy & Eggs",), brands=("Brand A",)))
    assert set(out["category_name"]) == {"Dairy & Eggs"} and set(out["brand"]) == {"Brand A"}


def test_filters_combine_with_and_and_options_within_a_filter_with_or(sales_context):
    one = len(filter_sales(sales_context, SalesFilters(wilayas=("Oran",))))
    two = len(filter_sales(sales_context, SalesFilters(wilayas=("Oran", "Blida"))))
    both = len(filter_sales(sales_context, SalesFilters(wilayas=("Oran", "Blida"), brands=("Brand A",))))
    assert one < two and both < two == 20_819


def test_date_filter_is_inclusive_on_both_ends(sales_context):
    out = filter_sales(sales_context, SalesFilters(date_start=date(2025, 3, 10), date_end=date(2025, 3, 12)))
    assert out["full_date"].min() == pd.Timestamp("2025-03-10")
    assert out["full_date"].max() == pd.Timestamp("2025-03-12")
    assert out["full_date"].nunique() == 3


def test_prior_month_lookup_ignores_the_date_range_but_keeps_the_other_filters(sales_context):
    f = SalesFilters(date_start=date(2025, 4, 1), date_end=date(2025, 4, 2), wilayas=("Alger",),
                     brands=("Brand A",))
    unfiltered_dates = filter_sales_without_dates(sales_context, f)
    assert unfiltered_dates["full_date"].nunique() == 90
    assert set(unfiltered_dates["wilaya"]) == {"Alger"} and set(unfiltered_dates["brand"]) == {"Brand A"}


# ------------------------------------------------------------------ scenarios vs the independent oracle


@pytest.mark.parametrize("name", NAMES)
def test_row_counts(results, name):
    assert results[name].sales_rows == ov.ORACLES[name]["sales_rows"]


@pytest.mark.parametrize("name", NAMES)
def test_grand_total_row(results, name):
    t, e = results[name].monthly_total, ov.ORACLES[name]["total"]
    close(t["net_sales"], e["net_sales"], abs=0.01)
    close(t["gross_margin"], e["gross_margin"])
    close(t["avg_daily_net_sales"], e["avg_daily_net_sales"])
    assert t["units_sold"] == e["units_sold"]
    close(t["mom"], e["mom"])


@pytest.mark.parametrize("name", NAMES)
def test_monthly_matrix(results, name):
    r, e = results[name].monthly, {m["year_month_label"]: m for m in ov.ORACLES[name]["monthly"]}
    assert set(r["year_month_label"]) == set(e)
    for _, row in r.iterrows():
        m = e[row["year_month_label"]]
        close(row["net_sales"], m["net_sales"], abs=0.01)
        close(row["mom"], m["mom"])
        close(row["gross_margin"], m["gross_margin"])
        close(row["avg_daily_net_sales"], m["avg_daily_net_sales"])
        close(row["units_sold"], m["units_sold"])


@pytest.mark.parametrize("name", NAMES)
def test_monthly_matrix_is_sorted_by_mom_descending_with_blanks_last(results, name):
    mom = list(results[name].monthly["mom"])
    values = [x for x in mom if pd.notna(x)]
    assert values == sorted(values, reverse=True)
    assert all(pd.isna(x) for x in mom[len(values):])


@pytest.mark.parametrize("name", NAMES)
def test_daily_series(results, name):
    d, e = results[name].daily, ov.ORACLES[name]["daily"]
    assert list(d["full_date"].dt.strftime("%Y-%m-%d")) == [x[0] for x in e]
    assert list(d["net_sales"]) == pytest.approx([x[1] for x in e], abs=0.01)
    assert list(d["gross_profit"]) == pytest.approx([x[2] for x in e], abs=0.01)


@pytest.mark.parametrize("name", NAMES)
def test_category_charts_keep_the_report_ordering(results, name):
    r, e = results[name], ov.ORACLES[name]
    labels, values = pairs(r.by_category, "category_name", "net_sales")
    assert labels == [x[0] for x in e["by_category"]]
    assert values == pytest.approx([x[1] for x in e["by_category"]], abs=0.01)
    labels, values = pairs(r.margin_by_category, "category_name", "gross_margin")
    assert labels == [x[0] for x in e["gm_by_category"]]
    assert values == pytest.approx([x[1] for x in e["gm_by_category"]], rel=REL)
    assert r.by_category["net_sales"].is_monotonic_decreasing
    assert r.margin_by_category["gross_margin"].is_monotonic_decreasing


@pytest.mark.parametrize("name", NAMES)
def test_net_sales_per_sqm_by_store(results, name):
    labels, values = pairs(results[name].per_sqm_by_store, "store_name", "net_sales_per_sqm")
    e = ov.ORACLES[name]["sqm_by_store"]
    assert labels == [x[0] for x in e]
    assert values == pytest.approx([x[1] for x in e], rel=REL)


@pytest.mark.parametrize("name", NAMES)
def test_household_chart_includes_the_blank_group(results, name):
    labels, values = pairs(results[name].by_household, "household_type", "net_sales")
    e = ov.ORACLES[name]["household"]
    assert labels == [x[0] for x in e]
    assert values == pytest.approx([x[1] for x in e], abs=0.01)


@pytest.mark.parametrize("name", NAMES)
def test_weekday_chart_runs_monday_to_sunday(results, name):
    w, e = results[name].by_weekday, ov.ORACLES[name]["weekday"]
    assert list(w["day_name"]) == [x[0] for x in e]
    assert list(w["net_sales"]) == pytest.approx([x[2] for x in e], abs=0.01)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert list(w["day_name"]) == [d for d in order if d in set(w["day_name"])]


@pytest.mark.parametrize("name", NAMES)
def test_product_table(results, name):
    r, e = results[name], ov.ORACLES[name]
    assert [(c, p) for c, p in zip(r.products["category_name"], r.products["product_name"])] \
        == [(p["category_name"], p["product_name"]) for p in e["products"]]
    for row, p in zip(r.products.itertuples(index=False), e["products"]):
        close(row.net_sales, p["net_sales"], abs=0.01)
        close(row.gross_profit, p["gross_profit"], abs=0.01)
        close(row.gross_margin, p["gross_margin"])
        close(row.avg_selling_price, p["avg_selling_price"])
        close(row.units_sold, p["units_sold"])
        close(row.promo_share, p["promo_share"])
    t, tot = r.products_total, e["total"]
    close(t["net_sales"], tot["net_sales"], abs=0.01)
    close(t["gross_margin"], tot["gross_margin"])
    close(t["avg_selling_price"], tot["avg_selling_price"])
    close(t["promo_share"], tot["promo_share"])
    assert t["units_sold"] == tot["units_sold"]


# ------------------------------------------------------------------ audit values and business rules


def test_audit_default_figures(results):
    r = results["default"]
    assert r.monthly_total["net_sales"] == pytest.approx(40_389_951.00, abs=0.01)
    assert r.monthly_total["units_sold"] == 165_893
    assert r.monthly_total["gross_margin"] == pytest.approx(6_842_371.34 / 40_389_951.00, rel=1e-12)
    assert len(r.daily) == 90 and len(r.products) == 100 and len(r.by_category) == 14


def test_default_mom_values_only_for_march_and_april(results):
    m = results["default"].monthly.set_index("year_month_label")["mom"]
    assert m["2025-03"] == pytest.approx(0.57151, abs=5e-6)
    assert m["2025-04"] == pytest.approx(-0.28211, abs=5e-6)
    assert pd.isna(m["2025-02"]) and pd.isna(m["2025-05"])           # no January / incomplete May
    assert list(results["default"].monthly["year_month_label"][:2]) == ["2025-03", "2025-04"]
    assert pd.isna(results["default"].monthly_total["mom"])           # the range ends in incomplete May


def test_gross_margin_is_sum_of_profit_over_sum_of_net_sales_never_an_average_of_ratios(results, sales_context):
    r = results["default"]
    manual = sales_context.sales.groupby("category_name")[["gross_profit", "net_sales"]].sum()
    manual = (manual["gross_profit"] / manual["net_sales"]).to_dict()
    for cat, gm in zip(r.margin_by_category["category_name"], r.margin_by_category["gross_margin"]):
        assert gm == pytest.approx(manual[cat], rel=1e-12)
    assert r.monthly_total["gross_margin"] != pytest.approx(
        sum(manual.values()) / len(manual), rel=1e-6)


def test_transactions_are_distinct_basket_ids(sales_context):
    assert base.transactions(sales_context.sales) == 5_776
    assert len(sales_context.sales) == 48_216                          # lines, not baskets
    assert base.average_basket_value(sales_context.sales) == pytest.approx(6_992.72, abs=0.005)


def test_sums_are_consistent_across_the_visuals(results):
    for name, r in results.items():
        if r.sales_rows == 0:
            continue
        total = r.monthly_total["net_sales"]
        assert r.by_category["net_sales"].sum() == pytest.approx(total, abs=0.01)
        assert r.by_household["net_sales"].sum() == pytest.approx(total, abs=0.01)
        assert r.by_weekday["net_sales"].sum() == pytest.approx(total, abs=0.01)
        assert r.daily["net_sales"].sum() == pytest.approx(total, abs=0.01)
        assert r.products["net_sales"].sum() == pytest.approx(total, abs=0.01)
        assert r.monthly["net_sales"].sum() == pytest.approx(total, abs=0.01)


def test_the_product_table_is_sorted_by_category_as_saved_in_the_report(results):
    p = results["default"].products
    assert list(p["category_name"]) == sorted(p["category_name"])
    for _, group in p.groupby("category_name", sort=False):
        assert group["net_sales"].is_monotonic_decreasing


def test_blank_household_group_is_the_anonymous_baskets(results, sales_context):
    h = results["default"].by_household.set_index("household_type")["net_sales"]
    anonymous = sales_context.sales[sales_context.sales["customer_key"].isna()]["net_sales"].sum()
    assert h[BLANK_HOUSEHOLD] == pytest.approx(anonymous, abs=0.01)


def test_products_without_promotions_have_a_blank_share_not_zero(results):
    p = results["default"].products
    assert p["promo_share"].isna().sum() == 38
    assert (p["promo_share"].dropna() > 0).all()


# ------------------------------------------------------------------ date filtering, combinations, empties


def test_date_filter_changes_sales_but_the_matrix_still_compares_with_the_unfiltered_prior_month(results):
    partial = results["partial_range"]
    m = partial.monthly.set_index("year_month_label")
    assert list(m.index) and set(m.index) == {"2025-03", "2025-04"}
    assert m.loc["2025-03", "mom"] == pytest.approx(0.64714, abs=5e-6)     # Mar 10-31 vs Feb 10-28
    assert m.loc["2025-04", "mom"] == pytest.approx(-0.2312, abs=5e-5)     # Apr 1-20 vs Mar 1-20


def test_a_single_incomplete_day_has_no_mom(results):
    r = results["last_day_only"]
    assert list(r.monthly["year_month_label"]) == ["2025-05"]
    assert pd.isna(r.monthly.loc[0, "mom"]) and pd.isna(r.monthly_total["mom"])
    assert r.monthly_total["net_sales"] == pytest.approx(419_930.0, abs=0.01)


def test_combined_filters_reduce_every_visual(results):
    combined, default = results["combined"], results["default"]
    assert 0 < combined.sales_rows < default.sales_rows
    assert set(combined.by_category["category_name"]) == {"Dairy & Eggs", "Personal Care"}
    assert set(combined.per_sqm_by_store["store_name"]) <= {"Oran Es Senia", "Blida Ville"}


def test_empty_selection_returns_blanks_and_empty_frames(results):
    r = results["empty_selection"]
    assert r.sales_rows == 0
    for frame in (r.daily, r.monthly, r.by_category, r.margin_by_category, r.per_sqm_by_store,
                  r.by_household, r.products, r.by_weekday):
        assert len(frame) == 0
    t = r.monthly_total
    assert t["net_sales"] is None and t["gross_margin"] is None and t["units_sold"] is None
    assert t["avg_daily_net_sales"] is None and t["mom"] is None
    assert all(v is None for v in r.products_total.values())


def test_stores_without_sales_are_absent_from_the_per_sqm_chart(results):
    assert list(results["wilaya_alger"].per_sqm_by_store["store_name"]) == ["Alger Centre"]
    assert len(results["default"].per_sqm_by_store) == 5


def test_default_filters_equal_an_explicit_full_range(sales_context):
    implicit = compute_sales_intelligence(sales_context, SalesFilters())
    explicit = compute_sales_intelligence(
        sales_context, SalesFilters(date_start=sales_context.date_min, date_end=sales_context.date_max))
    pd.testing.assert_frame_equal(implicit.monthly, explicit.monthly)
    pd.testing.assert_frame_equal(implicit.products, explicit.products)
    assert implicit.monthly_total == explicit.monthly_total


# ------------------------------------------------------------------ read-only and regression


def test_computing_does_not_modify_the_shared_frames(sales_context):
    def fingerprint():
        return (int(pd.util.hash_pandas_object(sales_context.sales, index=True).sum()),
                int(pd.util.hash_pandas_object(sales_context.dates, index=True).sum()))
    before = fingerprint()
    for name in NAMES:
        compute_sales_intelligence(sales_context, filters_for(name))
    assert fingerprint() == before


def test_executive_overview_is_unchanged(overview_context):
    """B1 regression: the Page 1 figures still equal the audited values."""
    from dzmi_app.filters import OverviewFilters
    from dzmi_app.overview import compute_overview
    k = compute_overview(overview_context, OverviewFilters()).kpis
    assert k.net_sales == pytest.approx(40_389_951.00, abs=0.01)
    assert k.gross_profit == pytest.approx(6_842_371.34, abs=0.01)
    assert k.units_sold == 165_893
    assert (k.stockout_now, k.critical, k.low) == (4, 83, 193)


def test_navigation_keeps_the_six_pages_in_the_original_order(app_dir):
    import re
    source = (app_dir / "app.py").read_text(encoding="utf-8")
    titles = re.findall(r'(?<![A-Za-z_])title="([^"]+)"', source)
    assert titles == ["Executive Overview", "Sales Intelligence", "Inventory Intelligence",
                      "Forecasting Performance", "Forecast Accuracy by Horizon", "Data & Method Notes"]
    assert "render_sales_intelligence" in source and "render_executive_overview" in source
