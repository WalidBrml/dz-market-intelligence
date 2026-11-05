"""Streamlit AppTest of the Inventory Intelligence page (rendered in isolation from navigation)."""

from datetime import date

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from apptest_helpers import all_markdown, card  # noqa: E402
from dzmi_app.constants import BANNER_TEXT  # noqa: E402


def _page():
    from pages.p3_inventory_intelligence import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=120).run()


def test_page_renders_without_errors_and_shows_the_banner(at):
    assert not at.exception
    assert BANNER_TEXT in all_markdown(at)
    assert "Synthetic data loaded: 2025-02-01 to 2025-05-01" in all_markdown(at)


def test_all_five_filters_start_at_all(at):
    assert at.date_input(key="p3_date").value == (date(2025, 2, 1), date(2025, 5, 1))
    for key in ("p3_store", "p3_category", "p3_product", "p3_tier"):
        assert at.multiselect(key=key).value == []
    assert list(at.multiselect(key="p3_tier").options) == ["Stockout", "Critical", "Low", "Healthy"]
    assert len(at.multiselect(key="p3_product").options) == 100


def test_default_cards_show_the_audited_values(at):
    expected = {
        "Stockout rate": "2.11%", "Availability rate": "97.89%", "Stockout SKU-days": "794",
        "Closing stock (last day)": "7,291", "Days of cover - overall": "4.3",
        "Days of cover - median SKU": "5.4", "At-risk SKUs (latest day)": "66.8 %",
    }
    for label, value in expected.items():
        assert card(at, label) == value, label


def test_three_charts_and_the_sku_grid_are_drawn(at):
    assert len(at.get("plotly_chart")) == 3
    assert len(at.dataframe) == 1


def test_date_range_moves_the_rates_and_the_last_day_but_not_the_snapshot(at):
    at.date_input(key="p3_date").set_value((date(2025, 3, 1), date(2025, 3, 29))).run()
    assert not at.exception
    assert card(at, "Stockout rate") == "3.99%"
    assert card(at, "Closing stock (last day)") == "6,313"
    assert card(at, "Days of cover - median SKU") == "5.4"
    assert card(at, "At-risk SKUs (latest day)") == "66.8 %"


def test_store_filter_reaches_the_rates_and_the_snapshot_share(at):
    at.multiselect(key="p3_store").select("Alger Centre").run()
    assert not at.exception
    assert card(at, "Stockout rate") == "2.05%"
    assert card(at, "At-risk SKUs (latest day)") != "66.8 %"


def test_risk_tier_filter_does_not_change_the_cards(at):
    at.multiselect(key="p3_tier").select("Critical").run()
    assert not at.exception
    assert card(at, "At-risk SKUs (latest day)") == "66.8 %"
    assert card(at, "Stockout rate") == "2.11%"


def test_empty_selection_shows_blanks_zero_counts_and_a_message(at):
    at.multiselect(key="p3_store").select("Alger Centre").run()
    at.multiselect(key="p3_product").select("Eggs 1 L (Brand I)").run()
    assert not at.exception
    assert card(at, "Stockout rate") == "\u2014" and card(at, "Closing stock (last day)") == "\u2014"
    assert card(at, "Stockout SKU-days") == "0"
    assert any("No inventory records match the current filters" in i.value for i in at.info)


def test_reset_returns_every_filter_to_all(at):
    at.multiselect(key="p3_store").select("Alger Centre").run()
    at.multiselect(key="p3_tier").select("Low").run()
    at.button(key="p3_reset").click().run()
    assert not at.exception
    for key in ("p3_store", "p3_category", "p3_product", "p3_tier"):
        assert at.multiselect(key=key).value == []
    assert card(at, "Stockout rate") == "2.11%"
