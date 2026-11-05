"""Streamlit AppTest of the Executive Overview page (rendered in isolation from navigation)."""

import re
from datetime import date
from html import escape

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dzmi_app.constants import BANNER_TEXT  # noqa: E402


def _page():
    from pages.p1_executive_overview import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=120).run()


def card(app, label):
    """Value text of the KPI card with this label."""
    pattern = (r'dz-card__label">' + re.escape(escape(label)) + r'</div><div class="dz-card__value">([^<]*)<')
    for md in app.markdown:
        found = re.search(pattern, md.value)
        if found:
            return found.group(1)
    raise AssertionError(f"card not found: {label}")


def test_page_renders_without_errors(at):
    assert not at.exception


def test_synthetic_banner_is_visible(at):
    assert any(BANNER_TEXT in md.value for md in at.markdown)


def test_data_coverage_card_and_title(at):
    texts = " ".join(md.value for md in at.markdown)
    assert "Synthetic data loaded: 2025-02-01 to 2025-05-01" in texts
    assert "Executive Overview" in texts


def test_all_filters_start_at_all(at):
    assert at.date_input(key="p1_date").value == (date(2025, 2, 1), date(2025, 5, 1))
    for key in ("p1_store", "p1_category", "p1_format"):
        assert at.multiselect(key=key).value == []


def test_default_kpi_cards_show_the_audited_values(at):
    expected = {
        "Net sales (M DZD)": "40.4M", "Gross profit (M DZD)": "6.8M", "Gross margin": "16.9%",
        "Units sold (thousands)": "165.9K", "Average basket (DZD)": "6,992.72",
        "Stockout rate (SKU-days)": "2.11%", "Stockout now (SKUs)": "4",
        "Critical (<3 days cover)": "83", "Low (3-7 days)": "193",
        "Stock value at cost (M DZD)": "1.61M",
    }
    for label, value in expected.items():
        assert card(at, label) == value, label


def test_five_charts_are_drawn(at):
    assert len(at.get("plotly_chart")) == 5


def test_store_filter_changes_the_cards(at):
    at.multiselect(key="p1_store").select("Alger Centre").run()
    assert not at.exception
    assert card(at, "Net sales (M DZD)") == "9.5M"
    assert card(at, "Critical (<3 days cover)") == "17"
    assert card(at, "Stockout rate (SKU-days)") == "2.05%"


def test_date_range_changes_sales_and_stockout_rate_but_not_the_snapshot(at):
    at.date_input(key="p1_date").set_value((date(2025, 3, 1), date(2025, 3, 29))).run()
    assert not at.exception
    assert card(at, "Net sales (M DZD)") == "16.7M"
    assert card(at, "Stockout rate (SKU-days)") == "3.99%"
    assert card(at, "Critical (<3 days cover)") == "83"
    assert card(at, "Stock value at cost (M DZD)") == "1.61M"


def test_empty_selection_shows_blanks_and_a_message(at):
    at.multiselect(key="p1_store").select("Alger Centre").run()
    at.multiselect(key="p1_format").select("Supermarket").run()
    assert not at.exception
    assert card(at, "Net sales (M DZD)") == "\u2014"
    assert card(at, "Stockout now (SKUs)") == "0"
    assert any("No sales match the current filters" in i.value for i in at.info)


def test_reset_returns_every_filter_to_all(at):
    at.multiselect(key="p1_store").select("Alger Centre").run()
    at.multiselect(key="p1_category").select("Bakery & Pastry").run()
    assert card(at, "Net sales (M DZD)") != "40.4M"
    at.button(key="p1_reset").click().run()
    assert not at.exception
    assert at.multiselect(key="p1_store").value == []
    assert at.multiselect(key="p1_category").value == []
    assert at.date_input(key="p1_date").value == (date(2025, 2, 1), date(2025, 5, 1))
    assert card(at, "Net sales (M DZD)") == "40.4M"
