"""Streamlit AppTest of the Sales Intelligence page (rendered in isolation from navigation)."""

import re
from datetime import date
from html import unescape

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dzmi_app.constants import BANNER_TEXT  # noqa: E402


def _page():
    from pages.p2_sales_intelligence import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=120).run()


def matrix(app):
    """Rows of the 'Monthly performance' matrix as lists of cell texts (total row last)."""
    for md in app.markdown:
        if "Monthly performance" in md.value and "<table" in md.value:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", md.value)
            return [[unescape(c) for c in re.findall(r"<td>(.*?)</td>", r)] for r in rows if "<td>" in r]
    raise AssertionError("matrix not found")


def test_page_renders_without_errors(at):
    assert not at.exception


def test_banner_coverage_and_title(at):
    texts = " ".join(md.value for md in at.markdown)
    assert BANNER_TEXT in texts
    assert "Synthetic data loaded: 2025-02-01 to 2025-05-01" in texts
    assert "Sales Intelligence" in texts


def test_all_four_filters_start_at_all(at):
    assert at.date_input(key="p2_date").value == (date(2025, 2, 1), date(2025, 5, 1))
    for key in ("p2_wilaya", "p2_category", "p2_brand"):
        assert at.multiselect(key=key).value == []


def test_wilaya_options_follow_the_report_slicer_order(at):
    assert list(at.multiselect(key="p2_wilaya").options) == ["S\u00e9tif", "Oran", "Constantine", "Blida", "Alger"]


def test_default_matrix_is_sorted_by_mom_and_shows_the_audited_values(at):
    rows = matrix(at)
    assert [r[0] for r in rows] == ["2025-03", "2025-04", "2025-02", "2025-05", "Total"]
    assert rows[0][1:3] == ["16,978,046", "57.2%"]
    assert rows[1][1:3] == ["12,188,345", "-28.2%"]
    assert rows[2][2] == "" and rows[3][2] == ""                       # no January, incomplete May
    assert rows[4][1] == "40,389,951" and rows[4][5] == "165,893" and rows[4][3] == "16.9%"


def test_charts_grid_and_total_strip_are_drawn(at):
    assert len(at.get("plotly_chart")) == 6
    assert len(at.dataframe) == 1
    assert any("40,389,951" in md.value and "dz-strip" in md.value for md in at.markdown)


def test_wilaya_filter_propagates_to_the_matrix(at):
    at.multiselect(key="p2_wilaya").select("Alger").run()
    assert not at.exception
    assert matrix(at)[-1][1] == "9,540,683"


def test_combined_filters_and_empty_result(at):
    at.multiselect(key="p2_wilaya").select("Alger").run()
    at.multiselect(key="p2_category").select("Dairy & Eggs").run()
    at.multiselect(key="p2_brand").select("Brand E").run()
    assert not at.exception
    assert any("No sales match the current filters" in i.value for i in at.info)
    assert matrix(at) == [["Total", "", "", "", "", ""]]


def test_date_range_uses_the_unfiltered_prior_month(at):
    at.date_input(key="p2_date").set_value((date(2025, 3, 10), date(2025, 4, 20))).run()
    assert not at.exception
    rows = {r[0]: r for r in matrix(at)}
    assert rows["2025-03"][2] == "64.7%" and rows["2025-04"][2] == "-23.1%"
    assert "2025-02" not in rows and "2025-05" not in rows


def test_reset_returns_every_filter_to_all(at):
    at.multiselect(key="p2_wilaya").select("Alger").run()
    at.multiselect(key="p2_brand").select("Brand A").run()
    assert matrix(at)[-1][1] != "40,389,951"
    at.button(key="p2_reset").click().run()
    assert not at.exception
    for key in ("p2_wilaya", "p2_category", "p2_brand"):
        assert at.multiselect(key=key).value == []
    assert at.date_input(key="p2_date").value == (date(2025, 2, 1), date(2025, 5, 1))
    assert matrix(at)[-1][1] == "40,389,951"
