"""Streamlit AppTest of the Forecast Accuracy by Horizon page (backtest accuracy only)."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from apptest_helpers import all_markdown, matrix  # noqa: E402
from dzmi_app.constants import BANNER_TEXT  # noqa: E402
from dzmi_app.forecast_horizon import AXIS_DAY, AXIS_WEEK  # noqa: E402

PIVOT = "MAE by horizon x model"


def _page():
    from pages.p5_forecast_accuracy_by_horizon import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=120).run()


def test_page_renders_without_errors_and_shows_the_banner(at):
    assert not at.exception
    assert BANNER_TEXT in all_markdown(at)


def test_filters_start_at_all_and_four_charts_are_drawn(at):
    for key in ("p5_basis", "p5_level", "p5_model", "p5_origin"):
        assert at.multiselect(key=key).value == []
    assert len(at.get("plotly_chart")) == 4


def test_scope_and_axis_are_stated(at):
    text = all_markdown(at)
    assert "Scope: Observed sales (default)  |  Series x day (default)" in text
    assert "Horizon axis (read before comparing)" in text and AXIS_DAY in text


def test_default_pivot_has_fourteen_day_rows_in_numeric_order(at):
    rows = matrix(at, PIVOT)
    assert [r[0] for r in rows] == [f"Day {i}" for i in range(1, 15)]
    assert rows[0] == ["Day 1", "3.731", "4.119", "2.879", "3.094", "2.879", "3.045"]


def test_week_level_has_two_rows_and_its_own_axis_title(at):
    at.multiselect(key="p5_level").select("Store x category x week").run()
    assert not at.exception
    assert [r[0] for r in matrix(at, PIVOT)] == ["Week 1", "Week 2"]
    assert AXIS_WEEK in all_markdown(at)


def test_no_stockout_basis_at_week_level_shows_the_empty_state(at):
    at.multiselect(key="p5_basis").select("Observed, no-stockout days").run()
    at.multiselect(key="p5_level").select("Store x category x week").run()
    assert not at.exception
    assert any("No data for this basis / level combination" in w.value for w in at.warning)
    assert matrix(at, PIVOT) == []


def test_model_filter_limits_the_pivot_columns(at):
    at.multiselect(key="p5_model").select("Croston-SBA").select("Naive").run()
    assert not at.exception
    rows = matrix(at, PIVOT)
    assert len(rows[0]) == 3 and rows[0][0] == "Day 1"


def test_reset_returns_every_filter_to_all(at):
    at.multiselect(key="p5_level").select("Store x category x week").run()
    at.button(key="p5_reset").click().run()
    assert not at.exception
    assert at.multiselect(key="p5_level").value == []
    assert len(matrix(at, PIVOT)) == 14
