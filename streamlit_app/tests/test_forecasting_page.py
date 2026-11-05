"""Streamlit AppTest of the Forecasting Performance page (backtest accuracy only)."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from apptest_helpers import all_markdown, card, matrix  # noqa: E402
from dzmi_app.constants import BANNER_TEXT  # noqa: E402

MODEL_TABLE = "Model comparison"
BEST_TABLE = "Best model by evaluation basis"


def _page():
    from pages.p4_forecasting_performance import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=120).run()


def test_page_renders_without_errors_and_shows_the_banner(at):
    assert not at.exception
    assert BANNER_TEXT in all_markdown(at)


def test_four_filters_start_at_all(at):
    for key in ("p4_basis", "p4_level", "p4_model", "p4_origin"):
        assert at.multiselect(key=key).value == []
    assert list(at.multiselect(key="p4_origin").options) == ["2025-03-27", "2025-04-03", "2025-04-10", "2025-04-17"]


def test_run_cards(at):
    assert card(at, "Run") == "smoke_m3_20261025_01"
    assert card(at, "Series scored") == "419" and card(at, "Rolling origins") == "4"
    assert card(at, "Horizon (days)") == "14" and card(at, "Stockout days") == "exclude"
    assert "Scope: Observed sales (default)  |  Series x day (default)" in all_markdown(at)


def test_default_model_comparison_is_sorted_by_mae(at):
    rows = matrix(at, MODEL_TABLE)
    assert [r[0] for r in rows] == ["Croston-SBA", "Exp. smoothing + weekday", "Moving average", "Gradient boosting",
                                    "Seasonal naive (benchmark)", "Naive"]
    assert rows[0] == ["Croston-SBA", "23,464", "2.509", "5.142", "61.4%", "+9.4%", "+26.4%", "1", "3"]
    assert rows[4][6] == "0.0%" and rows[5][6] == "-6.8%"


def test_best_model_matrix_has_the_three_existing_combinations(at):
    rows = matrix(at, BEST_TABLE)
    assert [r[:2] for r in rows] == [["Observed sales", "Series x day"], ["Observed sales", "Store x category x week"],
                                     ["Observed, no-stockout days", "Series x day"]]
    assert rows[0][2] == "Croston-SBA"


def test_two_charts_and_the_read_with_care_note(at):
    assert len(at.get("plotly_chart")) == 2
    assert any("Read with care" in i.value for i in at.info)


def test_level_filter_switches_the_scope_to_weeks(at):
    at.multiselect(key="p4_level").select("Store x category x week").run()
    assert not at.exception
    rows = matrix(at, MODEL_TABLE)
    assert rows[0][0] == "Croston-SBA" and rows[0][1] == "560" and rows[0][2] == "31.507"
    assert "Store x category x week  |" in all_markdown(at)


def test_no_stockout_basis_at_week_level_shows_the_empty_state(at):
    at.multiselect(key="p4_basis").select("Observed, no-stockout days").run()
    at.multiselect(key="p4_level").select("Store x category x week").run()
    assert not at.exception
    assert matrix(at, MODEL_TABLE) == []
    assert any("No data for this basis / level combination" in w.value for w in at.warning)


def test_model_and_origin_filters(at):
    at.multiselect(key="p4_model").select("Croston-SBA").select("Naive").run()
    assert not at.exception
    assert [r[0] for r in matrix(at, MODEL_TABLE)] == ["Croston-SBA", "Naive"]
    at.multiselect(key="p4_origin").select("2025-04-10").run()
    assert card(at, "Rolling origins") == "1"


def test_reset_returns_every_filter_to_all(at):
    at.multiselect(key="p4_level").select("Store x category x week").run()
    at.button(key="p4_reset").click().run()
    assert not at.exception
    for key in ("p4_basis", "p4_level", "p4_model", "p4_origin"):
        assert at.multiselect(key=key).value == []
    assert matrix(at, MODEL_TABLE)[0][1] == "23,464"
