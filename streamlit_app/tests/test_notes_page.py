"""Streamlit AppTest of the Data & Method Notes page."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from apptest_helpers import all_markdown, card, matrix  # noqa: E402
from dzmi_app import notes  # noqa: E402
from dzmi_app.constants import BANNER_TEXT  # noqa: E402


def _page():
    from pages.p6_data_and_method_notes import render
    render()


@pytest.fixture()
def at():
    return AppTest.from_function(_page, default_timeout=180).run()


def test_page_renders_without_errors_and_shows_the_banner(at):
    assert not at.exception
    assert BANNER_TEXT in all_markdown(at)
    assert "Synthetic data loaded: 2025-02-01 to 2025-05-01" in all_markdown(at)


def test_the_report_text_is_shown_verbatim(at):
    text = all_markdown(at)
    for heading, body in notes.WHAT_THIS_REPORT_IS:
        assert heading in text and body in text
    for line in notes.KNOWN_LIMITS:
        assert line in text


def test_coverage_cards_come_from_the_data(at):
    assert card(at, "Period") == "2025-02-01 to 2025-05-01"
    assert card(at, "Stores / products") == "5 / 100"
    assert card(at, "Sales lines") == "48,216"
    assert card(at, "Inventory SKU-days") == "37,710"
    assert card(at, "Risk snapshot") == "2025-05-01"
    assert card(at, "Stored forecasts") == "0 rows"


def test_backtest_versus_forecast_section_states_that_no_forecast_exists(at):
    text = all_markdown(at)
    assert "Backtesting is not forecasting" in text and "fact_demand_forecast has 0 rows" in text


def test_file_table_lists_22_files_and_flags_the_empty_ones(at):
    rows = matrix(at, "Exported files and row counts")
    assert len(rows) == 22
    flagged = {r[0] for r in rows if r[2] == "EMPTY (0 rows)"}
    assert flagged == {"fact_demand_forecast", "model_run"}


def test_integrity_checks_are_listed_and_all_passed(at):
    rows = matrix(at, "Data integrity checks")
    assert len(rows) == 18 and {r[3] for r in rows} == {"passed"}


def test_reconciliation_table(at):
    rows = matrix(at, "MAE reconciliation")
    assert len(rows) == 18 and {r[6] for r in rows} == {"OK"}
    assert "18 of 18 OK" in all_markdown(at)


def test_definitions_and_deviations_are_present(at):
    assert len(matrix(at, "Definitions")) == len(notes.DEFINITIONS)
    assert len(matrix(at, "Intentional differences")) == len(notes.DEVIATIONS)
