"""Sales Intelligence line chart."""

import pytest

pytest.importorskip("plotly")

from dzmi_app import charts_sales, theme  # noqa: E402


def test_line_chart_has_one_line_per_measure_in_the_given_order():
    fig = charts_sales.line_chart(
        ["2025-02-01", "2025-02-02"],
        [("Net Sales", [1.0, 2.0], theme.BLUE), ("Gross Profit", [0.5, 0.7], theme.PRIMARY)],
        "Daily net sales and gross profit (DZD)")
    assert [t.name for t in fig.data] == ["Net Sales", "Gross Profit"]
    assert all(t.mode == "lines" for t in fig.data)
    assert list(fig.data[0].y) == [1.0, 2.0]
    assert fig.layout.title.text == "Daily net sales and gross profit (DZD)"
    assert fig.layout.showlegend is True


def test_empty_series_gives_the_message_figure():
    fig = charts_sales.line_chart([], [("Net Sales", [], theme.BLUE)], "T")
    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == "No data for the current filters"
