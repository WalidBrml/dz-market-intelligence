"""Plotly figure builders: order, orientation, colours and the empty state."""

import pytest

pytest.importorskip("plotly")

from dzmi_app import charts, theme  # noqa: E402


def test_hbar_keeps_the_given_order_with_the_first_category_on_top():
    fig = charts.hbar_chart(["a", "b", "c"], [3, 2, 1], ["3", "2", "1"], ["x", "y", "z"], "T")
    bar = fig.data[0]
    assert bar.orientation == "h"
    assert list(bar.y) == ["a", "b", "c"] and list(bar.x) == [3, 2, 1]
    assert fig.layout.yaxis.autorange == "reversed"
    assert list(bar.text) == ["3", "2", "1"]
    assert fig.layout.title.text == "T"
    assert fig.layout.showlegend is False


def test_hbar_accepts_one_colour_per_bar():
    colors = [theme.TIER_COLORS[t] for t in ("Stockout", "Critical", "Low")]
    fig = charts.hbar_chart(["a", "b", "c"], [1, 2, 3], ["1", "2", "3"], ["", "", ""], "T", colors=colors)
    assert list(fig.data[0].marker.color) == colors


def test_column_chart_uses_categorical_x_in_the_given_order():
    fig = charts.column_chart(["2025-02", "2025-03"], [1.0, 2.0], ["1", "2"], ["h1", "h2"], "T")
    assert list(fig.data[0].x) == ["2025-02", "2025-03"]
    assert fig.layout.xaxis.type == "category"


@pytest.mark.parametrize("build", [
    lambda: charts.hbar_chart([], [], [], [], "Empty"),
    lambda: charts.column_chart([], [], [], [], "Empty"),
])
def test_empty_input_gives_a_message_instead_of_an_empty_plot(build):
    fig = build()
    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == "No data for the current filters"
    assert fig.layout.title.text == "Empty"


def test_tier_colours_cover_every_tier_and_keep_the_original_three():
    assert set(theme.TIER_COLORS) == {"Stockout", "Critical", "Low", "Healthy", "Overstock", "No recent sales"}
    assert theme.TIER_COLORS["Critical"] == "#D9534F"
    assert theme.TIER_COLORS["Stockout"] == "#A61C1C"
    assert theme.TIER_COLORS["Low"] == "#5B9BD5"
