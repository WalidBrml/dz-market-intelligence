"""Plotly builders of pages 3-6."""

import numpy as np
import pytest

pytest.importorskip("plotly")

from dzmi_app import charts_forecast, charts_inventory, theme  # noqa: E402


def test_rate_line_chart_is_a_single_red_line_with_percent_axis():
    fig = charts_inventory.rate_line_chart(["2025-02-01", "2025-02-02"], [0.01, 0.02], "T")
    assert len(fig.data) == 1 and fig.data[0].mode == "lines" and fig.data[0].line.color == theme.RED
    assert fig.layout.yaxis.tickformat == ".1%" and fig.layout.title.text == "T"


def test_heatmap_keeps_blank_cells_blank_and_the_category_order():
    fig = charts_inventory.rate_heatmap(["A", "B"], ["S1", "S2"], np.array([[0.5, 0.0], [1.0, np.nan]]), "T")
    z = fig.data[0].z
    assert z[1][1] is None and z[0][0] == 0.5 and z[1][0] == 1.0
    assert list(fig.data[0].y) == ["A", "B"] and fig.layout.yaxis.autorange == "reversed"
    assert fig.data[0].text[1][1] == "" and fig.data[0].text[0][0] == "50.0%"


@pytest.mark.parametrize("build", [
    lambda: charts_inventory.rate_line_chart([], [], "E"),
    lambda: charts_inventory.rate_heatmap([], [], np.empty((0, 0)), "E"),
    lambda: charts_forecast.grouped_bar_chart([], [], "E"),
    lambda: charts_forecast.multi_line_chart([], [], "E"),
])
def test_empty_input_gives_the_message_figure(build):
    fig = build()
    assert len(fig.data) == 0 and fig.layout.annotations[0].text == "No data for the current filters"


def test_grouped_bars_have_one_trace_per_series_and_first_category_on_top():
    fig = charts_forecast.grouped_bar_chart(["a", "b"], [("MAE", [1, 2], "#111111"), ("Benchmark", [3, 3], "#222222")], "T")
    assert [t.name for t in fig.data] == ["MAE", "Benchmark"] and all(t.orientation == "h" for t in fig.data)
    assert fig.layout.yaxis.autorange == "reversed" and fig.layout.barmode == "group"


def test_multi_line_chart_uses_categorical_x_and_keeps_gaps():
    fig = charts_forecast.multi_line_chart(["Day 1", "Day 2", "Day 10"], [("m", [1.0, None, 3.0], "#123456")], "T")
    assert fig.layout.xaxis.type == "category" and fig.data[0].connectgaps is False
    assert list(fig.data[0].x) == ["Day 1", "Day 2", "Day 10"]


def test_every_model_has_a_colour_and_the_benchmark_is_navy():
    assert set(charts_forecast.MODEL_COLORS) == {"naive", "seasonal_naive", "moving_average", "ses_weekday",
                                                 "croston_sba", "gbm"}
    assert charts_forecast.MODEL_COLORS["seasonal_naive"] == theme.NAVY
    assert len(set(charts_forecast.MODEL_COLORS.values())) == 6
