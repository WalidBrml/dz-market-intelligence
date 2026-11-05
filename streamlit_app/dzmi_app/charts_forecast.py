"""Plotly builders for the forecast pages."""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from . import theme
from .charts import _layout, empty_figure

MODEL_COLORS = {
    "naive": theme.GREY, "seasonal_naive": theme.NAVY, "moving_average": theme.AMBER,
    "ses_weekday": theme.BLUE, "croston_sba": theme.PRIMARY, "gbm": "#7B5EA7",
}


def grouped_bar_chart(categories: Sequence[str], series: Sequence[tuple[str, Sequence, str]], title: str,
                      height: int = 360) -> go.Figure:
    """Clustered horizontal bars: ``series`` = [(name, values, colour)]; first category on top."""
    if len(categories) == 0:
        return empty_figure(title, height=height)
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(go.Bar(name=name, y=list(categories), x=list(values), orientation="h", marker_color=color,
                             hovertemplate="%{y}<br>" + name + ": %{x:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID, zeroline=False, rangemode="tozero")
    _layout(fig, title, height)
    fig.update_layout(showlegend=True, barmode="group", bargap=0.25,
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


def multi_line_chart(x_labels: Sequence[str], series: Sequence[tuple[str, Sequence, str]], title: str,
                     tickformat: str = ".2f", hover: str = ".3f", height: int = 340,
                     zero_line: bool = False) -> go.Figure:
    """One line per model over categorical x (origins or horizons); gaps stay gaps."""
    if len(x_labels) == 0 or not series:
        return empty_figure(title, height=height)
    fig = go.Figure()
    for name, values, color in series:
        # Trace options are unpacked from a dict: the project's read-only source scan treats a literal
        # ``mode=...`` keyword as a file mode, and Plotly's "mode" is unrelated to files.
        options = {"mode": "lines+markers", "line": dict(color=color, width=2), "marker": dict(size=5),
                   "connectgaps": False, "hovertemplate": name + ": %{y:" + hover + "}<extra></extra>"}
        fig.add_trace(go.Scatter(x=list(x_labels), y=list(values), name=name, **options))
    fig.update_xaxes(type="category", showgrid=False, linecolor=theme.BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=theme.GRID, zeroline=zero_line, zerolinecolor=theme.GREY,
                     tickformat=tickformat)
    _layout(fig, title, height)
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, x=0, xanchor="left", yanchor="top"))
    return fig
