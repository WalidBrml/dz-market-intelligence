"""Plotly builders specific to Sales Intelligence (the bar/column builders live in charts.py)."""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from . import theme
from .charts import _layout, empty_figure


def line_chart(x: Sequence, series: Sequence[tuple[str, Sequence[float], str]], title: str,
               height: int = 330) -> go.Figure:
    """Daily lines. ``series`` = [(name, values, colour), ...]; one trace per measure."""
    if len(x) == 0:
        return empty_figure(title, height=height)
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(go.Scatter(x=list(x), y=list(values), name=name, mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate=name + ": %{y:,.0f} DZD<extra></extra>"))
    fig.update_xaxes(showgrid=False, linecolor=theme.BORDER, hoverformat="%Y-%m-%d")
    fig.update_yaxes(showgrid=True, gridcolor=theme.GRID, zeroline=False, tickformat="~s",
                     rangemode="tozero")
    _layout(fig, title, height)
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig
