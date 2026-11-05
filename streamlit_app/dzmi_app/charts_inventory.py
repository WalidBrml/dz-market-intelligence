"""Plotly builders for Inventory Intelligence."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

from . import theme
from .charts import _layout, empty_figure
from .formatting import fmt_percent


def rate_line_chart(dates: Sequence, rates: Sequence[float], title: str, height: int = 330) -> go.Figure:
    if len(dates) == 0:
        return empty_figure(title, height=height)
    fig = go.Figure(go.Scatter(x=list(dates), y=list(rates), mode="lines", line=dict(color=theme.RED, width=2),
                               hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>"))
    fig.update_xaxes(showgrid=False, linecolor=theme.BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=theme.GRID, zeroline=False, tickformat=".1%", rangemode="tozero")
    return _layout(fig, title, height)


def rate_heatmap(categories: Sequence[str], stores: Sequence[str], matrix, title: str, height: int = 420) -> go.Figure:
    """Category x store stockout rate; blank pairs stay empty. Colour scale from light to red."""
    if len(categories) == 0 or len(stores) == 0:
        return empty_figure(title, height=height)
    z = np.asarray(matrix, dtype=float)
    text = [["" if np.isnan(v) else fmt_percent(v, 1) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=np.where(np.isnan(z), None, z).tolist(), x=list(stores), y=list(categories), text=text,
        texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [0.5, "#F4B6B3"], [1, theme.DEEP_RED]], zmin=0,
        hoverongaps=False, hovertemplate="%{y} / %{x}: %{text}<extra></extra>", showscale=False, xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(side="top")
    return _layout(fig, title, height)
