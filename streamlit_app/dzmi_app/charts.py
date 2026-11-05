"""Plotly figure builders for the Executive Overview (one consistent chart template)."""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

from . import theme


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=theme.NAVY)),
        height=height,
        margin=dict(l=8, r=24, t=48, b=8),
        paper_bgcolor=theme.WHITE,
        plot_bgcolor=theme.WHITE,
        font=dict(family=theme.FONT_STACK, size=12, color=theme.TEXT),
        showlegend=False,
        bargap=0.35,
        hoverlabel=dict(font_family=theme.FONT_STACK),
    )
    return fig


def empty_figure(title: str, message: str = "No data for the current filters", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=theme.GREY))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, height)


def column_chart(categories: Sequence[str], values: Sequence[float], labels: Sequence[str],
                 hover: Sequence[str], title: str, height: int = 320,
                 color: str = theme.PRIMARY) -> go.Figure:
    if len(categories) == 0:
        return empty_figure(title, height=height)
    fig = go.Figure(go.Bar(
        x=list(categories), y=list(values), marker_color=color, text=list(labels),
        textposition="outside", cliponaxis=False, customdata=list(hover),
        hovertemplate="%{x}<br>%{customdata}<extra></extra>"))
    fig.update_xaxes(type="category", showgrid=False, linecolor=theme.BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=theme.GRID, zeroline=False, tickformat="~s")
    return _layout(fig, title, height)


def hbar_chart(categories: Sequence[str], values: Sequence[float], labels: Sequence[str],
               hover: Sequence[str], title: str, height: int = 320,
               colors: Sequence[str] | str = theme.PRIMARY, tickformat: str = "~s") -> go.Figure:
    """Horizontal bars, first category on top (the order given is the order shown)."""
    if len(categories) == 0:
        return empty_figure(title, height=height)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(categories), orientation="h",
        marker_color=colors if isinstance(colors, str) else list(colors),
        text=list(labels), textposition="outside", cliponaxis=False, customdata=list(hover),
        hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True, linecolor=theme.BORDER)
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID, zeroline=False, tickformat=tickformat,
                     rangemode="tozero")
    return _layout(fig, title, height)
