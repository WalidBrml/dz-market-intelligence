"""Thin Streamlit wrappers around the HTML snippets and Plotly figures."""

from __future__ import annotations

import inspect

import streamlit as st

from . import ui_html


def inject_css() -> None:
    st.markdown(f"<style>{ui_html.APP_CSS}</style>", unsafe_allow_html=True)


def show_banner() -> None:
    st.markdown(ui_html.banner_html(), unsafe_allow_html=True)


def page_header(title: str, coverage_text: str, subtitle: str = "") -> None:
    left, right = st.columns([3, 2], gap="small", vertical_alignment="center")
    left.markdown(ui_html.title_html(title, subtitle), unsafe_allow_html=True)
    if coverage_text:
        right.markdown(ui_html.coverage_html(coverage_text), unsafe_allow_html=True)


def section(text: str, caption: str = "") -> None:
    st.markdown(ui_html.section_html(text, caption), unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", tone: str = "default") -> None:
    st.markdown(ui_html.kpi_card_html(label, value, note, tone), unsafe_allow_html=True)


def _plotly_kwargs() -> dict:
    """``width='stretch'`` on current Streamlit, ``use_container_width`` on older releases."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def show_chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **_plotly_kwargs())
