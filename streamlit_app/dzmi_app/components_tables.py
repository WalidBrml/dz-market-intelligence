"""Streamlit wrapper for sortable grids (kept apart from components.py, which is unchanged)."""

from __future__ import annotations

import streamlit as st


def show_dataframe(data, key: str, height: int = 380) -> None:
    """Sortable grid, full width on current Streamlit and on older releases."""
    try:
        st.dataframe(data, hide_index=True, key=key, height=height, width="stretch")
    except Exception:
        st.dataframe(data, hide_index=True, key=key, height=height, use_container_width=True)
