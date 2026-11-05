"""Per-page filter state (decision D1: every filter starts at "All", pages are independent).

Streamlit forgets a widget's value when its page is not rendered, so each selection is mirrored
into a page-scoped shadow key and used as the widget default the next time the page is shown.
Keys are prefixed with the page id (for example ``p1_store``): no state is shared between pages.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import streamlit as st

from .filters import normalize_date_range


def _shadow(key: str) -> str:
    return f"_saved__{key}"


def multiselect(page: str, name: str, label: str, options: Sequence[str]) -> tuple[str, ...]:
    key = f"{page}_{name}"
    saved = [v for v in st.session_state.get(_shadow(key), []) if v in options]
    chosen = st.multiselect(label, list(options), default=saved, key=key, placeholder="All")
    st.session_state[_shadow(key)] = list(chosen)
    return tuple(chosen)


def date_range(page: str, name: str, label: str, lo: date, hi: date) -> tuple[date, date]:
    key = f"{page}_{name}"
    saved = st.session_state.get(_shadow(key), (lo, hi))
    value = st.date_input(label, value=saved, min_value=lo, max_value=hi, key=key,
                          format="YYYY-MM-DD")
    st.session_state[_shadow(key)] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    return normalize_date_range(value, lo, hi)


def reset(page: str, names: Sequence[str]) -> None:
    """Callback: forget the selections of ``page`` so every widget returns to All."""
    for name in names:
        key = f"{page}_{name}"
        st.session_state.pop(key, None)
        st.session_state.pop(_shadow(key), None)
