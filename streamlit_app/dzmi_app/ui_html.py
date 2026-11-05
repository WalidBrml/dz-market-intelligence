"""HTML snippets for the UI (pure strings, no Streamlit import, so they are unit-testable).

Every snippet is a single line without leading spaces: Markdown would otherwise turn indented
lines into code blocks.
"""

from __future__ import annotations

from html import escape

from . import theme
from .constants import BANNER_TEXT

APP_CSS = (
    ".block-container{padding-top:2rem;padding-bottom:3rem;max-width:1500px}"
    "footer{visibility:hidden}"
    f".dz-title{{font-size:1.75rem;font-weight:700;color:{theme.NAVY};letter-spacing:-0.01em;"
    "line-height:1.15;margin:0}"
    f".dz-subtitle{{color:{theme.GREY};font-size:.9rem;margin-top:.25rem}}"
    f".dz-coverage{{background:{theme.SURFACE};border:1px solid {theme.BORDER};border-radius:10px;"
    f"padding:.55rem .9rem;color:{theme.NAVY};font-size:.85rem;text-align:right}}"
    f".dz-banner{{background:#FFF6E0;border:1px solid #F0D58A;border-left:5px solid {theme.AMBER};"
    "color:#7A5A00;border-radius:8px;padding:.6rem .9rem;font-weight:600;font-size:.88rem;"
    "margin:.6rem 0 1rem}"
    f".dz-card{{background:{theme.WHITE};border:1px solid {theme.BORDER};border-top:4px solid {theme.PRIMARY};"
    "border-radius:12px;padding:.8rem 1rem .85rem;box-shadow:0 1px 2px rgba(23,50,77,.06);height:100%}"
    f".dz-card--danger{{border-top-color:{theme.DEEP_RED}}}"
    f".dz-card--warn{{border-top-color:{theme.RED}}}"
    f".dz-card--info{{border-top-color:{theme.LOW_BLUE}}}"
    f".dz-card--neutral{{border-top-color:{theme.NAVY}}}"
    f".dz-card__label{{font-size:.78rem;color:{theme.GREY};font-weight:600}}"
    f".dz-card__value{{font-size:1.65rem;font-weight:700;color:{theme.NAVY};line-height:1.2;margin-top:.15rem}}"
    f".dz-card__note{{font-size:.72rem;color:{theme.GREY};margin-top:.15rem;min-height:1em}}"
    f".dz-section{{font-size:1.05rem;font-weight:700;color:{theme.NAVY};margin:1.3rem 0 .1rem}}"
    f".dz-caption{{font-size:.8rem;color:{theme.GREY};margin-bottom:.5rem}}"
)

TONES = ("default", "danger", "warn", "info", "neutral")


def banner_html() -> str:
    """Decision D4: the original wording, shown on every page."""
    return f'<div class="dz-banner" role="note">{escape(BANNER_TEXT)}</div>'


def title_html(title: str, subtitle: str = "") -> str:
    sub = f'<div class="dz-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="dz-title">{escape(title)}</div>{sub}'


def coverage_html(text: str) -> str:
    return f'<div class="dz-coverage">{escape(text)}</div>'


def kpi_card_html(label: str, value: str, note: str = "", tone: str = "default") -> str:
    if tone not in TONES:
        raise ValueError(f"unknown tone: {tone}")
    cls = "dz-card" if tone == "default" else f"dz-card dz-card--{tone}"
    return (
        f'<div class="{cls}"><div class="dz-card__label">{escape(label)}</div>'
        f'<div class="dz-card__value">{escape(value)}</div>'
        f'<div class="dz-card__note">{escape(note)}</div></div>'
    )


def section_html(text: str, caption: str = "") -> str:
    cap = f'<div class="dz-caption">{escape(caption)}</div>' if caption else ""
    return f'<div class="dz-section">{escape(text)}</div>{cap}'
