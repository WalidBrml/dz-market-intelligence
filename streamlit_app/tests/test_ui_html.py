"""HTML snippets: the banner (decision D4), cards and escaping. Pure strings."""

import pytest

from dzmi_app import ui_html
from dzmi_app.constants import BANNER_TEXT


def test_banner_contains_the_original_wording_verbatim():
    html = ui_html.banner_html()
    assert BANNER_TEXT in html
    assert "SYNTHETIC DATA - demonstration only. Nothing here describes real Algerian retail demand." in html
    assert 'class="dz-banner"' in html


def test_card_shows_label_value_and_note():
    html = ui_html.kpi_card_html("Net sales (M DZD)", "40.4M", "note")
    assert "Net sales (M DZD)" in html and ">40.4M<" in html and ">note<" in html


@pytest.mark.parametrize("tone,css", [
    ("default", 'class="dz-card"'), ("danger", "dz-card--danger"), ("warn", "dz-card--warn"),
    ("info", "dz-card--info"), ("neutral", "dz-card--neutral"),
])
def test_card_tones(tone, css):
    assert css in ui_html.kpi_card_html("l", "v", tone=tone)


def test_unknown_tone_is_rejected():
    with pytest.raises(ValueError):
        ui_html.kpi_card_html("l", "v", tone="loud")


def test_text_is_escaped():
    html = ui_html.kpi_card_html("Critical (<3 days cover)", "<b>1</b>", "a & b")
    assert "&lt;3 days cover" in html and "&lt;b&gt;1&lt;/b&gt;" in html and "a &amp; b" in html
    assert "<b>" not in html
    assert "&lt;script&gt;" in ui_html.title_html("<script>")
    assert "<script>" not in ui_html.section_html("<script>", "<script>")
    assert "<script>" not in ui_html.coverage_html("<script>")


def test_every_snippet_is_a_single_unindented_line():
    snippets = [
        ui_html.banner_html(), ui_html.title_html("T", "S"), ui_html.coverage_html("c"),
        ui_html.kpi_card_html("l", "v", "n", "warn"), ui_html.section_html("s", "c"), ui_html.APP_CSS,
    ]
    for s in snippets:
        assert "\n" not in s
        assert not s.startswith(" ")


def test_css_loads_nothing_from_the_network():
    assert "http" not in ui_html.APP_CSS and "@import" not in ui_html.APP_CSS and "url(" not in ui_html.APP_CSS


def test_optional_parts_are_omitted_when_empty():
    assert "dz-subtitle" not in ui_html.title_html("T")
    assert "dz-caption" not in ui_html.section_html("s")
