"""HTML tables of the Sales Intelligence page (pure strings) and the product-grid formats."""

import re

import pandas as pd
import pytest

from dzmi_app import ui_tables

COLUMNS = [("label", "Month", str), ("net", "Net Sales", ui_tables.fmt_int),
           ("pct", "Net Sales MoM %", ui_tables.fmt_pct1)]


def cells(html, tag="td"):
    return re.findall(rf"<{tag}>(.*?)</{tag}>", html)


def test_matrix_shows_headers_rows_and_a_total_row():
    html = ui_tables.matrix_html(
        "Monthly performance", COLUMNS,
        [{"label": "2025-03", "net": 16978045.5, "pct": 0.5715}, {"label": "2025-04", "net": 12188345.0, "pct": -0.2821}],
        {"label": "Total", "net": 40389951.0, "pct": None})
    assert cells(html, "th") == ["Month", "Net Sales", "Net Sales MoM %"]
    assert cells(html)[:6] == ["2025-03", "16,978,046", "57.2%", "2025-04", "12,188,345", "-28.2%"]
    assert 'class="dz-total"' in html
    assert re.findall(r'<tr class="dz-total">(.*?)</tr>', html)[0] == "<td>Total</td><td>40,389,951</td><td></td>"
    assert "Monthly performance" in html


def test_blank_values_are_empty_cells_like_power_bi():
    html = ui_tables.matrix_html("t", COLUMNS, [{"label": "2025-02", "net": 1.0, "pct": None}], None)
    assert cells(html)[2] == ""
    assert "dz-total" not in html.replace("dz-total{", "")      # no total row requested


def test_matrix_without_rows_still_renders_headers_and_total():
    html = ui_tables.matrix_html("t", COLUMNS, [], {"label": "Total", "net": None, "pct": None})
    assert cells(html, "th") == ["Month", "Net Sales", "Net Sales MoM %"]
    assert re.findall(r'<tr class="dz-total">(.*?)</tr>', html)[0] == "<td>Total</td><td></td><td></td>"


def test_text_is_escaped():
    html = ui_tables.matrix_html("<b>t</b>", COLUMNS, [{"label": "<i>x</i>", "net": 1.0, "pct": None}], None)
    assert "<b>t</b>" not in html and "<i>x</i>" not in html
    assert "&lt;i&gt;x&lt;/i&gt;" in html


def test_total_strip():
    html = ui_tables.total_strip_html([("Net Sales", "40,389,951"), ("Gross Margin %", "16.9%")])
    assert "<b>Total</b>" in html and "Net Sales <b>40,389,951</b>" in html and "16.9%" in html
    assert "<script>" not in ui_tables.total_strip_html([("<script>", "<script>")])


def test_snippets_are_single_unindented_lines():
    html = ui_tables.matrix_html("t", COLUMNS, [{"label": "a", "net": 1.0, "pct": 0.1}], {"label": "T", "net": 1.0, "pct": 0.1})
    for s in (html, ui_tables.total_strip_html([("a", "b")]), ui_tables.TABLE_CSS):
        assert "\n" not in s and not s.startswith(" ")
    assert "http" not in ui_tables.TABLE_CSS and "url(" not in ui_tables.TABLE_CSS


def test_product_grid_uses_the_model_format_strings():
    pytest.importorskip("jinja2")                      # pandas Styler needs it (installed with Streamlit)
    df = pd.DataFrame({"product_name": ["p"], "category_name": ["c"], "net_sales": [1234567.4],
                       "gross_profit": [1000.0], "gross_margin": [0.169], "avg_selling_price": [12.346],
                       "units_sold": [1200], "promo_share": [float("nan")]})
    html = ui_tables.product_styler(df).to_html()
    values = re.findall(r"<td[^>]*>([^<]*)</td>", html)
    assert values == ["p", "c", "1,234,567", "1,000", "16.9%", "12.35", "1,200", ""]
    headers = re.findall(r"<th[^>]*>([^<]*)</th>", html)
    assert "Gross Margin %" in headers and "Promo Sales Share" in headers and "Average Selling Price" in headers


def test_styling_does_not_change_the_underlying_numbers():
    pytest.importorskip("jinja2")
    df = pd.DataFrame({"product_name": ["p"], "category_name": ["c"], "net_sales": [1234567.4],
                       "gross_profit": [1.0], "gross_margin": [0.169], "avg_selling_price": [12.346],
                       "units_sold": [1200], "promo_share": [float("nan")]})
    styled = ui_tables.product_styler(df)
    assert styled.data["Net Sales"].iloc[0] == 1234567.4
    assert pd.isna(styled.data["Promo Sales Share"].iloc[0])
