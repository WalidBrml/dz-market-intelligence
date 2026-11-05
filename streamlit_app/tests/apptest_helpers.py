"""Helpers shared by the AppTest page tests (not collected: no test_ prefix)."""

import re
from html import escape, unescape


def card(app, label):
    """Value text of the KPI card with this label."""
    pattern = r'dz-card__label">' + re.escape(escape(label)) + r'</div><div class="dz-card__value">([^<]*)<'
    for md in app.markdown:
        found = re.search(pattern, md.value)
        if found:
            return unescape(found.group(1))
    raise AssertionError(f"card not found: {label}")


def matrix(app, title_fragment):
    """Cell texts of the HTML table whose title contains ``title_fragment`` (header row excluded)."""
    for md in app.markdown:
        if title_fragment in md.value and "<table" in md.value:
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", md.value)
            return [[unescape(c) for c in re.findall(r"<td>(.*?)</td>", r)] for r in rows if "<td>" in r]
    raise AssertionError(f"table not found: {title_fragment}")


def all_markdown(app):
    return " ".join(md.value for md in app.markdown)
