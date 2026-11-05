"""Display formatting that mirrors the Power BI format strings of the Page 1 cards.

* ``#,0.0,,"M"`` -> fmt_millions(x, 1)      * ``#,0.0,"K"`` -> fmt_thousands(x, 1)
* ``#,0.00,,"M"`` -> fmt_millions(x, 2)     * ``#,0.00``    -> fmt_number(x, 2)
* ``0.0%`` / ``0.00%`` -> fmt_percent(x, 1 / 2)

Rounding is half away from zero (as Power BI/Excel do), not Python's banker's rounding.
A blank value (None / NaN) is shown as an em dash.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

BLANK = "\u2014"


def is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _round(x: float, decimals: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def fmt_number(x, decimals: int = 0) -> str:
    """``#,0`` / ``#,0.00`` style number with thousands separators."""
    if is_blank(x):
        return BLANK
    return f"{_round(x, decimals):,.{decimals}f}"


def fmt_millions(x, decimals: int = 1) -> str:
    if is_blank(x):
        return BLANK
    return f"{_round(float(x) / 1_000_000, decimals):,.{decimals}f}M"


def fmt_thousands(x, decimals: int = 1) -> str:
    if is_blank(x):
        return BLANK
    return f"{_round(float(x) / 1_000, decimals):,.{decimals}f}K"


def fmt_percent(x, decimals: int = 1) -> str:
    if is_blank(x):
        return BLANK
    return f"{_round(float(x) * 100, decimals):,.{decimals}f}%"


def fmt_count(x) -> str:
    """Integer count; a blank count is shown as 0 (see deviation V13)."""
    return f"{0 if is_blank(x) else int(x):,}"
