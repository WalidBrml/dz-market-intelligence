"""Extra display formats for pages 3-6 (formatting.py is unchanged).

* ``0.000``                 -> fmt_decimal(x, 3)         * ``#,0.0`` -> fmt_number(x, 1) (formatting.py)
* ``+0.0%;-0.0%;0.0%``      -> fmt_signed_percent(x, 1)  * ``0.0\\ %`` -> fmt_percent_spaced(x, 1)
A blank (None / NaN) is shown as an empty string, as Power BI does in tables and matrices.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .formatting import is_blank


def _round(x: float, decimals: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def fmt_decimal(x, decimals: int = 3) -> str:
    """Plain fixed decimals without thousands separators (format string ``0.000``)."""
    if is_blank(x):
        return ""
    value = _round(x, decimals)
    return f"{abs(value) if value == 0 else value:.{decimals}f}"      # never "-0.000"


def fmt_signed_percent(x, decimals: int = 1) -> str:
    """``+0.0%;-0.0%;0.0%``: explicit sign for non-zero values."""
    if is_blank(x):
        return ""
    value = _round(float(x) * 100, decimals)
    body = f"{abs(value):.{decimals}f}%"
    return f"+{body}" if value > 0 else (f"-{body}" if value < 0 else body)


def fmt_percent_spaced(x, decimals: int = 1) -> str:
    """Format string ``0.0\\ %`` of the At-Risk SKU Share measure (a space before the percent sign)."""
    return "" if is_blank(x) else f"{_round(float(x) * 100, decimals):.{decimals}f} %"


def fmt_text_or_blank(x) -> str:
    return "" if is_blank(x) else str(x)
