"""Additional sales measures used by the Sales Intelligence page, ported from _Measures.tmdl.

Reused unchanged from measures_sales: Net Sales, Gross Profit, Units Sold, Gross Margin %, DIVIDE.

    Average Selling Price   = DIVIDE ( [Net Sales], [Units Sold] )
    Promo Net Sales         = CALCULATE ( [Net Sales], FILTER ( fact_sales, NOT ISBLANK ( fact_sales[promotion_key] ) ) )
    Promo Sales Share       = DIVIDE ( [Promo Net Sales], [Net Sales] )
    Sales Days              = CALCULATE ( DISTINCTCOUNT ( fact_sales[date_key] ) )
    Average Daily Net Sales = DIVIDE ( [Net Sales], [Sales Days] )
    Net Sales per sqm       = DIVIDE ( [Net Sales], SUM ( dim_store[floor_area_sqm] ) )
    Net Sales Prior Month   = CALCULATE ( [Net Sales], DATEADD ( dim_date[full_date], -1, MONTH ) )
    Net Sales MoM %         =
        VAR _cur_ok  = CALCULATE ( MIN ( dim_date[is_complete_month] ) )
        VAR _prev    = [Net Sales Prior Month]
        VAR _prev_ok = CALCULATE ( MIN ( dim_date[is_complete_month] ), DATEADD ( dim_date[full_date], -1, MONTH ) )
        RETURN IF ( _cur_ok = 1 && _prev_ok = 1 && NOT ISBLANK ( _prev ) && _prev <> 0,
                    DIVIDE ( [Net Sales] - _prev, _prev ) )

``None`` stands for DAX BLANK(). The frame arguments are the equivalent of a filter context.
"""

from __future__ import annotations

import calendar
from datetime import date

import pandas as pd

from .measures_sales import divide, net_sales, units_sold


# --------------------------------------------------------------------------- scalar measures


def average_selling_price(sales: pd.DataFrame):
    return divide(net_sales(sales), units_sold(sales))


def promo_net_sales(sales: pd.DataFrame):
    """Net Sales of the lines that carry a promotion_key; BLANK when there are none."""
    return net_sales(sales[sales["promotion_key"].notna()])


def promo_sales_share(sales: pd.DataFrame):
    return divide(promo_net_sales(sales), net_sales(sales))


def sales_days(sales: pd.DataFrame) -> int:
    return int(sales["date_key"].nunique())


def average_daily_net_sales(sales: pd.DataFrame):
    return divide(net_sales(sales), sales_days(sales))


def net_sales_per_sqm(sales: pd.DataFrame, floor_area_sqm):
    """``floor_area_sqm`` is SUM(dim_store[floor_area_sqm]) in the same filter context."""
    return divide(net_sales(sales), floor_area_sqm)


# --------------------------------------------------------------------------- grouped measures


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorised DIVIDE: NaN (BLANK) when the denominator is 0 or BLANK."""
    den = denominator.where(denominator != 0)
    return numerator / den


def measures_by(sales: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Net Sales, Gross Profit, Units, Gross Margin %, ASP, Promo Net Sales / Share, Sales Days and
    Average Daily Net Sales for every group of ``by`` that has at least one sales line.

    Equivalent to evaluating each measure in the filter context of one group (a visual row).
    """
    g = sales.groupby(by, dropna=False)
    out = g.agg(net_sales=("net_sales", "sum"), gross_profit=("gross_profit", "sum"),
                units_sold=("quantity", "sum"), sales_days=("date_key", "nunique")).reset_index()
    promo = (sales[sales["promotion_key"].notna()].groupby(by, dropna=False)["net_sales"].sum()
             .rename("promo_net_sales").reset_index())
    out = out.merge(promo, on=by, how="left")           # NaN = BLANK when a group has no promo line
    out["gross_margin"] = _ratio(out["gross_profit"], out["net_sales"])
    out["avg_selling_price"] = _ratio(out["net_sales"], out["units_sold"])
    out["promo_share"] = _ratio(out["promo_net_sales"], out["net_sales"])
    out["avg_daily_net_sales"] = _ratio(out["net_sales"], out["sales_days"])
    return out


# --------------------------------------------------------------------------- DATEADD and MoM


def _one_month_back(d: date, month_end_rule: bool) -> date:
    year, month = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    last = calendar.monthrange(year, month)[1]
    if month_end_rule and d.day == calendar.monthrange(d.year, d.month)[1]:
        return date(year, month, last)
    return date(year, month, min(d.day, last))


def dateadd_back_one_month(start: date, end: date) -> tuple[date, date]:
    """DATEADD ( dim_date[full_date], -1, MONTH ) for the contiguous dates start..end.

    The first day moves to the same day one month earlier (clamped to that month's last day). The
    last day does the same, except that a month-end maps to the previous month's end. The
    month-end rule is the documented behaviour of DAX time-intelligence shifts on whole months
    but it could not be executed here (no Power BI Desktop): it is an assumption, see B2 notes.
    Dates that do not exist in dim_date simply drop out of the filter.
    """
    return _one_month_back(start, False), _one_month_back(end, True)


def _min_complete(dates: pd.DataFrame, start: date, end: date):
    """MIN(dim_date[is_complete_month]) over the dim_date rows in [start, end]; None if no rows."""
    window = dates[(dates["full_date"] >= pd.Timestamp(start)) & (dates["full_date"] <= pd.Timestamp(end))]
    return None if len(window) == 0 else int(window["is_complete_month"].min())


def net_sales_mom(dates: pd.DataFrame, sales_without_date_filter: pd.DataFrame,
                  current_net_sales, start: date, end: date):
    """Net Sales MoM % in the filter context whose dates are ``start``..``end``.

    ``dates``: the whole dim_date (full_date, is_complete_month). ``sales_without_date_filter``: the
    sales lines after every non-date filter (DATEADD replaces the date filter, so the prior-month
    sum is not limited by the date range). ``current_net_sales``: [Net Sales] in the current context
    (None = BLANK, which DAX subtracts as 0). Returns None (BLANK) unless both periods are complete.
    """
    cur_ok = _min_complete(dates, start, end)
    p_start, p_end = dateadd_back_one_month(start, end)
    prev_ok = _min_complete(dates, p_start, p_end)
    in_prev = sales_without_date_filter[
        (sales_without_date_filter["full_date"] >= pd.Timestamp(p_start))
        & (sales_without_date_filter["full_date"] <= pd.Timestamp(p_end))]
    prev = net_sales(in_prev)
    if cur_ok == 1 and prev_ok == 1 and prev is not None and prev != 0:
        return ((current_net_sales or 0.0) - prev) / prev
    return None


def sort_by_mom_desc(frame: pd.DataFrame, label: str = "year_month_label") -> pd.DataFrame:
    """Matrix sort on Net Sales MoM % descending: BLANK sorts last; ties keep month order."""
    return (frame.assign(_blank=frame["mom"].isna())
            .sort_values(["_blank", "mom", label], ascending=[True, False, True], kind="stable")
            .drop(columns="_blank").reset_index(drop=True))
