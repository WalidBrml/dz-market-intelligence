"""Calendar table with Algerian weekend and (approximate, configurable) Ramadan/Eid flags."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _in_ranges(dates: pd.DatetimeIndex, ranges: list[tuple[date, date]]) -> np.ndarray:
    flag = np.zeros(len(dates), dtype=bool)
    for a, b in ranges:
        flag |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return flag


def build_calendar(start: date, end: date, cal: dict[str, Any]) -> pd.DataFrame:
    """One row per day in [start, end] with calendar and religious-event flags."""
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"full_date": dates})
    df["date_key"] = (dates.year * 10000 + dates.month * 100 + dates.day).astype("int64")
    df["year"] = dates.year
    df["quarter"] = dates.quarter
    df["month"] = dates.month
    df["month_name"] = [MONTH_NAMES[m - 1] for m in dates.month]
    df["day_of_month"] = dates.day
    df["day_of_week"] = dates.dayofweek + 1
    df["day_name"] = [DAY_NAMES[d] for d in dates.dayofweek]
    df["iso_week"] = dates.isocalendar().week.astype("int64").to_numpy()
    df["is_weekend"] = np.isin(dates.dayofweek, cal.get("weekend_weekdays", [4, 5]))

    df["is_ramadan"] = _in_ranges(dates, cal["ramadan"])
    ramadan_day = np.full(len(dates), np.nan)
    for a, b in cal["ramadan"]:
        m = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
        ramadan_day[m] = (dates[m] - pd.Timestamp(a)).days + 1
    df["ramadan_day"] = pd.array(ramadan_day, dtype="Int64")

    df["is_eid_al_fitr"] = _in_ranges(dates, cal["eid_al_fitr"])
    df["is_eid_al_adha"] = _in_ranges(dates, cal["eid_al_adha"])
    pre_f = [(a - timedelta(days=cal["pre_eid_al_fitr_days"]), a - timedelta(days=1)) for a, _ in cal["eid_al_fitr"]]
    pre_a = [(a - timedelta(days=cal["pre_eid_al_adha_days"]), a - timedelta(days=1)) for a, _ in cal["eid_al_adha"]]
    df["is_pre_eid_al_fitr"] = _in_ranges(dates, pre_f)
    df["is_pre_eid_al_adha"] = _in_ranges(dates, pre_a)
    return df
