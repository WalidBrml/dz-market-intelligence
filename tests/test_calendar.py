import pandas as pd

from dzmi.config import load_calendar
from dzmi.data_generation.calendar_features import build_calendar
from datetime import date


def _cal(a=date(2025, 2, 1), b=date(2025, 5, 1)):
    return build_calendar(a, b, load_calendar())


def test_gap_free_daily_series():
    c = _cal(date(2024, 1, 1), date(2024, 12, 31))
    assert len(c) == 366                                   # leap year
    assert (c["full_date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert c["date_key"].is_unique


def test_date_key_format():
    c = _cal()
    assert c.loc[0, "date_key"] == 20250201 and c["date_key"].iloc[-1] == 20250501


def test_algerian_weekend_is_friday_saturday():
    c = _cal()
    wk = c[c["is_weekend"]]["day_name"].unique().tolist()
    assert sorted(wk) == ["Friday", "Saturday"]


def test_ramadan_2025_window_and_day_counter():
    c = _cal().set_index("full_date")
    assert not c.loc["2025-02-28", "is_ramadan"] and c.loc["2025-03-01", "is_ramadan"]
    assert c.loc["2025-03-29", "is_ramadan"] and not c.loc["2025-03-30", "is_ramadan"]
    assert c.loc["2025-03-01", "ramadan_day"] == 1 and c.loc["2025-03-29", "ramadan_day"] == 29
    assert pd.isna(c.loc["2025-03-30", "ramadan_day"])


def test_eid_and_pre_eid_flags():
    c = _cal().set_index("full_date")
    assert c.loc["2025-03-30", "is_eid_al_fitr"] and c.loc["2025-03-31", "is_eid_al_fitr"]
    assert c.loc["2025-03-23", "is_pre_eid_al_fitr"] and not c.loc["2025-03-30", "is_pre_eid_al_fitr"]
    assert not c["is_eid_al_adha"].any()                   # Adha 2025 is in June, outside the smoke window
