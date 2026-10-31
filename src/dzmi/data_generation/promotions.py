"""Promotion campaigns. At most ONE promotion per store x product x day (occupancy-checked)."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

TYPES = ["Seasonal discount", "Weekend deal", "Clearance", "Loyalty offer"]


def build_promotions(
    cfg, cal: pd.DataFrame, sp_store_idx: np.ndarray, sp_prod_idx: np.ndarray,
    store_ids: np.ndarray, product_ids: np.ndarray, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (promotion, promotion_scope, promo_id[n_sp, n_days], discount_pct[n_sp, n_days])."""
    pc = cfg.promotions
    n_sp, n_days = len(sp_store_idx), cfg.n_days
    promo_id = np.zeros((n_sp, n_days), dtype=np.int32)
    disc = np.zeros((n_sp, n_days), dtype=np.float32)
    occupied = np.zeros((n_sp, n_days), dtype=bool)

    start = pd.Timestamp(cfg.start_date)
    n_years = n_days / 365.25
    n_campaigns = max(1, int(round(pc["campaigns_per_year"] * n_years)))
    campaigns: list[tuple[int, int, str, float]] = []  # (start_idx, length, type, pct)
    starts = np.sort(rng.integers(0, max(1, n_days - pc["min_days"]), size=n_campaigns))
    for s in starts:
        length = int(rng.integers(pc["min_days"], pc["max_days"] + 1))
        campaigns.append((int(s), length, str(rng.choice(TYPES)), float(rng.choice(pc["discount_choices"]))))
    if pc.get("ramadan_campaign", True):
        for a, _ in cfg.calendar["ramadan"]:
            idx = (pd.Timestamp(a) - start).days - 3
            if 0 <= idx < n_days:
                campaigns.append((idx, 14, "Ramadan campaign", float(rng.choice([10, 15, 20]))))
    campaigns.sort(key=lambda c: c[0])

    promo_rows, scope_rows = [], []
    pid = 0
    for s_idx, length, ptype, pct in campaigns:
        e_idx = min(n_days - 1, s_idx + length - 1)
        n_prod = max(1, int(round(pc["product_share"] * len(product_ids))))
        n_store = max(1, int(round(pc["store_share"] * len(store_ids))))
        prods = rng.choice(len(product_ids), size=n_prod, replace=False)
        stores = rng.choice(len(store_ids), size=n_store, replace=False)
        mask = np.isin(sp_prod_idx, prods) & np.isin(sp_store_idx, stores)
        mask &= ~occupied[:, s_idx:e_idx + 1].any(axis=1)
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            continue
        pid += 1
        sl = slice(s_idx, e_idx + 1)
        occupied[np.ix_(rows, np.arange(s_idx, e_idx + 1))] = True
        promo_id[rows, sl] = pid
        disc[rows, sl] = pct
        d0 = start + timedelta(days=s_idx)
        promo_rows.append(dict(
            promotion_id=pid, promotion_name=f"{ptype} {pct:.0f}% #{pid:03d}", promotion_type=ptype,
            discount_pct=pct, start_date=d0, end_date=start + timedelta(days=e_idx)))
        scope_rows.append(pd.DataFrame({
            "promotion_id": pid, "store_id": store_ids[sp_store_idx[rows]], "product_id": product_ids[sp_prod_idx[rows]]}))
    promotion = pd.DataFrame(promo_rows, columns=[
        "promotion_id", "promotion_name", "promotion_type", "discount_pct", "start_date", "end_date"])
    scope = (pd.concat(scope_rows, ignore_index=True) if scope_rows
             else pd.DataFrame(columns=["promotion_id", "store_id", "product_id"]))
    return promotion, scope, promo_id, disc
