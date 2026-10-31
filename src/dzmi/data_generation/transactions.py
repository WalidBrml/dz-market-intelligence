"""Turn daily store x product unit sales into baskets (transactions) and lines.

Quantities are conserved exactly: sum(line quantity) per store x product x day equals units sold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import PAYMENT_METHODS, PAYMENT_WEIGHTS


def build_transactions(cfg, sold: np.ndarray, sp_store_idx: np.ndarray, sp_prod_idx: np.ndarray,
                       store_ids: np.ndarray, product_ids: np.ndarray, price_by_year: np.ndarray,
                       year_idx: np.ndarray, promo_id: np.ndarray, disc: np.ndarray,
                       customers: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = cfg.baskets
    sp_i, day_i = np.nonzero(sold)
    q = sold[sp_i, day_i].astype(np.int64)

    # 1) split each store-product-day into purchase lines (conserving units)
    extra = rng.binomial(np.maximum(q - 1, 0), b["split_probability"] / 2.0)
    n_parts = 1 + extra
    row = np.repeat(np.arange(len(q)), q)
    part = np.floor(rng.random(row.size) * n_parts[row]).astype(np.int64)
    key = row * (n_parts.max() + 1) + part
    uk, qty = np.unique(key, return_counts=True)
    l_row = uk // (n_parts.max() + 1)
    l_sp, l_day = sp_i[l_row], day_i[l_row]
    store_idx, prod_idx = sp_store_idx[l_sp], sp_prod_idx[l_sp]

    # 2) group lines into baskets per store-day
    n_days = sold.shape[1]
    grp = store_idx.astype(np.int64) * n_days + l_day
    ug, ginv, gcount = np.unique(grp, return_inverse=True, return_counts=True)
    n_baskets = np.maximum(1, np.round(gcount / b["avg_lines_per_basket"]).astype(np.int64))
    slot = np.floor(rng.random(len(grp)) * n_baskets[ginv]).astype(np.int64)
    bkey = grp * 100000 + slot
    # merge same product within a basket (keeps units, avoids duplicate product lines)
    df = pd.DataFrame({"bkey": bkey, "sp": l_sp, "qty": qty, "store_idx": store_idx, "prod_idx": prod_idx, "day": l_day})
    df = df.groupby(["bkey", "sp"], as_index=False, sort=True).agg(
        qty=("qty", "sum"), store_idx=("store_idx", "first"), prod_idx=("prod_idx", "first"), day=("day", "first"))

    # 3) transactions: timestamp, ids, customer
    tx = df.groupby("bkey", as_index=False, sort=True).agg(store_idx=("store_idx", "first"), day=("day", "first"))
    secs = rng.integers(b["open_hour"] * 3600, b["close_hour"] * 3600, len(tx))
    start = pd.Timestamp(cfg.start_date)
    tx["transaction_ts"] = start + pd.to_timedelta(tx["day"].to_numpy(), unit="D") + pd.to_timedelta(secs, unit="s")
    tx = tx.sort_values(["day", "store_idx", "transaction_ts", "bkey"], kind="mergesort").reset_index(drop=True)
    tx["transaction_id"] = np.arange(1, len(tx) + 1, dtype=np.int64)
    tx["store_id"] = store_ids[tx["store_idx"].to_numpy()]
    tx["transaction_date"] = tx["transaction_ts"].dt.normalize()
    tx["payment_method"] = rng.choice(PAYMENT_METHODS, size=len(tx), p=PAYMENT_WEIGHTS)
    cust = np.full(len(tx), np.nan)
    linked = rng.random(len(tx)) < cfg.loyalty_share
    for sid in np.unique(tx["store_id"]):
        pool = customers.loc[customers["home_store_id"] == sid, "customer_id"].to_numpy()
        m = linked & (tx["store_id"].to_numpy() == sid)
        if pool.size and m.any():
            cust[m] = rng.choice(pool, size=int(m.sum()))
    tx["customer_id"] = pd.array(cust, dtype="Int64")

    # 4) lines
    df = df.merge(tx[["bkey", "transaction_id"]], on="bkey", how="left")
    df = df.sort_values(["transaction_id", "prod_idx"], kind="mergesort").reset_index(drop=True)
    df["line_no"] = df.groupby("transaction_id").cumcount() + 1
    sp = df["sp"].to_numpy()
    d = df["day"].to_numpy()
    price = price_by_year[df["prod_idx"].to_numpy(), year_idx[d]]
    pct = disc[sp, d].astype(float)
    pid = promo_id[sp, d]
    qty_l = df["qty"].to_numpy()
    lines = pd.DataFrame({
        "transaction_id": df["transaction_id"].to_numpy(),
        "line_no": df["line_no"].to_numpy(),
        "product_id": product_ids[df["prod_idx"].to_numpy()],
        "quantity": qty_l.astype(np.int64),
        "unit_price": price.astype(float),
        "discount_amount": np.round(qty_l * price * pct / 100.0, 2),
        "promotion_id": pd.array(np.where(pid > 0, pid, -1), dtype="Int64").astype("Int64"),
    })
    lines["promotion_id"] = lines["promotion_id"].where(lines["promotion_id"] > 0, pd.NA)
    tx_out = tx[["transaction_id", "store_id", "customer_id", "transaction_ts", "transaction_date", "payment_method"]]
    return tx_out, lines
