"""Stock-constrained simulation: converts latent demand into OBSERVED sales.

Policy (a simple reorder-point / order-up-to rule, NOT claimed optimal):
  - The retailer estimates demand with an EWMA of OBSERVED (censored) sales.
  - Orders are placed at end of day when inventory position <= reorder point.
  - Orders arrive after a store/product-specific lead time, rounded up to supplier case size.
  - Occasionally suppliers short-ship (partial fill). Excess demand is lost (no backorders).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InventoryResult:
    opening: np.ndarray
    received: np.ndarray
    sold: np.ndarray
    closing: np.ndarray
    on_order: np.ndarray


def simulate_inventory(latent: np.ndarray, lam_init: np.ndarray, lead: np.ndarray, pack: np.ndarray,
                       inv_cfg: dict, rng: np.random.Generator) -> InventoryResult:
    n_sp, n_days = latent.shape
    size = int(lead.max()) + 1
    buf = np.zeros((n_sp, size), dtype=np.int64)
    rows = np.arange(n_sp)
    rate = lam_init.astype(float).copy()
    z, cover, alpha = inv_cfg["service_z"], inv_cfg["cover_days"], inv_cfg["ewma_alpha"]
    p_short = inv_cfg["supplier_shortfall_prob"]
    fill_lo, fill_hi = inv_cfg["supplier_shortfall_min_fill"], inv_cfg["supplier_shortfall_max_fill"]

    def targets(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizon = lead + 1.0
        rop = r * horizon + z * np.sqrt(np.maximum(r * horizon, 0.0) * 2.0)
        return rop, rop + r * cover

    _, s0 = targets(rate)
    stock = (np.ceil(s0 / pack) * pack).astype(np.int64)
    stock = np.maximum(stock, pack)

    opening = np.zeros((n_sp, n_days), dtype=np.int32)
    received = np.zeros_like(opening)
    sold = np.zeros_like(opening)
    closing = np.zeros_like(opening)
    on_order = np.zeros_like(opening)

    for d in range(n_days):
        slot = d % size
        arrive = buf[:, slot].copy()
        buf[:, slot] = 0
        short = (rng.random(n_sp) < p_short) & (arrive > 0)
        fill = rng.uniform(fill_lo, fill_hi, n_sp)
        arrive = np.where(short, np.floor(arrive * fill), arrive).astype(np.int64)

        avail = stock + arrive
        s = np.minimum(latent[:, d].astype(np.int64), avail)
        close = avail - s

        opening[:, d], received[:, d], sold[:, d], closing[:, d] = stock, arrive, s, close
        rate = (1 - alpha) * rate + alpha * s

        pos = close + buf.sum(axis=1)
        rop, up_to = targets(rate)
        need = pos <= rop
        qty = np.where(need, np.ceil(np.maximum(up_to - pos, 1) / pack) * pack, 0).astype(np.int64)
        m = qty > 0
        buf[rows[m], (d + lead[m]) % size] += qty[m]
        on_order[:, d] = buf.sum(axis=1)
        stock = close

    return InventoryResult(opening, received, sold, closing, on_order)
