"""SIMULATION-ONLY diagnostic: compare forecasts with the simulator's latent demand.

This is the ONLY module that reads ops.sim_latent_demand. Latent demand does not exist in real life (lost sales are not
observed), so it is used exclusively to MEASURE how much a stockout-censored history hurts, never to train, tune or
select a model. Results are stored under evaluation_basis = 'latent_simulation_only' and must not be quoted as
real-world accuracy.
"""
from __future__ import annotations

import numpy as np

from ..features.panel import Panel
from ..ingestion.db import Database


def load_latent(db: Database, panel: Panel) -> np.ndarray:
    """(n_series, T) latent demand aligned with the panel, NaN where the simulator has no row."""
    df = db.query_df(
        "SELECT s.store_key, p.product_key, d.date_key, l.latent_demand_units "
        "FROM ops.sim_latent_demand l "
        "JOIN dwh.dim_store s ON s.store_id = l.store_id JOIN dwh.dim_product p ON p.product_id = l.product_id "
        "JOIN dwh.dim_date d ON d.full_date = l.demand_date")
    out = np.full((panel.n_series, panel.n_days), np.nan)
    if df.empty:
        return out
    key = lambda a, b: a.to_numpy(dtype=np.int64) * 1_000_000 + b.to_numpy(dtype=np.int64)
    import pandas as pd
    rows = pd.Index(key(panel.series["store_key"], panel.series["product_key"])).get_indexer(key(df["store_key"], df["product_key"]))
    date_keys = (panel.dates.year * 10000 + panel.dates.month * 100 + panel.dates.day).to_numpy(dtype=np.int64)
    cols = np.searchsorted(date_keys, df["date_key"].to_numpy(dtype=np.int64))
    keep = (rows >= 0) & (cols < len(date_keys))
    keep[keep] &= date_keys[cols[keep]] == df["date_key"].to_numpy(dtype=np.int64)[keep]
    out[rows[keep], cols[keep]] = df["latent_demand_units"].to_numpy(dtype=float)[keep]
    return out
