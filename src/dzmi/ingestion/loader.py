"""Load ops and dwh tables into the database in foreign-key order."""
from __future__ import annotations

import logging

import pandas as pd

from ..errors import DatabaseError
from .db import Database

log = logging.getLogger(__name__)

OPS_LOAD_ORDER = ["data_provenance", "category", "product", "product_price_history", "store",
                  "store_assortment", "customer", "promotion", "promotion_scope", "sales_transaction",
                  "sales_transaction_line", "sim_latent_demand"]
DWH_LOAD_ORDER = ["dim_date", "dim_store", "dim_product", "dim_customer", "dim_promotion",
                  "fact_sales", "fact_inventory_daily", "fact_promotions"]


def load_all(db: Database, raw: dict[str, pd.DataFrame], dwh: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Insert all tables. Refuses to load into non-empty tables (use --reset to start clean)."""
    for schema, order in (("ops", OPS_LOAD_ORDER), ("dwh", DWH_LOAD_ORDER)):
        for t in order:
            n = db.row_count(schema, t)
            if n:
                raise DatabaseError(
                    f"{schema}.{t} already has {n:,} rows. Re-run with --reset to drop and reload the "
                    "database (destructive), or point DATABASE_URL at an empty database.")
    counts: dict[str, int] = {}
    for schema, order, src in (("ops", OPS_LOAD_ORDER, raw), ("dwh", DWH_LOAD_ORDER, dwh)):
        for t in order:
            counts[f"{schema}.{t}"] = db.write_df(src[t], schema, t)
            log.info("Loaded %-32s %10s rows", f"{schema}.{t}", f"{counts[f'{schema}.{t}']:,}")
    return counts
