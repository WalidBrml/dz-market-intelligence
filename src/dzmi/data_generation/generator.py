"""End-to-end deterministic generation of the operational-style (ops) tables."""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from .. import GENERATOR_VERSION, SYNTHETIC_DISCLAIMER
from ..config import GenConfig
from . import catalog
from .calendar_features import build_calendar
from .demand import base_rates, category_day_multipliers, draw_latent_demand
from .inventory_sim import simulate_inventory
from .promotions import build_promotions
from .transactions import build_transactions

log = logging.getLogger(__name__)

# Independent random streams so changing one component does not shift the others.
_STREAMS = ["catalog", "stores", "customers", "assortment", "prices", "promotions", "demand", "inventory", "baskets"]


def _rngs(seed: int) -> dict[str, np.random.Generator]:
    kids = np.random.SeedSequence(seed).spawn(len(_STREAMS))
    return {n: np.random.default_rng(s) for n, s in zip(_STREAMS, kids)}


def generate(cfg: GenConfig) -> dict[str, pd.DataFrame]:
    """Generate all ops tables. Same config + seed => identical output."""
    rng = _rngs(cfg.seed)
    cal = build_calendar(cfg.start_date, cfg.end_date, cfg.calendar)
    n_days = cfg.n_days
    log.info("Generating profile=%s: %d stores x %d products x %d days", cfg.profile, cfg.n_stores,
             cfg.n_products, n_days)

    categories = catalog.build_categories()
    products = catalog.build_products(cfg.n_products, rng["catalog"])
    stores = catalog.build_stores(cfg.n_stores, cfg.start_date, rng["stores"])
    customers = catalog.build_customers(cfg.n_customers, stores, cfg.start_date, rng["customers"])
    store_ids, product_ids = stores["store_id"].to_numpy(), products["product_id"].to_numpy()

    # --- assortment: which products each store carries; defines the store x product series
    n_s, n_p = len(stores), len(products)
    carry = rng["assortment"].random((n_s, n_p)) < cfg.assortment_fraction
    for s in range(n_s):  # every store carries at least one product from each category
        for c in products["category_id"].unique():
            cols = np.flatnonzero(products["category_id"].to_numpy() == c)
            if not carry[s, cols].any():
                carry[s, rng["assortment"].choice(cols)] = True
    sp_store_idx, sp_prod_idx = np.nonzero(carry)
    n_sp = len(sp_store_idx)
    lead = rng["assortment"].integers(cfg.inventory["lead_time_min"], cfg.inventory["lead_time_max"] + 1, n_sp)
    assortment = pd.DataFrame({"store_id": store_ids[sp_store_idx], "product_id": product_ids[sp_prod_idx],
                               "lead_time_days": lead})

    # --- prices: list price and cost step up each calendar year (inflation assumption)
    years = np.arange(cfg.start_date.year, cfg.end_date.year + 1)
    infl = rng["prices"].uniform(cfg.prices["annual_inflation_min"], cfg.prices["annual_inflation_max"], n_p)
    steps = (1 + infl)[:, None] ** (years - years[0])[None, :]
    price_by_year = np.maximum(np.round(products["base_price"].to_numpy()[:, None] * steps), 1.0)
    cost_by_year = np.round(products["base_cost"].to_numpy()[:, None] * steps, 2)
    cost_by_year = np.minimum(cost_by_year, price_by_year - 0.01)  # keep cost < price after rounding
    year_idx = (cal["year"].to_numpy() - years[0]).astype(int)
    hist = []
    for yi, y in enumerate(years):
        hist.append(pd.DataFrame({"product_id": product_ids, "valid_from": pd.Timestamp(date(int(y), 1, 1)),
                                  "list_price": price_by_year[:, yi], "unit_cost": cost_by_year[:, yi]}))
    price_history = pd.concat(hist, ignore_index=True)

    # --- promotions, latent demand, stock-constrained observed sales
    promotion, scope, promo_id, disc = build_promotions(cfg, cal, sp_store_idx, sp_prod_idx, store_ids,
                                                        product_ids, rng["promotions"])
    cat_idx = products["category_id"].to_numpy()[sp_prod_idx] - 1
    lam0 = base_rates(cfg, stores, products, sp_store_idx, sp_prod_idx, rng["demand"])
    latent = draw_latent_demand(cfg, lam0, cat_idx, category_day_multipliers(cfg, cal), disc, rng["demand"])
    inv = simulate_inventory(latent, lam0, lead.astype(np.int64),
                             products["pack_size"].to_numpy()[sp_prod_idx].astype(np.int64),
                             cfg.inventory, rng["inventory"])
    assert (inv.sold <= latent).all()

    dates = cal["full_date"].to_numpy()
    sp_rows = np.repeat(np.arange(n_sp), n_days)
    day_rows = np.tile(np.arange(n_days), n_sp)
    inventory_daily = pd.DataFrame({
        "store_id": store_ids[sp_store_idx][sp_rows], "product_id": product_ids[sp_prod_idx][sp_rows],
        "inventory_date": dates[day_rows],
        "opening_stock": inv.opening.ravel(), "received_qty": inv.received.ravel(),
        "units_sold": inv.sold.ravel(), "closing_stock": inv.closing.ravel(),
        "on_order_qty": inv.on_order.ravel(),
    })
    inventory_daily["stockout_flag"] = inventory_daily["closing_stock"] == 0
    latent_df = pd.DataFrame({
        "store_id": inventory_daily["store_id"], "product_id": inventory_daily["product_id"],
        "demand_date": inventory_daily["inventory_date"], "latent_demand_units": latent.ravel(),
        "unmet_demand_units": (latent - inv.sold).ravel(),
    })

    tx, lines = build_transactions(cfg, inv.sold, sp_store_idx, sp_prod_idx, store_ids, product_ids,
                                   price_by_year, year_idx, promo_id, disc, customers, rng["baskets"])

    provenance = pd.DataFrame({
        "provenance_key": ["is_synthetic", "disclaimer", "profile", "seed", "config_fingerprint",
                           "generator_version", "currency", "calendar_note"],
        "provenance_value": ["true", SYNTHETIC_DISCLAIMER, cfg.profile, str(cfg.seed), cfg.fingerprint(),
                             GENERATOR_VERSION, "DZD",
                             "Ramadan/Eid dates are approximate and configurable (config/calendar.yaml)."],
    })
    out = {
        "data_provenance": provenance,
        "category": categories,
        "product": products[["product_id", "sku", "product_name", "category_id", "brand", "pack_size"]],
        "product_price_history": price_history,
        "store": stores,
        "store_assortment": assortment,
        "customer": customers,
        "promotion": promotion,
        "promotion_scope": scope,
        "sales_transaction": tx,
        "sales_transaction_line": lines,
        "inventory_daily": inventory_daily,
        "sim_latent_demand": latent_df,
    }
    log.info("Generated %s transactions, %s lines, %s inventory rows", f"{len(tx):,}", f"{len(lines):,}",
             f"{len(inventory_daily):,}")
    return out
