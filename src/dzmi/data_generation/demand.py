"""Latent (unconstrained) demand. This is what customers WANT to buy, before stock limits.

Observed sales are produced later by the inventory simulation: sold = min(latent, available).
A day with zero observed sales and zero stock therefore does NOT imply zero demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import CATEGORIES, FORMAT_FACTOR


def category_day_multipliers(cfg, cal: pd.DataFrame) -> np.ndarray:
    """Matrix [n_categories, n_days] of calendar-driven multipliers."""
    d = cfg.demand
    n_days = len(cal)
    wd = np.asarray(d["weekday_multipliers"])[cal["day_of_week"].to_numpy() - 1]
    t_years = np.arange(n_days) / 365.25
    trend = (1 + d["annual_growth"]) ** t_years
    doy = pd.DatetimeIndex(cal["full_date"]).dayofyear.to_numpy()
    seasonal_phase = np.sin(2 * np.pi * (doy - d["summer_peak_day_of_year"] + 91.3) / 365.25)
    ram = cal["is_ramadan"].to_numpy().astype(float)
    pre_f = cal["is_pre_eid_al_fitr"].to_numpy().astype(float)
    pre_a = cal["is_pre_eid_al_adha"].to_numpy().astype(float)
    eid = (cal["is_eid_al_fitr"] | cal["is_eid_al_adha"]).to_numpy()
    out = np.empty((len(CATEGORIES), n_days))
    for c, cat in enumerate(CATEGORIES):
        m = wd * trend
        m = m * (1 + cat["summer"] * seasonal_phase)
        m = m * (1 + (cat["ramadan"] - 1) * ram)
        m = m * (1 + (cat["pre_fitr"] - 1) * pre_f)
        m = m * (1 + (cat["pre_adha"] - 1) * pre_a)
        m = np.where(eid, m * d["eid_day_multiplier"], m)
        out[c] = m
    return out


def base_rates(cfg, stores: pd.DataFrame, products: pd.DataFrame, sp_store_idx, sp_prod_idx,
               rng: np.random.Generator) -> np.ndarray:
    """Per store-product mean daily demand BEFORE calendar multipliers."""
    d = cfg.demand
    pop = np.exp(rng.normal(0, d["popularity_sigma"], len(products)))
    store_eff = np.exp(rng.normal(0, d["store_sigma"], len(stores))) * \
        stores["store_format"].map(FORMAT_FACTOR).to_numpy()
    cat_rate = np.array([CATEGORIES[c - 1]["base_rate"] for c in products["category_id"]])
    lam = d["demand_scale"] * cat_rate[sp_prod_idx] * pop[sp_prod_idx] * store_eff[sp_store_idx]
    return lam


def draw_latent_demand(cfg, lam0: np.ndarray, sp_cat_idx: np.ndarray, cat_mult: np.ndarray,
                       disc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer latent demand [n_sp, n_days] (gamma-Poisson = negative binomial)."""
    elasticity = np.array([c["elasticity"] for c in CATEGORIES])[sp_cat_idx]
    mu = lam0[:, None] * cat_mult[sp_cat_idx, :] * (1.0 + elasticity[:, None] * disc / 100.0)
    k = cfg.demand["dispersion"]
    lam = mu * rng.gamma(shape=k, scale=1.0 / k, size=mu.shape)
    return rng.poisson(lam).astype(np.int32)
