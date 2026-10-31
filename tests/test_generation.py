import numpy as np
import pandas as pd

from dzmi.config import load_config
from dzmi.data_generation.generator import generate
from helpers import smoke


def _same(a: dict, b: dict) -> bool:
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    return True


def test_generation_is_deterministic():
    cfg = load_config("smoke")
    assert _same(generate(cfg), generate(cfg))


def test_different_seed_changes_data():
    a = generate(load_config("smoke"))
    b = generate(load_config("smoke", seed=43))
    assert not a["inventory_daily"]["units_sold"].equals(b["inventory_daily"]["units_sold"])


def test_smoke_dataset_shape():
    cfg, raw, _ = smoke()
    assert len(raw["store"]) == 5 and len(raw["product"]) == 100 and len(raw["category"]) == 14
    assert raw["inventory_daily"]["inventory_date"].nunique() == 90
    assert 20_000 < len(raw["sales_transaction_line"]) < 100_000


def test_algerian_locations_and_dzd():
    _, raw, _ = smoke()
    assert {"Alger", "Blida", "Oran", "Constantine", "Sétif"} <= set(raw["store"]["wilaya"])
    assert raw["data_provenance"].set_index("provenance_key").loc["currency", "provenance_value"] == "DZD"
    assert raw["data_provenance"].set_index("provenance_key").loc["is_synthetic", "provenance_value"] == "true"


def test_observed_sales_never_exceed_latent_demand_or_stock():
    _, raw, _ = smoke()
    inv, lat = raw["inventory_daily"], raw["sim_latent_demand"]
    assert (inv["units_sold"].to_numpy() <= lat["latent_demand_units"].to_numpy()).all()
    assert (inv["units_sold"] <= inv["opening_stock"] + inv["received_qty"]).all()


def test_stockouts_censor_demand_but_do_not_imply_zero_demand():
    _, raw, _ = smoke()
    lat = raw["sim_latent_demand"]
    assert (lat["unmet_demand_units"] > 0).any()           # the smoke set contains lost sales
    inv = raw["inventory_daily"]
    out_days = inv["stockout_flag"].to_numpy()
    assert lat.loc[out_days, "latent_demand_units"].sum() > 0   # demand existed on stocked-out days


def test_line_quantities_conserve_inventory_units():
    _, raw, _ = smoke()
    ln, tx = raw["sales_transaction_line"], raw["sales_transaction"]
    assert ln["quantity"].sum() == raw["inventory_daily"]["units_sold"].sum()
    assert len(ln) > len(tx) and set(ln["transaction_id"]) == set(tx["transaction_id"])


def test_at_most_one_promotion_per_store_product_day():
    _, _, dwh = smoke()
    assert not dwh["fact_promotions"].duplicated(["store_key", "product_key", "date_key"]).any()


def test_costs_are_below_prices():
    _, raw, _ = smoke()
    h = raw["product_price_history"]
    assert (h["unit_cost"] < h["list_price"]).all() and (h["list_price"] > 0).all()


def test_ramadan_lifts_latent_demand_for_dates_category():
    cfg, raw, dwh = smoke()
    lat = raw["sim_latent_demand"].merge(raw["product"][["product_id", "category_id"]], on="product_id")
    dates_cat = raw["category"].set_index("category_name").loc["Dates & Dried Fruit", "category_id"]
    cal = dwh["dim_date"].set_index("full_date")["is_ramadan"]
    lat = lat[lat["category_id"] == dates_cat]
    flag = lat["demand_date"].map(cal)
    assert lat.loc[flag, "latent_demand_units"].mean() > 1.5 * lat.loc[~flag, "latent_demand_units"].mean()


def test_eid_days_are_quiet():
    _, raw, dwh = smoke()
    lat = raw["sim_latent_demand"]
    eid = set(dwh["dim_date"].loc[dwh["dim_date"]["is_eid_al_fitr"], "full_date"])
    normal = lat.loc[~lat["demand_date"].isin(eid), "latent_demand_units"].mean()
    assert lat.loc[lat["demand_date"].isin(eid), "latent_demand_units"].mean() < 0.7 * normal
