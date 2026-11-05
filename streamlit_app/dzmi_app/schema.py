"""Table schemas, taken from the TMDL partitions of the Power BI semantic model.

Column spec mini-language: ``name:kind`` with an optional ``?`` suffix when nulls are legal.
Kinds: int, Int (nullable integer), float, str, bool, date, datetime.

The two forecast tables that are empty in this dataset (fact_demand_forecast, model_run) are
not part of the semantic model, so their column types are an assumption (marked below).
"""

from __future__ import annotations

from dataclasses import dataclass

_RAW: dict[str, tuple[str, ...]] = {
    "dim_date": (
        "date_key:int", "full_date:date", "year:int", "quarter:int", "month:int",
        "month_name:str", "day_of_month:int", "day_of_week:int", "day_name:str",
        "iso_week:int", "is_weekend:int", "is_ramadan:int", "ramadan_day:Int?",
        "is_eid_al_fitr:int", "is_eid_al_adha:int", "is_pre_eid_al_fitr:int",
        "is_pre_eid_al_adha:int", "year_month:int", "year_month_label:str",
        "year_quarter_label:str", "days_in_month:int", "is_complete_month:int",
        "prior_year_date_key:int", "day_seq:int", "is_last_loaded_date:int",
    ),
    "dim_store": (
        "store_key:int", "store_id:int", "store_code:str", "store_name:str", "wilaya:str",
        "city:str", "store_format:str", "floor_area_sqm:int", "opened_date:date",
    ),
    "dim_product": (
        "product_key:int", "product_id:int", "sku:str", "product_name:str",
        "category_name:str", "is_perishable:bool", "brand:str", "pack_size:int",
        "current_list_price:float", "current_unit_cost:float",
    ),
    "dim_customer": (
        "customer_key:int", "customer_id:int", "wilaya:str", "household_type:str",
        "signup_date:date",
    ),
    "dim_promotion": (
        "promotion_key:int", "promotion_name:str", "promotion_type:str", "discount_pct:int",
        "start_date:date", "end_date:date",
    ),
    "fact_sales": (
        "sales_line_id:int", "transaction_id:int", "line_no:int", "date_key:int",
        "store_key:int", "product_key:int", "customer_key:Int?", "promotion_key:Int?",
        "quantity:int", "unit_price:float", "unit_cost:float", "gross_sales:float",
        "discount_amount:float", "net_sales:float", "cogs:float", "gross_profit:float",
    ),
    "fact_inventory_daily": (
        "store_key:int", "product_key:int", "date_key:int", "opening_stock:int",
        "received_qty:int", "units_sold:int", "closing_stock:int", "on_order_qty:int",
        "stockout_flag:bool",
    ),
    "fact_promotions": (
        "store_key:int", "product_key:int", "date_key:int", "promotion_key:int",
        "discount_pct:int",
    ),
    "v_inventory_risk": (
        "store_key:int", "store_code:str", "store_name:str", "wilaya:str", "product_key:int",
        "sku:str", "product_name:str", "category_name:str", "is_perishable:int",
        "snapshot_date_key:int", "snapshot_date:date", "closing_stock:int",
        "on_order_qty:int", "units_sold_14d:int", "days_observed_14d:int",
        "avg_daily_units_14d:float?", "days_of_cover:float?", "days_of_cover_incl_on_order:float?",
        "stockout_days_14d:int", "stock_value_at_current_cost:float", "risk_tier:str",
        "risk_tier_order:int",
    ),
    "backtest_run": (
        "backtest_run_id:str", "created_at:datetime", "profile:str", "horizon_days:int",
        "n_origins:int", "origin_step_days:int", "min_train_days:int", "n_series:int",
        "censor_policy:str", "seed:int", "models:str", "settings_json:str", "notes:str?",
    ),
    "forecast_backtest_metric": (
        "backtest_run_id:str", "model_name:str", "evaluation_basis:str", "level:str",
        "origin_date:date", "horizon_bucket:int", "n_obs:int", "sum_actual:float",
        "sum_pred:float", "sum_abs_error:float", "sum_sq_error:float",
    ),
    "v_forecast_accuracy": (
        "backtest_run_id:str", "model_name:str", "evaluation_basis:str", "level:str",
        "n_origins:int", "n_obs:int", "sum_actual:float", "sum_pred:float",
        "sum_abs_error:float", "sum_sq_error:float", "mae:float?", "mse:float?",
        "wape:float?", "bias:float?", "skill_mae_vs_seasonal_naive:float?",
    ),
    "v_forecast_accuracy_by_horizon": (
        "backtest_run_id:str", "model_name:str", "evaluation_basis:str", "level:str",
        "horizon_bucket:int", "n_origins:int", "n_obs:int", "sum_actual:float",
        "sum_pred:float", "sum_abs_error:float", "sum_sq_error:float", "mae:float?",
        "mse:float?", "wape:float?", "bias:float?", "skill_mae_vs_seasonal_naive:float?",
    ),
    # ASSUMPTION: empty in this dataset and absent from the semantic model; types are a guess.
    "fact_demand_forecast": (
        "forecast_run_id:str?", "store_key:Int?", "product_key:Int?", "date_key:Int?",
        "horizon_step:Int?", "predicted_units:float?", "lower_bound:float?",
        "upper_bound:float?",
    ),
    "model_run": (
        "forecast_run_id:str?", "model_name:str?", "model_version:str?", "trained_at:datetime?",
        "train_end_date:date?", "horizon_days:Int?", "feature_definition:str?", "notes:str?",
    ),
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    nullable: bool


def _parse(spec: str) -> ColumnSpec:
    name, kind = spec.split(":")
    nullable = kind.endswith("?")
    kind = kind.rstrip("?")
    if kind not in {"int", "Int", "float", "str", "bool", "date", "datetime"}:
        raise ValueError(f"unknown kind in schema: {spec}")
    return ColumnSpec(name, kind, nullable)


SCHEMAS: dict[str, tuple[ColumnSpec, ...]] = {
    table: tuple(_parse(s) for s in specs) for table, specs in _RAW.items()
}

# Tables the application loads (the semantic model's data tables plus the two empty
# forecast tables, which are loaded only so their row counts can be reported).
APP_TABLES: tuple[str, ...] = tuple(SCHEMAS)

# Every table listed in bi_export_manifest.json. The six below are exported but are not
# loaded by the semantic model; they are only counted here and used by tests later.
EXPORT_ONLY_TABLES: tuple[str, ...] = (
    "v_kpi_network_daily",
    "v_kpi_network_monthly",
    "v_kpi_category_monthly",
    "v_kpi_product",
    "v_kpi_store",
    "v_kpi_wilaya",
    "v_stock_availability_daily",
)
