"""Fixed values for the smoke_m3_20261025 dataset, taken from the manifest and from the
audit's recomputation. Only data-level facts live here (Milestone A); metric oracles are
added in Milestone B."""

MANIFEST_ROWS = {
    "dim_date": 90, "dim_store": 5, "dim_product": 100, "dim_customer": 600,
    "dim_promotion": 4, "fact_sales": 48216, "fact_inventory_daily": 37710,
    "fact_promotions": 2086, "v_kpi_network_daily": 90, "v_kpi_network_monthly": 4,
    "v_kpi_category_monthly": 56, "v_kpi_product": 100, "v_kpi_store": 5, "v_kpi_wilaya": 5,
    "v_stock_availability_daily": 450, "v_inventory_risk": 419, "model_run": 0,
    "fact_demand_forecast": 0, "backtest_run": 1, "forecast_backtest_metric": 720,
    "v_forecast_accuracy": 18, "v_forecast_accuracy_by_horizon": 180,
}

DATE_MIN, DATE_MAX, N_DAYS = "2025-02-01", "2025-05-01", 90
NULLS = {("fact_sales", "customer_key"): 31262, ("fact_sales", "promotion_key"): 45288,
         ("dim_date", "ramadan_day"): 61}
PROMO_LINES = 2928
TRACKED_PAIRS = 419
INVENTORY_PAIRS_POSSIBLE = 500          # 5 stores x 100 products; only 419 are carried
STOCKOUT_SKU_DAYS = 794
NEGATIVE_MARGIN_LINES = 1128
TRANSACTIONS = 5776
# Fingerprint of the facts (raw column sums, not business measures)
SUM_NET_SALES = 40_389_951.00
SUM_QUANTITY = 165_893
SNAPSHOT_DATE_KEY = 20250501
TIER_COUNTS = {"Stockout": 4, "Critical": 83, "Low": 193, "Healthy": 139}

MODELS_IN_ORDER = ["naive", "seasonal_naive", "moving_average", "ses_weekday", "croston_sba", "gbm"]
MODEL_LABELS = {
    "naive": "Naive", "seasonal_naive": "Seasonal naive (benchmark)",
    "moving_average": "Moving average", "ses_weekday": "Exp. smoothing + weekday",
    "croston_sba": "Croston-SBA", "gbm": "Gradient boosting",
}
ORIGINS = ["2025-03-27", "2025-04-03", "2025-04-10", "2025-04-17"]
