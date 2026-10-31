-- Milestone 2 post-load validation of the analytical views (sql/schema/004_analytics_views.sql).
-- Each statement returns ONE row: (violations). A check passes when violations = 0.
-- Metadata comment format:  -- check: <name> | <severity: error|warning> | <description>
-- Money and unit totals must reconcile EXACTLY (within 0.01 DZD) between fact tables and views.
-- Portable across PostgreSQL and SQLite.

-- check: an_daily_rows_equal_dim_date | error | v_kpi_network_daily must have exactly one row per dim_date row (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(*) FROM dwh.v_kpi_network_daily) = (SELECT COUNT(*) FROM dwh.dim_date)
            THEN 0 ELSE 1 END AS violations;

-- check: an_daily_net_sales_reconciles | error | sum of v_kpi_network_daily.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_network_daily)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_daily_units_reconcile | error | sum of v_kpi_network_daily.units_sold differs from fact_sales quantity (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(units_sold), 0) FROM dwh.v_kpi_network_daily)
               = (SELECT COALESCE(SUM(quantity), 0) FROM dwh.fact_sales)
            THEN 0 ELSE 1 END AS violations;

-- check: an_daily_profit_identity | error | days where net_sales - cogs differs from gross_profit
SELECT COUNT(*) AS violations FROM dwh.v_kpi_network_daily
WHERE ABS(net_sales - cogs - gross_profit) >= 0.01;

-- check: an_monthly_net_sales_reconciles | error | sum of v_kpi_network_monthly.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_network_monthly)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_category_monthly_net_sales_reconciles | error | sum of v_kpi_category_monthly.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_category_monthly)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_category_sku_days_reconcile | error | sum of v_kpi_category_monthly.sku_days differs from fact_inventory_daily rows (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(sku_days), 0) FROM dwh.v_kpi_category_monthly)
               = (SELECT COUNT(*) FROM dwh.fact_inventory_daily)
            THEN 0 ELSE 1 END AS violations;

-- check: an_product_rows_equal_dim_product | error | v_kpi_product must have exactly one row per product (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(*) FROM dwh.v_kpi_product) = (SELECT COUNT(*) FROM dwh.dim_product)
            THEN 0 ELSE 1 END AS violations;

-- check: an_product_net_sales_reconciles | error | sum of v_kpi_product.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_product)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_product_stockout_days_reconcile | error | sum of v_kpi_product.stockout_sku_days differs from flagged inventory rows (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(stockout_sku_days), 0) FROM dwh.v_kpi_product)
               = (SELECT COUNT(*) FROM dwh.fact_inventory_daily WHERE stockout_flag)
            THEN 0 ELSE 1 END AS violations;

-- check: an_product_rank_unique | error | net_sales_rank must be 1..N without duplicates or gaps (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(DISTINCT net_sales_rank) FROM dwh.v_kpi_product)
               = (SELECT COUNT(*) FROM dwh.v_kpi_product)
             AND (SELECT COALESCE(MAX(net_sales_rank), 0) FROM dwh.v_kpi_product)
               = (SELECT COUNT(*) FROM dwh.v_kpi_product)
            THEN 0 ELSE 1 END AS violations;

-- check: an_store_net_sales_reconciles | error | sum of v_kpi_store.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_store)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_store_transactions_reconcile | error | sum of v_kpi_store.transactions differs from distinct transactions in fact_sales (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(transactions), 0) FROM dwh.v_kpi_store)
               = (SELECT COUNT(DISTINCT transaction_id) FROM dwh.fact_sales)
            THEN 0 ELSE 1 END AS violations;

-- check: an_wilaya_net_sales_reconciles | error | sum of v_kpi_wilaya.net_sales differs from fact_sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_kpi_wilaya)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: an_wilaya_store_count_reconciles | error | sum of v_kpi_wilaya.store_count differs from dim_store rows (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(store_count), 0) FROM dwh.v_kpi_wilaya)
               = (SELECT COUNT(*) FROM dwh.dim_store)
            THEN 0 ELSE 1 END AS violations;

-- check: an_availability_sku_days_reconcile | error | sum of v_stock_availability_daily.sku_count differs from fact_inventory_daily rows (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(sku_count), 0) FROM dwh.v_stock_availability_daily)
               = (SELECT COUNT(*) FROM dwh.fact_inventory_daily)
            THEN 0 ELSE 1 END AS violations;

-- check: an_availability_stockouts_reconcile | error | sum of v_stock_availability_daily.stockout_sku_count differs from flagged inventory rows (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(stockout_sku_count), 0) FROM dwh.v_stock_availability_daily)
               = (SELECT COUNT(*) FROM dwh.fact_inventory_daily WHERE stockout_flag)
            THEN 0 ELSE 1 END AS violations;

-- check: an_availability_rate_bounds | error | availability and stockout rates must be within 0..1 and add up to 1
SELECT COUNT(*) AS violations FROM dwh.v_stock_availability_daily
WHERE availability_rate < 0 OR availability_rate > 1
   OR ABS(availability_rate + stockout_rate - 1) > 0.000001;

-- check: an_inventory_risk_one_row_per_sku | error | v_inventory_risk must have one row per inventory row on the latest date (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(*) FROM dwh.v_inventory_risk)
               = (SELECT COUNT(*) FROM dwh.fact_inventory_daily
                  WHERE date_key = (SELECT MAX(date_key) FROM dwh.fact_inventory_daily))
            THEN 0 ELSE 1 END AS violations;

-- check: an_inventory_risk_tier_assigned | error | v_inventory_risk rows without a risk tier
SELECT COUNT(*) AS violations FROM dwh.v_inventory_risk WHERE risk_tier IS NULL;

-- check: an_dim_date_key_matches_date | error | dim_date rows whose date_key is not yyyymmdd of year/month/day
SELECT COUNT(*) AS violations FROM dwh.dim_date
WHERE date_key <> year * 10000 + month * 100 + day_of_month;

-- check: an_dim_date_contiguous | error | dim_date days that do not directly follow the previous day (Power BI time intelligence needs a gap-free date table)
SELECT COUNT(*) AS violations FROM (
    SELECT d.year, d.month, d.day_of_month,
           LAG(d.year)         OVER (ORDER BY d.date_key) AS prev_year,
           LAG(d.month)        OVER (ORDER BY d.date_key) AS prev_month,
           LAG(d.day_of_month) OVER (ORDER BY d.date_key) AS prev_day
    FROM dwh.dim_date d
) t
WHERE t.prev_year IS NOT NULL
  AND NOT (
        (t.year = t.prev_year AND t.month = t.prev_month AND t.day_of_month = t.prev_day + 1)
     OR (t.day_of_month = 1
         AND t.prev_day = CASE WHEN t.prev_month IN (1, 3, 5, 7, 8, 10, 12) THEN 31
                               WHEN t.prev_month IN (4, 6, 9, 11) THEN 30
                               WHEN (t.prev_year % 4 = 0 AND t.prev_year % 100 <> 0) OR t.prev_year % 400 = 0 THEN 29
                               ELSE 28 END
         AND ((t.year = t.prev_year AND t.month = t.prev_month + 1)
              OR (t.year = t.prev_year + 1 AND t.month = 1 AND t.prev_month = 12)))
  );

-- check: an_dim_date_bi_rows_equal_dim_date | error | v_dim_date_bi must have exactly one row per dim_date row (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(*) FROM dwh.v_dim_date_bi) = (SELECT COUNT(*) FROM dwh.dim_date)
            THEN 0 ELSE 1 END AS violations;
