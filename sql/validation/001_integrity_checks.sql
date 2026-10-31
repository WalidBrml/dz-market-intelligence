-- Post-load validation. Each statement returns ONE row: (violations).
-- Metadata comment format:  -- check: <name> | <severity: error|warning> | <description>
-- A check passes when violations = 0. Portable across PostgreSQL and SQLite.

-- check: orphan_sales_store | error | fact_sales rows without a matching dim_store
SELECT COUNT(*) AS violations FROM dwh.fact_sales f
LEFT JOIN dwh.dim_store d ON d.store_key = f.store_key WHERE d.store_key IS NULL;

-- check: orphan_sales_product | error | fact_sales rows without a matching dim_product
SELECT COUNT(*) AS violations FROM dwh.fact_sales f
LEFT JOIN dwh.dim_product d ON d.product_key = f.product_key WHERE d.product_key IS NULL;

-- check: orphan_sales_date | error | fact_sales rows without a matching dim_date
SELECT COUNT(*) AS violations FROM dwh.fact_sales f
LEFT JOIN dwh.dim_date d ON d.date_key = f.date_key WHERE d.date_key IS NULL;

-- check: orphan_inventory_keys | error | fact_inventory_daily rows with a missing dimension member
SELECT COUNT(*) AS violations FROM dwh.fact_inventory_daily f
LEFT JOIN dwh.dim_store s ON s.store_key = f.store_key
LEFT JOIN dwh.dim_product p ON p.product_key = f.product_key
LEFT JOIN dwh.dim_date d ON d.date_key = f.date_key
WHERE s.store_key IS NULL OR p.product_key IS NULL OR d.date_key IS NULL;

-- check: sales_amount_arithmetic | error | gross/net/cogs/profit columns inconsistent with quantity x price/cost
SELECT COUNT(*) AS violations FROM dwh.fact_sales
WHERE ABS(gross_sales - quantity * unit_price) >= 0.005
   OR ABS(net_sales - (gross_sales - discount_amount)) >= 0.005
   OR ABS(cogs - quantity * unit_cost) >= 0.005
   OR ABS(gross_profit - (net_sales - cogs)) >= 0.005;

-- check: discount_without_promotion | error | discounted lines that carry no promotion
SELECT COUNT(*) AS violations FROM dwh.fact_sales
WHERE discount_amount > 0 AND promotion_key IS NULL;

-- check: promotion_line_without_promo_fact | error | lines with a promotion but no fact_promotions row for that store/product/day
SELECT COUNT(*) AS violations FROM dwh.fact_sales f
LEFT JOIN dwh.fact_promotions p
       ON p.store_key = f.store_key AND p.product_key = f.product_key AND p.date_key = f.date_key
WHERE f.promotion_key IS NOT NULL AND p.promotion_key IS NULL;

-- check: duplicate_product_in_basket | error | the same product appearing twice in one transaction
SELECT COUNT(*) AS violations FROM (
    SELECT transaction_id, product_key FROM dwh.fact_sales
    GROUP BY transaction_id, product_key HAVING COUNT(*) > 1
) t;

-- check: transaction_without_lines | error | ops transactions with zero lines
SELECT COUNT(*) AS violations FROM ops.sales_transaction t
LEFT JOIN ops.sales_transaction_line l ON l.transaction_id = t.transaction_id
WHERE l.transaction_id IS NULL;

-- check: ops_vs_dwh_line_count | error | ops line count differs from fact_sales row count (1 = mismatch)
SELECT CASE WHEN (SELECT COUNT(*) FROM ops.sales_transaction_line) = (SELECT COUNT(*) FROM dwh.fact_sales)
            THEN 0 ELSE 1 END AS violations;

-- check: ops_vs_dwh_units | error | ops units differ from fact_sales units (1 = mismatch)
SELECT CASE WHEN (SELECT COALESCE(SUM(quantity), 0) FROM ops.sales_transaction_line)
               = (SELECT COALESCE(SUM(quantity), 0) FROM dwh.fact_sales)
            THEN 0 ELSE 1 END AS violations;

-- check: stock_balance | error | closing <> opening + received - sold
SELECT COUNT(*) AS violations FROM dwh.fact_inventory_daily
WHERE closing_stock <> opening_stock + received_qty - units_sold;

-- check: stock_continuity | error | opening stock differs from previous day's closing stock
SELECT COUNT(*) AS violations FROM (
    SELECT opening_stock,
           LAG(closing_stock) OVER (PARTITION BY store_key, product_key ORDER BY date_key) AS prev_close
    FROM dwh.fact_inventory_daily
) t WHERE prev_close IS NOT NULL AND opening_stock <> prev_close;

-- check: inventory_day_completeness | error | store/product series whose day count differs from dim_date
SELECT COUNT(*) AS violations FROM (
    SELECT store_key, product_key, COUNT(*) AS n_days FROM dwh.fact_inventory_daily
    GROUP BY store_key, product_key
) t WHERE n_days <> (SELECT COUNT(*) FROM dwh.dim_date);

-- check: assortment_without_inventory | error | assortment pairs that have no inventory rows
SELECT COUNT(*) AS violations FROM ops.store_assortment a
LEFT JOIN (SELECT DISTINCT store_key, product_key FROM dwh.fact_inventory_daily) i
       ON i.store_key = a.store_id AND i.product_key = a.product_id
WHERE i.store_key IS NULL;

-- check: sales_vs_inventory_units | error | store/product/day where fact_sales units differ from inventory units sold
SELECT COUNT(*) AS violations FROM dwh.v_store_product_day
WHERE sales_units <> inventory_units_sold;

-- check: sales_without_inventory_row | error | sales aggregates with no inventory row at the same grain
SELECT COUNT(*) AS violations FROM dwh.v_sales_daily d
LEFT JOIN dwh.fact_inventory_daily i
       ON i.store_key = d.store_key AND i.product_key = d.product_key AND i.date_key = d.date_key
WHERE i.store_key IS NULL;

-- check: double_count_guard | error | v_store_product_day net sales differs from fact_sales net sales (1 = mismatch)
SELECT CASE WHEN ABS((SELECT COALESCE(SUM(net_sales), 0) FROM dwh.v_store_product_day)
                   - (SELECT COALESCE(SUM(net_sales), 0) FROM dwh.fact_sales)) < 0.01
            THEN 0 ELSE 1 END AS violations;

-- check: latent_below_observed | error | latent demand lower than observed sales
SELECT COUNT(*) AS violations FROM ops.sim_latent_demand l
JOIN dwh.dim_date dd ON dd.full_date = l.demand_date
JOIN dwh.fact_inventory_daily i
  ON i.store_key = l.store_id AND i.product_key = l.product_id AND i.date_key = dd.date_key
WHERE l.latent_demand_units < i.units_sold
   OR l.unmet_demand_units <> l.latent_demand_units - i.units_sold;

-- check: unmet_demand_without_stockout | warning | unmet demand on a day with stock left over (should be 0)
SELECT COUNT(*) AS violations FROM ops.sim_latent_demand l
JOIN dwh.dim_date dd ON dd.full_date = l.demand_date
JOIN dwh.fact_inventory_daily i
  ON i.store_key = l.store_id AND i.product_key = l.product_id AND i.date_key = dd.date_key
WHERE l.unmet_demand_units > 0 AND i.closing_stock > 0;

-- check: promotion_outside_window | error | fact_promotions days outside the promotion start/end dates
SELECT COUNT(*) AS violations FROM dwh.fact_promotions fp
JOIN dwh.dim_promotion p ON p.promotion_key = fp.promotion_key
JOIN dwh.dim_date d ON d.date_key = fp.date_key
WHERE d.full_date < p.start_date OR d.full_date > p.end_date;
