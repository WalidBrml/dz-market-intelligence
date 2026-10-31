-- =============================================================================
-- Milestone 2 - analytical views (all in schema dwh)
-- ALL DATA IS SYNTHETIC. Run order: after 001 (tables), 002 (indexes), 003 (base views).
--
-- Design rules (documented in docs/analytical_views.md):
--  1. Every view states its GRAIN (one row per ...). Money is DZD.
--  2. Every view is built directly on the base tables and does NOT depend on other views,
--     so this file can be re-applied without PostgreSQL dependency errors.
--  3. No double counting: each fact is aggregated on its own to the view grain BEFORE the
--     results are joined. fact_sales is never joined row-to-row to fact_inventory_daily
--     or fact_promotions. Promotion information comes from fact_sales.promotion_key.
--  4. Ratios are fractions (0.25 = 25 percent) and are NULL when the denominator is 0.
--     Every division is written 1.0 * a / NULLIF(b, 0): the 1.0 forces decimal division
--     on SQLite (integer / integer truncates there) and is harmless on PostgreSQL.
--  5. Portable SQL only: CASE instead of boolean arithmetic, no FULL OUTER JOIN, no
--     CREATE OR REPLACE, no dialect functions. Window functions are used (SQLite 3.25+).
--  6. Flags are exported as 0/1 integers so both backends return the same types.
--  7. Year-over-year (YoY): the prior-year row is found with date_key - 10000 (daily) or
--     year_month - 100 (monthly), because date_key is yyyymmdd. If the prior-year row does
--     not exist (outside the loaded calendar, or 29 February) the YoY columns are NULL.
--
-- Comments in this file must not contain a semicolon (the statement splitter cuts on it).
-- =============================================================================

DROP VIEW IF EXISTS dwh.v_dim_date_bi;
DROP VIEW IF EXISTS dwh.v_inventory_risk;
DROP VIEW IF EXISTS dwh.v_stock_availability_daily;
DROP VIEW IF EXISTS dwh.v_kpi_wilaya;
DROP VIEW IF EXISTS dwh.v_kpi_store;
DROP VIEW IF EXISTS dwh.v_kpi_product;
DROP VIEW IF EXISTS dwh.v_kpi_category_monthly;
DROP VIEW IF EXISTS dwh.v_kpi_network_monthly;
DROP VIEW IF EXISTS dwh.v_kpi_network_daily;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_network_daily
-- GRAIN: one row per calendar date in dwh.dim_date (days without sales appear with zeros).
-- Whole network (all stores, all products). Sales only: no inventory join.
-- transactions is a distinct count of baskets. A basket belongs to one store and one date,
-- so it can be summed across stores for the same date but NOT across products.
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_network_daily AS
WITH sales AS (
    SELECT s.date_key,
           SUM(s.quantity)                  AS units_sold,
           SUM(s.gross_sales)               AS gross_sales,
           SUM(s.discount_amount)           AS discount_amount,
           SUM(s.net_sales)                 AS net_sales,
           SUM(s.cogs)                      AS cogs,
           SUM(s.gross_profit)              AS gross_profit,
           COUNT(DISTINCT s.transaction_id) AS transactions
    FROM dwh.fact_sales s
    GROUP BY s.date_key
), base AS (
    SELECT d.date_key,
           d.full_date,
           d.year                                   AS calendar_year,
           d.quarter                                AS calendar_quarter,
           d.month                                  AS calendar_month,
           d.year * 100 + d.month                   AS year_month,
           d.month_name,
           d.day_of_month,
           d.day_of_week,
           d.day_name,
           d.iso_week,
           CASE WHEN d.is_weekend     THEN 1 ELSE 0 END AS is_weekend,
           CASE WHEN d.is_ramadan     THEN 1 ELSE 0 END AS is_ramadan,
           d.ramadan_day,
           CASE WHEN d.is_eid_al_fitr THEN 1 ELSE 0 END AS is_eid_al_fitr,
           CASE WHEN d.is_eid_al_adha THEN 1 ELSE 0 END AS is_eid_al_adha,
           COALESCE(s.units_sold, 0)      AS units_sold,
           COALESCE(s.gross_sales, 0)     AS gross_sales,
           COALESCE(s.discount_amount, 0) AS discount_amount,
           COALESCE(s.net_sales, 0)       AS net_sales,
           COALESCE(s.cogs, 0)            AS cogs,
           COALESCE(s.gross_profit, 0)    AS gross_profit,
           COALESCE(s.transactions, 0)    AS transactions
    FROM dwh.dim_date d
    LEFT JOIN sales s ON s.date_key = d.date_key
)
SELECT cur.date_key,
       cur.full_date,
       cur.calendar_year,
       cur.calendar_quarter,
       cur.calendar_month,
       cur.year_month,
       cur.month_name,
       cur.day_of_month,
       cur.day_of_week,
       cur.day_name,
       cur.iso_week,
       cur.is_weekend,
       cur.is_ramadan,
       cur.ramadan_day,
       cur.is_eid_al_fitr,
       cur.is_eid_al_adha,
       cur.units_sold,
       cur.gross_sales,
       cur.discount_amount,
       cur.net_sales,
       cur.cogs,
       cur.gross_profit,
       cur.transactions,
       1.0 * cur.gross_profit    / NULLIF(cur.net_sales, 0)    AS gross_margin,
       1.0 * cur.net_sales       / NULLIF(cur.units_sold, 0)   AS avg_selling_price,
       1.0 * cur.discount_amount / NULLIF(cur.gross_sales, 0)  AS discount_rate,
       1.0 * cur.net_sales       / NULLIF(cur.transactions, 0) AS avg_basket_value,
       cur.date_key - 10000                                    AS prior_year_date_key,
       CASE WHEN py.date_key IS NULL THEN 0 ELSE 1 END         AS has_prior_year_row,
       py.net_sales    AS net_sales_py,
       py.units_sold   AS units_sold_py,
       py.gross_profit AS gross_profit_py,
       1.0 * (cur.net_sales    - py.net_sales)    / NULLIF(py.net_sales, 0)    AS yoy_net_sales_growth,
       1.0 * (cur.units_sold   - py.units_sold)   / NULLIF(py.units_sold, 0)   AS yoy_units_growth,
       1.0 * (cur.gross_profit - py.gross_profit) / NULLIF(py.gross_profit, 0) AS yoy_gross_profit_growth
FROM base cur
LEFT JOIN base py ON py.date_key = cur.date_key - 10000;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_network_monthly
-- GRAIN: one row per calendar month present in dwh.dim_date. Whole network.
-- days_in_data = calendar days of that month present in dim_date. is_complete_month = 1 when
-- every day of the month is present. YoY growth is only reported when BOTH the current and the
-- prior-year month are complete, so a partial month is never compared with a full month.
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_network_monthly AS
WITH months AS (
    SELECT d.year  AS calendar_year,
           d.month AS calendar_month,
           COUNT(*) AS days_in_data,
           MIN(d.full_date) AS first_date,
           MAX(d.full_date) AS last_date
    FROM dwh.dim_date d
    GROUP BY d.year, d.month
), sales AS (
    SELECT d.year  AS calendar_year,
           d.month AS calendar_month,
           SUM(s.quantity)                  AS units_sold,
           SUM(s.gross_sales)               AS gross_sales,
           SUM(s.discount_amount)           AS discount_amount,
           SUM(s.net_sales)                 AS net_sales,
           SUM(s.cogs)                      AS cogs,
           SUM(s.gross_profit)              AS gross_profit,
           COUNT(DISTINCT s.transaction_id) AS transactions
    FROM dwh.fact_sales s
    JOIN dwh.dim_date d ON d.date_key = s.date_key
    GROUP BY d.year, d.month
), base AS (
    SELECT m.calendar_year,
           m.calendar_month,
           m.calendar_year * 100 + m.calendar_month AS year_month,
           m.first_date,
           m.last_date,
           m.days_in_data,
           CASE WHEN m.calendar_month IN (1, 3, 5, 7, 8, 10, 12) THEN 31
                WHEN m.calendar_month IN (4, 6, 9, 11) THEN 30
                WHEN (m.calendar_year % 4 = 0 AND m.calendar_year % 100 <> 0) OR m.calendar_year % 400 = 0 THEN 29
                ELSE 28 END AS days_in_month,
           COALESCE(s.units_sold, 0)      AS units_sold,
           COALESCE(s.gross_sales, 0)     AS gross_sales,
           COALESCE(s.discount_amount, 0) AS discount_amount,
           COALESCE(s.net_sales, 0)       AS net_sales,
           COALESCE(s.cogs, 0)            AS cogs,
           COALESCE(s.gross_profit, 0)    AS gross_profit,
           COALESCE(s.transactions, 0)    AS transactions
    FROM months m
    LEFT JOIN sales s ON s.calendar_year = m.calendar_year AND s.calendar_month = m.calendar_month
)
SELECT cur.year_month,
       cur.calendar_year,
       cur.calendar_month,
       cur.first_date,
       cur.last_date,
       cur.days_in_data,
       cur.days_in_month,
       CASE WHEN cur.days_in_data = cur.days_in_month THEN 1 ELSE 0 END AS is_complete_month,
       cur.units_sold,
       cur.gross_sales,
       cur.discount_amount,
       cur.net_sales,
       cur.cogs,
       cur.gross_profit,
       cur.transactions,
       1.0 * cur.gross_profit    / NULLIF(cur.net_sales, 0)    AS gross_margin,
       1.0 * cur.net_sales       / NULLIF(cur.units_sold, 0)   AS avg_selling_price,
       1.0 * cur.discount_amount / NULLIF(cur.gross_sales, 0)  AS discount_rate,
       1.0 * cur.net_sales       / NULLIF(cur.transactions, 0) AS avg_basket_value,
       py.net_sales    AS net_sales_py,
       py.units_sold   AS units_sold_py,
       py.gross_profit AS gross_profit_py,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1 ELSE 0 END AS has_prior_year_comparison,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1.0 * (cur.net_sales - py.net_sales) / NULLIF(py.net_sales, 0) END AS yoy_net_sales_growth,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1.0 * (cur.units_sold - py.units_sold) / NULLIF(py.units_sold, 0) END AS yoy_units_growth,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1.0 * (cur.gross_profit - py.gross_profit) / NULLIF(py.gross_profit, 0) END AS yoy_gross_profit_growth
FROM base cur
LEFT JOIN base py ON py.year_month = cur.year_month - 100;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_category_monthly
-- GRAIN: one row per product category x calendar month (every category x every month in
-- dim_date, zeros when nothing sold). Includes availability (SKU-days and stockout SKU-days).
-- Sales and inventory are aggregated separately to (category, month) and then joined 1:1:1.
-- Stockout rate = stockout_sku_days / sku_days. Sum both columns to re-aggregate correctly.
-- YoY follows the monthly rule (both months must be complete).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_category_monthly AS
WITH months AS (
    SELECT d.year  AS calendar_year,
           d.month AS calendar_month,
           COUNT(*) AS days_in_data
    FROM dwh.dim_date d
    GROUP BY d.year, d.month
), categories AS (
    SELECT p.category_name,
           MAX(CASE WHEN p.is_perishable THEN 1 ELSE 0 END) AS is_perishable
    FROM dwh.dim_product p
    GROUP BY p.category_name
), sales AS (
    SELECT p.category_name,
           d.year  AS calendar_year,
           d.month AS calendar_month,
           SUM(s.quantity)        AS units_sold,
           SUM(s.gross_sales)     AS gross_sales,
           SUM(s.discount_amount) AS discount_amount,
           SUM(s.net_sales)       AS net_sales,
           SUM(s.cogs)            AS cogs,
           SUM(s.gross_profit)    AS gross_profit
    FROM dwh.fact_sales s
    JOIN dwh.dim_product p ON p.product_key = s.product_key
    JOIN dwh.dim_date d    ON d.date_key = s.date_key
    GROUP BY p.category_name, d.year, d.month
), inv AS (
    SELECT p.category_name,
           d.year  AS calendar_year,
           d.month AS calendar_month,
           COUNT(*) AS sku_days,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_sku_days
    FROM dwh.fact_inventory_daily i
    JOIN dwh.dim_product p ON p.product_key = i.product_key
    JOIN dwh.dim_date d    ON d.date_key = i.date_key
    GROUP BY p.category_name, d.year, d.month
), base AS (
    SELECT c.category_name,
           c.is_perishable,
           m.calendar_year,
           m.calendar_month,
           m.calendar_year * 100 + m.calendar_month AS year_month,
           m.days_in_data,
           CASE WHEN m.calendar_month IN (1, 3, 5, 7, 8, 10, 12) THEN 31
                WHEN m.calendar_month IN (4, 6, 9, 11) THEN 30
                WHEN (m.calendar_year % 4 = 0 AND m.calendar_year % 100 <> 0) OR m.calendar_year % 400 = 0 THEN 29
                ELSE 28 END AS days_in_month,
           COALESCE(s.units_sold, 0)      AS units_sold,
           COALESCE(s.gross_sales, 0)     AS gross_sales,
           COALESCE(s.discount_amount, 0) AS discount_amount,
           COALESCE(s.net_sales, 0)       AS net_sales,
           COALESCE(s.cogs, 0)            AS cogs,
           COALESCE(s.gross_profit, 0)    AS gross_profit,
           COALESCE(i.sku_days, 0)          AS sku_days,
           COALESCE(i.stockout_sku_days, 0) AS stockout_sku_days
    FROM categories c
    CROSS JOIN months m
    LEFT JOIN sales s ON s.category_name = c.category_name
                     AND s.calendar_year = m.calendar_year AND s.calendar_month = m.calendar_month
    LEFT JOIN inv i   ON i.category_name = c.category_name
                     AND i.calendar_year = m.calendar_year AND i.calendar_month = m.calendar_month
)
SELECT cur.category_name,
       cur.is_perishable,
       cur.year_month,
       cur.calendar_year,
       cur.calendar_month,
       cur.days_in_data,
       cur.days_in_month,
       CASE WHEN cur.days_in_data = cur.days_in_month THEN 1 ELSE 0 END AS is_complete_month,
       cur.units_sold,
       cur.gross_sales,
       cur.discount_amount,
       cur.net_sales,
       cur.cogs,
       cur.gross_profit,
       1.0 * cur.gross_profit    / NULLIF(cur.net_sales, 0)   AS gross_margin,
       1.0 * cur.net_sales       / NULLIF(cur.units_sold, 0)  AS avg_selling_price,
       1.0 * cur.discount_amount / NULLIF(cur.gross_sales, 0) AS discount_rate,
       cur.sku_days,
       cur.stockout_sku_days,
       1.0 * cur.stockout_sku_days / NULLIF(cur.sku_days, 0)  AS stockout_rate,
       py.net_sales    AS net_sales_py,
       py.units_sold   AS units_sold_py,
       py.gross_profit AS gross_profit_py,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1 ELSE 0 END AS has_prior_year_comparison,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1.0 * (cur.net_sales - py.net_sales) / NULLIF(py.net_sales, 0) END AS yoy_net_sales_growth,
       CASE WHEN cur.days_in_data = cur.days_in_month AND py.days_in_data = py.days_in_month
            THEN 1.0 * (cur.units_sold - py.units_sold) / NULLIF(py.units_sold, 0) END AS yoy_units_growth
FROM base cur
LEFT JOIN base py ON py.category_name = cur.category_name AND py.year_month = cur.year_month - 100;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_product
-- GRAIN: one row per product (every product in dim_product, zeros when it never sold),
-- over the WHOLE loaded period and all stores. Promotion split uses fact_sales.promotion_key
-- (a sales line either carries a promotion or not), so promotions never multiply rows.
-- Availability: sku_days = store-days the product was assorted, stockout_sku_days = of those,
-- days with closing stock 0. stockout_rate = stockout_sku_days / sku_days.
-- net_sales_rank is 1 for the best seller (ties broken by product_key).
-- abc_class: A = products making up the first 80 percent of net sales, B = next 15 percent,
-- C = rest (a product is classed by the cumulative share BEFORE its own sales are added).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_product AS
WITH sales AS (
    SELECT s.product_key,
           SUM(s.quantity)        AS units_sold,
           SUM(s.gross_sales)     AS gross_sales,
           SUM(s.discount_amount) AS discount_amount,
           SUM(s.net_sales)       AS net_sales,
           SUM(s.cogs)            AS cogs,
           SUM(s.gross_profit)    AS gross_profit,
           SUM(CASE WHEN s.promotion_key IS NOT NULL THEN s.quantity ELSE 0 END)  AS promo_units_sold,
           SUM(CASE WHEN s.promotion_key IS NOT NULL THEN s.net_sales ELSE 0 END) AS promo_net_sales
    FROM dwh.fact_sales s
    GROUP BY s.product_key
), inv AS (
    SELECT i.product_key,
           COUNT(*) AS sku_days,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_sku_days,
           COUNT(DISTINCT i.store_key) AS stores_carrying
    FROM dwh.fact_inventory_daily i
    GROUP BY i.product_key
), base AS (
    SELECT p.product_key,
           p.product_id,
           p.sku,
           p.product_name,
           p.category_name,
           p.brand,
           CASE WHEN p.is_perishable THEN 1 ELSE 0 END AS is_perishable,
           p.pack_size,
           p.current_list_price,
           p.current_unit_cost,
           COALESCE(s.units_sold, 0)        AS units_sold,
           COALESCE(s.gross_sales, 0)       AS gross_sales,
           COALESCE(s.discount_amount, 0)   AS discount_amount,
           COALESCE(s.net_sales, 0)         AS net_sales,
           COALESCE(s.cogs, 0)              AS cogs,
           COALESCE(s.gross_profit, 0)      AS gross_profit,
           COALESCE(s.promo_units_sold, 0)  AS promo_units_sold,
           COALESCE(s.promo_net_sales, 0)   AS promo_net_sales,
           COALESCE(i.stores_carrying, 0)   AS stores_carrying,
           COALESCE(i.sku_days, 0)          AS sku_days,
           COALESCE(i.stockout_sku_days, 0) AS stockout_sku_days
    FROM dwh.dim_product p
    LEFT JOIN sales s ON s.product_key = p.product_key
    LEFT JOIN inv i   ON i.product_key = p.product_key
), ranked AS (
    SELECT b.*,
           ROW_NUMBER() OVER (ORDER BY b.net_sales DESC, b.product_key)    AS net_sales_rank,
           ROW_NUMBER() OVER (ORDER BY b.gross_profit DESC, b.product_key) AS gross_profit_rank,
           SUM(b.net_sales) OVER (ORDER BY b.net_sales DESC, b.product_key
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_net_sales,
           SUM(b.net_sales) OVER () AS total_net_sales
    FROM base b
)
SELECT r.product_key,
       r.product_id,
       r.sku,
       r.product_name,
       r.category_name,
       r.brand,
       r.is_perishable,
       r.pack_size,
       r.current_list_price,
       r.current_unit_cost,
       r.units_sold,
       r.gross_sales,
       r.discount_amount,
       r.net_sales,
       r.cogs,
       r.gross_profit,
       1.0 * r.gross_profit    / NULLIF(r.net_sales, 0)   AS gross_margin,
       1.0 * r.net_sales       / NULLIF(r.units_sold, 0)  AS avg_selling_price,
       1.0 * r.discount_amount / NULLIF(r.gross_sales, 0) AS discount_rate,
       r.promo_units_sold,
       r.promo_net_sales,
       1.0 * r.promo_units_sold / NULLIF(r.units_sold, 0) AS promo_unit_share,
       r.stores_carrying,
       r.sku_days,
       r.stockout_sku_days,
       1.0 * r.stockout_sku_days / NULLIF(r.sku_days, 0)  AS stockout_rate,
       r.net_sales_rank,
       r.gross_profit_rank,
       1.0 * r.net_sales / NULLIF(r.total_net_sales, 0)            AS net_sales_share,
       1.0 * r.cumulative_net_sales / NULLIF(r.total_net_sales, 0) AS cumulative_net_sales_share,
       CASE WHEN r.total_net_sales = 0 THEN 'C'
            WHEN 1.0 * (r.cumulative_net_sales - r.net_sales) / NULLIF(r.total_net_sales, 0) < 0.80 THEN 'A'
            WHEN 1.0 * (r.cumulative_net_sales - r.net_sales) / NULLIF(r.total_net_sales, 0) < 0.95 THEN 'B'
            ELSE 'C' END AS abc_class
FROM ranked r;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_store
-- GRAIN: one row per store, over the WHOLE loaded period (every store in dim_store).
-- transactions = distinct baskets (a basket belongs to exactly one store).
-- net_sales_per_sqm = net_sales / floor_area_sqm over the whole period (not annualised).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_store AS
WITH sales AS (
    SELECT s.store_key,
           SUM(s.quantity)                  AS units_sold,
           SUM(s.gross_sales)               AS gross_sales,
           SUM(s.discount_amount)           AS discount_amount,
           SUM(s.net_sales)                 AS net_sales,
           SUM(s.cogs)                      AS cogs,
           SUM(s.gross_profit)              AS gross_profit,
           COUNT(DISTINCT s.transaction_id) AS transactions
    FROM dwh.fact_sales s
    GROUP BY s.store_key
), inv AS (
    SELECT i.store_key,
           COUNT(*) AS sku_days,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_sku_days
    FROM dwh.fact_inventory_daily i
    GROUP BY i.store_key
), base AS (
    SELECT st.store_key,
           st.store_id,
           st.store_code,
           st.store_name,
           st.wilaya,
           st.city,
           st.store_format,
           st.floor_area_sqm,
           st.opened_date,
           COALESCE(s.units_sold, 0)        AS units_sold,
           COALESCE(s.gross_sales, 0)       AS gross_sales,
           COALESCE(s.discount_amount, 0)   AS discount_amount,
           COALESCE(s.net_sales, 0)         AS net_sales,
           COALESCE(s.cogs, 0)              AS cogs,
           COALESCE(s.gross_profit, 0)      AS gross_profit,
           COALESCE(s.transactions, 0)      AS transactions,
           COALESCE(i.sku_days, 0)          AS sku_days,
           COALESCE(i.stockout_sku_days, 0) AS stockout_sku_days
    FROM dwh.dim_store st
    LEFT JOIN sales s ON s.store_key = st.store_key
    LEFT JOIN inv i   ON i.store_key = st.store_key
)
SELECT b.store_key,
       b.store_id,
       b.store_code,
       b.store_name,
       b.wilaya,
       b.city,
       b.store_format,
       b.floor_area_sqm,
       b.opened_date,
       b.units_sold,
       b.gross_sales,
       b.discount_amount,
       b.net_sales,
       b.cogs,
       b.gross_profit,
       b.transactions,
       1.0 * b.gross_profit    / NULLIF(b.net_sales, 0)       AS gross_margin,
       1.0 * b.net_sales       / NULLIF(b.units_sold, 0)      AS avg_selling_price,
       1.0 * b.discount_amount / NULLIF(b.gross_sales, 0)     AS discount_rate,
       1.0 * b.net_sales       / NULLIF(b.transactions, 0)    AS avg_basket_value,
       1.0 * b.net_sales       / NULLIF(b.floor_area_sqm, 0)  AS net_sales_per_sqm,
       b.sku_days,
       b.stockout_sku_days,
       1.0 * b.stockout_sku_days / NULLIF(b.sku_days, 0)      AS stockout_rate
FROM base b;

-- -----------------------------------------------------------------------------
-- dwh.v_kpi_wilaya
-- GRAIN: one row per wilaya that has at least one store, over the WHOLE loaded period.
-- Computed from the facts grouped by wilaya (NOT as an average of store ratios), so margins,
-- selling prices and stockout rates are correctly weighted.
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_kpi_wilaya AS
WITH wilayas AS (
    SELECT st.wilaya,
           COUNT(*)              AS store_count,
           SUM(st.floor_area_sqm) AS floor_area_sqm
    FROM dwh.dim_store st
    GROUP BY st.wilaya
), sales AS (
    SELECT st.wilaya,
           SUM(s.quantity)                  AS units_sold,
           SUM(s.gross_sales)               AS gross_sales,
           SUM(s.discount_amount)           AS discount_amount,
           SUM(s.net_sales)                 AS net_sales,
           SUM(s.cogs)                      AS cogs,
           SUM(s.gross_profit)              AS gross_profit,
           COUNT(DISTINCT s.transaction_id) AS transactions
    FROM dwh.fact_sales s
    JOIN dwh.dim_store st ON st.store_key = s.store_key
    GROUP BY st.wilaya
), inv AS (
    SELECT st.wilaya,
           COUNT(*) AS sku_days,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_sku_days
    FROM dwh.fact_inventory_daily i
    JOIN dwh.dim_store st ON st.store_key = i.store_key
    GROUP BY st.wilaya
), base AS (
    SELECT w.wilaya,
           w.store_count,
           w.floor_area_sqm,
           COALESCE(s.units_sold, 0)        AS units_sold,
           COALESCE(s.gross_sales, 0)       AS gross_sales,
           COALESCE(s.discount_amount, 0)   AS discount_amount,
           COALESCE(s.net_sales, 0)         AS net_sales,
           COALESCE(s.cogs, 0)              AS cogs,
           COALESCE(s.gross_profit, 0)      AS gross_profit,
           COALESCE(s.transactions, 0)      AS transactions,
           COALESCE(i.sku_days, 0)          AS sku_days,
           COALESCE(i.stockout_sku_days, 0) AS stockout_sku_days
    FROM wilayas w
    LEFT JOIN sales s ON s.wilaya = w.wilaya
    LEFT JOIN inv i   ON i.wilaya = w.wilaya
)
SELECT b.wilaya,
       b.store_count,
       b.floor_area_sqm,
       b.units_sold,
       b.gross_sales,
       b.discount_amount,
       b.net_sales,
       b.cogs,
       b.gross_profit,
       b.transactions,
       1.0 * b.gross_profit    / NULLIF(b.net_sales, 0)       AS gross_margin,
       1.0 * b.net_sales       / NULLIF(b.units_sold, 0)      AS avg_selling_price,
       1.0 * b.discount_amount / NULLIF(b.gross_sales, 0)     AS discount_rate,
       1.0 * b.net_sales       / NULLIF(b.transactions, 0)    AS avg_basket_value,
       1.0 * b.net_sales       / NULLIF(b.floor_area_sqm, 0)  AS net_sales_per_sqm,
       b.sku_days,
       b.stockout_sku_days,
       1.0 * b.stockout_sku_days / NULLIF(b.sku_days, 0)      AS stockout_rate
FROM base b;

-- -----------------------------------------------------------------------------
-- dwh.v_stock_availability_daily
-- GRAIN: one row per store x calendar date (all assorted SKUs of that store that day).
-- Built only from fact_inventory_daily, so no sales or promotion join can inflate it.
-- sku_count = assorted SKUs in the store that day. stockout_sku_count = SKUs with closing
-- stock 0 (the observable stockout definition). Rates are fractions of sku_count.
-- To roll up to network or month, SUM the counts and divide: never average the rates.
-- closing_stock_units is a SNAPSHOT (stock on hand at day end): sum it over stores for one
-- date, but never sum it across dates.
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_stock_availability_daily AS
WITH agg AS (
    SELECT i.store_key,
           i.date_key,
           COUNT(*)           AS sku_count,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_sku_count,
           SUM(i.units_sold)    AS units_sold,
           SUM(i.closing_stock) AS closing_stock_units
    FROM dwh.fact_inventory_daily i
    GROUP BY i.store_key, i.date_key
)
SELECT a.store_key,
       st.store_code,
       st.store_name,
       st.wilaya,
       a.date_key,
       d.full_date,
       a.sku_count,
       a.stockout_sku_count,
       a.sku_count - a.stockout_sku_count AS in_stock_sku_count,
       1.0 * (a.sku_count - a.stockout_sku_count) / NULLIF(a.sku_count, 0) AS availability_rate,
       1.0 * a.stockout_sku_count / NULLIF(a.sku_count, 0)                 AS stockout_rate,
       a.units_sold,
       a.closing_stock_units
FROM agg a
JOIN dwh.dim_store st ON st.store_key = a.store_key
JOIN dwh.dim_date d   ON d.date_key = a.date_key;

-- -----------------------------------------------------------------------------
-- dwh.v_inventory_risk
-- GRAIN: one row per store x product that has an inventory row on the LATEST loaded
-- inventory date (the snapshot date). It is a current-position view, not a history.
-- Recent demand = observed units sold over the last 14 calendar days up to the snapshot
-- (fewer if the calendar is shorter). Observed sales are censored by stockouts, so cover is
-- optimistic for SKUs that were out of stock. Latent demand is NOT used.
-- days_of_cover = closing_stock / average daily units (NULL when no recent sales).
-- Risk tiers (assumption: fixed thresholds in days, adjust here if needed):
--   1 Stockout        closing stock is 0
--   6 No recent sales stock on hand but zero units sold in the window
--   2 Critical        cover below 3 days
--   3 Low             cover below 7 days
--   5 Overstock       cover above 60 days
--   4 Healthy         otherwise
-- stock_value_at_current_cost uses dim_product.current_unit_cost (latest standard cost).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_inventory_risk AS
WITH seq AS (
    SELECT d.date_key,
           ROW_NUMBER() OVER (ORDER BY d.date_key) AS day_seq
    FROM dwh.dim_date d
), latest AS (
    SELECT MAX(i.date_key) AS date_key
    FROM dwh.fact_inventory_daily i
), latest_seq AS (
    SELECT q.date_key, q.day_seq
    FROM seq q
    JOIN latest l ON l.date_key = q.date_key
), recent AS (
    SELECT i.store_key,
           i.product_key,
           SUM(i.units_sold) AS units_sold_14d,
           COUNT(*)          AS days_observed_14d,
           SUM(CASE WHEN i.stockout_flag THEN 1 ELSE 0 END) AS stockout_days_14d
    FROM dwh.fact_inventory_daily i
    JOIN seq q         ON q.date_key = i.date_key
    JOIN latest_seq ls ON q.day_seq BETWEEN ls.day_seq - 13 AND ls.day_seq
    GROUP BY i.store_key, i.product_key
), snap AS (
    SELECT i.store_key, i.product_key, i.date_key, i.closing_stock, i.on_order_qty
    FROM dwh.fact_inventory_daily i
    JOIN latest l ON l.date_key = i.date_key
), metrics AS (
    SELECT sn.store_key,
           sn.product_key,
           sn.date_key AS snapshot_date_key,
           sn.closing_stock,
           sn.on_order_qty,
           r.units_sold_14d,
           r.days_observed_14d,
           r.stockout_days_14d,
           1.0 * r.units_sold_14d / NULLIF(r.days_observed_14d, 0) AS avg_daily_units_14d,
           1.0 * sn.closing_stock * r.days_observed_14d / NULLIF(r.units_sold_14d, 0) AS days_of_cover,
           1.0 * (sn.closing_stock + sn.on_order_qty) * r.days_observed_14d
               / NULLIF(r.units_sold_14d, 0) AS days_of_cover_incl_on_order
    FROM snap sn
    JOIN recent r ON r.store_key = sn.store_key AND r.product_key = sn.product_key
)
SELECT m.store_key,
       st.store_code,
       st.store_name,
       st.wilaya,
       m.product_key,
       p.sku,
       p.product_name,
       p.category_name,
       CASE WHEN p.is_perishable THEN 1 ELSE 0 END AS is_perishable,
       m.snapshot_date_key,
       d.full_date AS snapshot_date,
       m.closing_stock,
       m.on_order_qty,
       m.units_sold_14d,
       m.days_observed_14d,
       m.avg_daily_units_14d,
       m.days_of_cover,
       m.days_of_cover_incl_on_order,
       m.stockout_days_14d,
       m.closing_stock * p.current_unit_cost AS stock_value_at_current_cost,
       CASE WHEN m.closing_stock = 0 THEN 'Stockout'
            WHEN m.units_sold_14d = 0 THEN 'No recent sales'
            WHEN m.days_of_cover < 3 THEN 'Critical'
            WHEN m.days_of_cover < 7 THEN 'Low'
            WHEN m.days_of_cover > 60 THEN 'Overstock'
            ELSE 'Healthy' END AS risk_tier,
       CASE WHEN m.closing_stock = 0 THEN 1
            WHEN m.units_sold_14d = 0 THEN 6
            WHEN m.days_of_cover < 3 THEN 2
            WHEN m.days_of_cover < 7 THEN 3
            WHEN m.days_of_cover > 60 THEN 5
            ELSE 4 END AS risk_tier_order
FROM metrics m
JOIN dwh.dim_store st   ON st.store_key = m.store_key
JOIN dwh.dim_product p  ON p.product_key = m.product_key
JOIN dwh.dim_date d     ON d.date_key = m.snapshot_date_key;

-- -----------------------------------------------------------------------------
-- dwh.v_dim_date_bi
-- GRAIN: one row per calendar date (same rows as dwh.dim_date). Power BI date table:
-- every dim_date column unchanged, plus helpers for sorting and time intelligence.
-- year_month (yyyymm integer) sorts year_month_label. prior_year_date_key = date_key - 10000.
-- day_seq numbers the loaded days 1..N with no gaps. is_complete_month = 1 when all calendar
-- days of the month are loaded. is_last_loaded_date marks the latest loaded date.
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_dim_date_bi AS
SELECT d.date_key,
       d.full_date,
       d.year,
       d.quarter,
       d.month,
       d.month_name,
       d.day_of_month,
       d.day_of_week,
       d.day_name,
       d.iso_week,
       CASE WHEN d.is_weekend         THEN 1 ELSE 0 END AS is_weekend,
       CASE WHEN d.is_ramadan         THEN 1 ELSE 0 END AS is_ramadan,
       d.ramadan_day,
       CASE WHEN d.is_eid_al_fitr     THEN 1 ELSE 0 END AS is_eid_al_fitr,
       CASE WHEN d.is_eid_al_adha     THEN 1 ELSE 0 END AS is_eid_al_adha,
       CASE WHEN d.is_pre_eid_al_fitr THEN 1 ELSE 0 END AS is_pre_eid_al_fitr,
       CASE WHEN d.is_pre_eid_al_adha THEN 1 ELSE 0 END AS is_pre_eid_al_adha,
       d.year * 100 + d.month AS year_month,
       CAST(d.year AS VARCHAR(4)) || '-'
           || CASE WHEN d.month < 10 THEN '0' || CAST(d.month AS VARCHAR(2))
                   ELSE CAST(d.month AS VARCHAR(2)) END AS year_month_label,
       CAST(d.year AS VARCHAR(4)) || '-Q' || CAST(d.quarter AS VARCHAR(1)) AS year_quarter_label,
       CASE WHEN d.month IN (1, 3, 5, 7, 8, 10, 12) THEN 31
            WHEN d.month IN (4, 6, 9, 11) THEN 30
            WHEN (d.year % 4 = 0 AND d.year % 100 <> 0) OR d.year % 400 = 0 THEN 29
            ELSE 28 END AS days_in_month,
       CASE WHEN COUNT(*) OVER (PARTITION BY d.year, d.month) =
                 CASE WHEN d.month IN (1, 3, 5, 7, 8, 10, 12) THEN 31
                      WHEN d.month IN (4, 6, 9, 11) THEN 30
                      WHEN (d.year % 4 = 0 AND d.year % 100 <> 0) OR d.year % 400 = 0 THEN 29
                      ELSE 28 END
            THEN 1 ELSE 0 END AS is_complete_month,
       d.date_key - 10000 AS prior_year_date_key,
       ROW_NUMBER() OVER (ORDER BY d.date_key) AS day_seq,
       CASE WHEN d.date_key = MAX(d.date_key) OVER () THEN 1 ELSE 0 END AS is_last_loaded_date
FROM dwh.dim_date d;
