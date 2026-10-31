-- Views that make the safe grain explicit and prevent double counting.

-- Sales aggregated to the inventory grain (store x product x day).
DROP VIEW IF EXISTS dwh.v_store_product_day;
DROP VIEW IF EXISTS dwh.v_sales_daily;

CREATE VIEW dwh.v_sales_daily AS
SELECT
    s.store_key,
    s.product_key,
    s.date_key,
    SUM(s.quantity)        AS units_sold,
    SUM(s.gross_sales)     AS gross_sales,
    SUM(s.discount_amount) AS discount_amount,
    SUM(s.net_sales)       AS net_sales,
    SUM(s.cogs)            AS cogs,
    SUM(s.gross_profit)    AS gross_profit,
    COUNT(*)               AS line_count
FROM dwh.fact_sales s
GROUP BY s.store_key, s.product_key, s.date_key;

-- One row per store x product x day: inventory + (pre-aggregated) sales + promotion.
-- Safe to SUM any column because every source is joined at the same grain.
CREATE VIEW dwh.v_store_product_day AS
SELECT
    i.store_key,
    i.product_key,
    i.date_key,
    i.opening_stock,
    i.received_qty,
    i.units_sold            AS inventory_units_sold,
    i.closing_stock,
    i.on_order_qty,
    i.stockout_flag,
    COALESCE(d.units_sold, 0)   AS sales_units,
    COALESCE(d.net_sales, 0)    AS net_sales,
    COALESCE(d.cogs, 0)         AS cogs,
    COALESCE(d.gross_profit, 0) AS gross_profit,
    p.promotion_key,
    p.discount_pct
FROM dwh.fact_inventory_daily i
LEFT JOIN dwh.v_sales_daily d
       ON d.store_key = i.store_key AND d.product_key = i.product_key AND d.date_key = i.date_key
LEFT JOIN dwh.fact_promotions p
       ON p.store_key = i.store_key AND p.product_key = i.product_key AND p.date_key = i.date_key;
