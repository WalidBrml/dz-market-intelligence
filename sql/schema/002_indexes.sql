-- Secondary indexes (primary keys and UNIQUE constraints already create their own).
CREATE INDEX IF NOT EXISTS ix_sales_date          ON dwh.fact_sales (date_key);
CREATE INDEX IF NOT EXISTS ix_sales_store_date    ON dwh.fact_sales (store_key, date_key);
CREATE INDEX IF NOT EXISTS ix_sales_product_date  ON dwh.fact_sales (product_key, date_key);
CREATE INDEX IF NOT EXISTS ix_sales_customer      ON dwh.fact_sales (customer_key);
CREATE INDEX IF NOT EXISTS ix_inv_date            ON dwh.fact_inventory_daily (date_key);
CREATE INDEX IF NOT EXISTS ix_inv_product_date    ON dwh.fact_inventory_daily (product_key, date_key);
CREATE INDEX IF NOT EXISTS ix_promo_date          ON dwh.fact_promotions (date_key);
CREATE INDEX IF NOT EXISTS ix_txn_store_date      ON ops.sales_transaction (store_id, transaction_date);
CREATE INDEX IF NOT EXISTS ix_txnline_product     ON ops.sales_transaction_line (product_id);
CREATE INDEX IF NOT EXISTS ix_product_category    ON ops.product (category_id);
