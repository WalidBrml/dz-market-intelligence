-- DZ Market Intelligence - schema (PostgreSQL target; SQLite fallback via dzmi.ingestion.db)
-- All statements are idempotent (IF NOT EXISTS) and non-destructive.
-- ALL DATA LOADED INTO THESE TABLES IS SYNTHETIC (see ops.data_provenance).
--
-- ops : normalized operational-style source model
-- dwh : star-schema analytical warehouse

CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS dwh;

-- ===================================================================== ops
CREATE TABLE IF NOT EXISTS ops.data_provenance (
    provenance_key   VARCHAR(60) PRIMARY KEY,
    provenance_value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ops.category (
    category_id   INTEGER PRIMARY KEY,
    category_name VARCHAR(80) NOT NULL UNIQUE,
    is_perishable BOOLEAN NOT NULL
);

-- Grain: one row per product (SKU).
CREATE TABLE IF NOT EXISTS ops.product (
    product_id    INTEGER PRIMARY KEY,
    sku           VARCHAR(20) NOT NULL UNIQUE,
    product_name  VARCHAR(120) NOT NULL,
    category_id   INTEGER NOT NULL REFERENCES ops.category (category_id),
    brand         VARCHAR(40) NOT NULL,
    pack_size     INTEGER NOT NULL CHECK (pack_size > 0)   -- supplier case size (units)
);

-- Grain: one row per product per price validity period (calendar year in the simulation).
CREATE TABLE IF NOT EXISTS ops.product_price_history (
    product_id  INTEGER NOT NULL REFERENCES ops.product (product_id),
    valid_from  DATE NOT NULL,
    list_price  NUMERIC(12,2) NOT NULL CHECK (list_price > 0),   -- DZD
    unit_cost   NUMERIC(12,2) NOT NULL CHECK (unit_cost > 0),    -- DZD, standard cost
    PRIMARY KEY (product_id, valid_from),
    CHECK (unit_cost < list_price)
);

CREATE TABLE IF NOT EXISTS ops.store (
    store_id        INTEGER PRIMARY KEY,
    store_code      VARCHAR(10) NOT NULL UNIQUE,
    store_name      VARCHAR(80) NOT NULL,
    wilaya          VARCHAR(50) NOT NULL,
    city            VARCHAR(50) NOT NULL,
    store_format    VARCHAR(20) NOT NULL CHECK (store_format IN ('Hypermarket', 'Supermarket', 'Express')),
    floor_area_sqm  INTEGER NOT NULL CHECK (floor_area_sqm > 0),
    opened_date     DATE NOT NULL
);

-- Which products each store carries (defines the expected inventory grain).
CREATE TABLE IF NOT EXISTS ops.store_assortment (
    store_id   INTEGER NOT NULL REFERENCES ops.store (store_id),
    product_id INTEGER NOT NULL REFERENCES ops.product (product_id),
    lead_time_days INTEGER NOT NULL CHECK (lead_time_days >= 1),
    PRIMARY KEY (store_id, product_id)
);

-- Optional loyalty customers (transactions may be anonymous).
CREATE TABLE IF NOT EXISTS ops.customer (
    customer_id    INTEGER PRIMARY KEY,
    home_store_id  INTEGER NOT NULL REFERENCES ops.store (store_id),
    wilaya         VARCHAR(50) NOT NULL,
    household_type VARCHAR(30) NOT NULL,
    signup_date    DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS ops.promotion (
    promotion_id  INTEGER PRIMARY KEY,
    promotion_name VARCHAR(100) NOT NULL,
    promotion_type VARCHAR(40) NOT NULL,
    discount_pct   NUMERIC(5,2) NOT NULL CHECK (discount_pct > 0 AND discount_pct <= 90),
    start_date     DATE NOT NULL,
    end_date       DATE NOT NULL,
    CHECK (end_date >= start_date)
);

-- Which store/product combinations a promotion applies to.
CREATE TABLE IF NOT EXISTS ops.promotion_scope (
    promotion_id INTEGER NOT NULL REFERENCES ops.promotion (promotion_id),
    store_id     INTEGER NOT NULL REFERENCES ops.store (store_id),
    product_id   INTEGER NOT NULL REFERENCES ops.product (product_id),
    PRIMARY KEY (promotion_id, store_id, product_id)
);

-- Grain: one row per sales transaction (basket).
CREATE TABLE IF NOT EXISTS ops.sales_transaction (
    transaction_id   BIGINT PRIMARY KEY,
    store_id         INTEGER NOT NULL REFERENCES ops.store (store_id),
    customer_id      INTEGER REFERENCES ops.customer (customer_id),
    transaction_ts   TIMESTAMP NOT NULL,
    transaction_date DATE NOT NULL,
    payment_method   VARCHAR(30) NOT NULL
);

-- Grain: one row per product per transaction.
CREATE TABLE IF NOT EXISTS ops.sales_transaction_line (
    transaction_id  BIGINT NOT NULL REFERENCES ops.sales_transaction (transaction_id),
    line_no         INTEGER NOT NULL CHECK (line_no >= 1),
    product_id      INTEGER NOT NULL REFERENCES ops.product (product_id),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(12,2) NOT NULL CHECK (unit_price > 0),
    discount_amount NUMERIC(12,2) NOT NULL CHECK (discount_amount >= 0),
    promotion_id    INTEGER REFERENCES ops.promotion (promotion_id),
    PRIMARY KEY (transaction_id, line_no),
    CHECK (discount_amount <= quantity * unit_price + 0.005)
);

-- SIMULATION GROUND TRUTH ONLY. In real life latent demand is unobservable.
-- Grain: store x product x day. latent_demand_units >= observed units sold;
-- the difference is demand lost to stockouts. NEVER use as a model feature.
CREATE TABLE IF NOT EXISTS ops.sim_latent_demand (
    store_id            INTEGER NOT NULL REFERENCES ops.store (store_id),
    product_id          INTEGER NOT NULL REFERENCES ops.product (product_id),
    demand_date         DATE NOT NULL,
    latent_demand_units INTEGER NOT NULL CHECK (latent_demand_units >= 0),
    unmet_demand_units  INTEGER NOT NULL CHECK (unmet_demand_units >= 0),
    PRIMARY KEY (store_id, product_id, demand_date)
);

-- ===================================================================== dwh
-- Grain: one row per calendar day.
CREATE TABLE IF NOT EXISTS dwh.dim_date (
    date_key             INTEGER PRIMARY KEY,           -- yyyymmdd
    full_date            DATE NOT NULL UNIQUE,
    year                 SMALLINT NOT NULL,
    quarter              SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month                SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name           VARCHAR(12) NOT NULL,
    day_of_month         SMALLINT NOT NULL,
    day_of_week          SMALLINT NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),   -- Monday = 1
    day_name             VARCHAR(12) NOT NULL,
    iso_week             SMALLINT NOT NULL,
    is_weekend           BOOLEAN NOT NULL,              -- Algeria: Friday and Saturday
    is_ramadan           BOOLEAN NOT NULL,
    ramadan_day          SMALLINT,
    is_eid_al_fitr       BOOLEAN NOT NULL,
    is_eid_al_adha       BOOLEAN NOT NULL,
    is_pre_eid_al_fitr   BOOLEAN NOT NULL,
    is_pre_eid_al_adha   BOOLEAN NOT NULL
);

-- Grain: one row per store.
CREATE TABLE IF NOT EXISTS dwh.dim_store (
    store_key      INTEGER PRIMARY KEY,
    store_id       INTEGER NOT NULL UNIQUE,
    store_code     VARCHAR(10) NOT NULL UNIQUE,
    store_name     VARCHAR(80) NOT NULL,
    wilaya         VARCHAR(50) NOT NULL,
    city           VARCHAR(50) NOT NULL,
    store_format   VARCHAR(20) NOT NULL,
    floor_area_sqm INTEGER NOT NULL,
    opened_date    DATE NOT NULL
);

-- Grain: one row per product. Prices are the LATEST list price/cost; per-sale values live in fact_sales.
CREATE TABLE IF NOT EXISTS dwh.dim_product (
    product_key         INTEGER PRIMARY KEY,
    product_id          INTEGER NOT NULL UNIQUE,
    sku                 VARCHAR(20) NOT NULL UNIQUE,
    product_name        VARCHAR(120) NOT NULL,
    category_name       VARCHAR(80) NOT NULL,
    is_perishable       BOOLEAN NOT NULL,
    brand               VARCHAR(40) NOT NULL,
    pack_size           INTEGER NOT NULL,
    current_list_price  NUMERIC(12,2) NOT NULL,
    current_unit_cost   NUMERIC(12,2) NOT NULL
);

-- Grain: one row per loyalty customer.
CREATE TABLE IF NOT EXISTS dwh.dim_customer (
    customer_key   INTEGER PRIMARY KEY,
    customer_id    INTEGER NOT NULL UNIQUE,
    wilaya         VARCHAR(50) NOT NULL,
    household_type VARCHAR(30) NOT NULL,
    signup_date    DATE NOT NULL
);

-- Grain: one row per promotion campaign.
CREATE TABLE IF NOT EXISTS dwh.dim_promotion (
    promotion_key  INTEGER PRIMARY KEY,
    promotion_name VARCHAR(100) NOT NULL,
    promotion_type VARCHAR(40) NOT NULL,
    discount_pct   NUMERIC(5,2) NOT NULL,
    start_date     DATE NOT NULL,
    end_date       DATE NOT NULL
);

-- Grain: one row per transaction line (product within a basket).
-- Money columns are DZD. gross = qty*price; net = gross - discount; profit = net - cogs.
CREATE TABLE IF NOT EXISTS dwh.fact_sales (
    sales_line_id  BIGINT PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    line_no        INTEGER NOT NULL,
    date_key       INTEGER NOT NULL REFERENCES dwh.dim_date (date_key),
    store_key      INTEGER NOT NULL REFERENCES dwh.dim_store (store_key),
    product_key    INTEGER NOT NULL REFERENCES dwh.dim_product (product_key),
    customer_key   INTEGER REFERENCES dwh.dim_customer (customer_key),
    promotion_key  INTEGER REFERENCES dwh.dim_promotion (promotion_key),
    quantity       INTEGER NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(12,2) NOT NULL CHECK (unit_price > 0),
    unit_cost      NUMERIC(12,2) NOT NULL CHECK (unit_cost > 0),
    gross_sales    NUMERIC(14,2) NOT NULL,
    discount_amount NUMERIC(14,2) NOT NULL CHECK (discount_amount >= 0),
    net_sales      NUMERIC(14,2) NOT NULL,
    cogs           NUMERIC(14,2) NOT NULL,
    gross_profit   NUMERIC(14,2) NOT NULL,
    UNIQUE (transaction_id, line_no),
    CHECK (ABS(gross_sales - quantity * unit_price) < 0.005),
    CHECK (ABS(net_sales - (gross_sales - discount_amount)) < 0.005),
    CHECK (ABS(cogs - quantity * unit_cost) < 0.005),
    CHECK (ABS(gross_profit - (net_sales - cogs)) < 0.005)
);

-- Grain: one row per store x product x day (every day for every assorted product).
-- Do NOT join this directly to fact_sales lines (fan-out). Use dwh.v_store_product_day.
CREATE TABLE IF NOT EXISTS dwh.fact_inventory_daily (
    store_key      INTEGER NOT NULL REFERENCES dwh.dim_store (store_key),
    product_key    INTEGER NOT NULL REFERENCES dwh.dim_product (product_key),
    date_key       INTEGER NOT NULL REFERENCES dwh.dim_date (date_key),
    opening_stock  INTEGER NOT NULL CHECK (opening_stock >= 0),
    received_qty   INTEGER NOT NULL CHECK (received_qty >= 0),
    units_sold     INTEGER NOT NULL CHECK (units_sold >= 0),
    closing_stock  INTEGER NOT NULL CHECK (closing_stock >= 0),
    on_order_qty   INTEGER NOT NULL CHECK (on_order_qty >= 0),   -- incoming stock after the day's orders
    stockout_flag  BOOLEAN NOT NULL,                              -- closing_stock = 0 (observable)
    PRIMARY KEY (store_key, product_key, date_key),
    CHECK (closing_stock = opening_stock + received_qty - units_sold),
    CHECK ((closing_stock = 0) = stockout_flag)
);

-- Grain: one row per store x product x day on which a promotion is active.
CREATE TABLE IF NOT EXISTS dwh.fact_promotions (
    store_key     INTEGER NOT NULL REFERENCES dwh.dim_store (store_key),
    product_key   INTEGER NOT NULL REFERENCES dwh.dim_product (product_key),
    date_key      INTEGER NOT NULL REFERENCES dwh.dim_date (date_key),
    promotion_key INTEGER NOT NULL REFERENCES dwh.dim_promotion (promotion_key),
    discount_pct  NUMERIC(5,2) NOT NULL,
    PRIMARY KEY (store_key, product_key, date_key)
);

-- Forecasting registry (populated from Milestone 5; created now so the model is stable).
CREATE TABLE IF NOT EXISTS dwh.model_run (
    forecast_run_id    VARCHAR(40) PRIMARY KEY,
    model_name         VARCHAR(60) NOT NULL,
    model_version      VARCHAR(40) NOT NULL,
    trained_at         TIMESTAMP NOT NULL,
    train_end_date     DATE NOT NULL,
    horizon_days       INTEGER NOT NULL CHECK (horizon_days > 0),
    feature_definition TEXT,
    notes              TEXT
);

-- Grain: one row per run x store x product x target day.
CREATE TABLE IF NOT EXISTS dwh.fact_demand_forecast (
    forecast_run_id VARCHAR(40) NOT NULL REFERENCES dwh.model_run (forecast_run_id),
    store_key       INTEGER NOT NULL REFERENCES dwh.dim_store (store_key),
    product_key     INTEGER NOT NULL REFERENCES dwh.dim_product (product_key),
    date_key        INTEGER NOT NULL REFERENCES dwh.dim_date (date_key),
    horizon_step    INTEGER NOT NULL CHECK (horizon_step >= 1),
    predicted_units NUMERIC(12,3) NOT NULL CHECK (predicted_units >= 0),
    lower_bound     NUMERIC(12,3),     -- NULL unless the method supports defensible intervals
    upper_bound     NUMERIC(12,3),
    PRIMARY KEY (forecast_run_id, store_key, product_key, date_key)
);
