-- =============================================================================
-- Milestone 3 - forecast evaluation storage (ADDITIVE: new tables and views only, nothing is altered or dropped
-- except the two views below, which are recreated). Safe to apply on an existing database with dzmi init-db.
-- ALL DATA IS SYNTHETIC. Forecast runs themselves use the existing dwh.model_run and dwh.fact_demand_forecast.
--
-- Metrics are stored as ADDITIVE SUMS (n_obs, sum_actual, sum_pred, sum_abs_error, sum_sq_error) so any roll-up is
-- exact: MAE = sum_abs_error / n_obs, WAPE = sum_abs_error / sum_actual, bias = (sum_pred - sum_actual) / sum_actual.
-- RMSE needs a square root, which SQLite does not guarantee, so views expose mse and RMSE = sqrt(mse) is taken outside SQL.
-- Comments in this file must not contain a semicolon (the statement splitter cuts on it).
-- =============================================================================

-- Grain: one row per backtest (all models, all origins of one dzmi backtest execution).
CREATE TABLE IF NOT EXISTS dwh.backtest_run (
    backtest_run_id  VARCHAR(40) PRIMARY KEY,
    created_at       TIMESTAMP NOT NULL,
    profile          VARCHAR(40) NOT NULL,
    horizon_days     INTEGER NOT NULL CHECK (horizon_days > 0),
    n_origins        INTEGER NOT NULL CHECK (n_origins > 0),
    origin_step_days INTEGER NOT NULL CHECK (origin_step_days > 0),
    min_train_days   INTEGER NOT NULL CHECK (min_train_days > 0),
    n_series         INTEGER NOT NULL CHECK (n_series > 0),
    censor_policy    VARCHAR(20) NOT NULL CHECK (censor_policy IN ('exclude', 'keep')),
    seed             INTEGER NOT NULL,
    models           TEXT NOT NULL,
    settings_json    TEXT,
    notes            TEXT
);

-- Grain: one row per backtest x model x evaluation basis x level x origin date x horizon bucket.
-- level series_day: horizon_bucket is the day step 1..H. level store_category_week: horizon_bucket is the forecast week 1..
-- evaluation_basis observed = sales a retailer can see. observed_uncensored = same without days that ended with zero stock.
-- latent_simulation_only = simulator ground truth, never used for training, never real-world accuracy.
CREATE TABLE IF NOT EXISTS dwh.forecast_backtest_metric (
    backtest_run_id  VARCHAR(40) NOT NULL REFERENCES dwh.backtest_run (backtest_run_id),
    model_name       VARCHAR(60) NOT NULL,
    evaluation_basis VARCHAR(30) NOT NULL
        CHECK (evaluation_basis IN ('observed', 'observed_uncensored', 'latent_simulation_only')),
    level            VARCHAR(30) NOT NULL CHECK (level IN ('series_day', 'store_category_week')),
    origin_date      DATE NOT NULL,
    horizon_bucket   INTEGER NOT NULL CHECK (horizon_bucket >= 1),
    n_obs            INTEGER NOT NULL CHECK (n_obs >= 0),
    sum_actual       NUMERIC(18,3) NOT NULL CHECK (sum_actual >= 0),
    sum_pred         NUMERIC(18,3) NOT NULL CHECK (sum_pred >= 0),
    sum_abs_error    NUMERIC(18,3) NOT NULL CHECK (sum_abs_error >= 0),
    sum_sq_error     NUMERIC(24,3) NOT NULL CHECK (sum_sq_error >= 0),
    PRIMARY KEY (backtest_run_id, model_name, evaluation_basis, level, origin_date, horizon_bucket)
);

DROP VIEW IF EXISTS dwh.v_forecast_accuracy_by_horizon;
DROP VIEW IF EXISTS dwh.v_forecast_accuracy;

-- -----------------------------------------------------------------------------
-- dwh.v_forecast_accuracy
-- GRAIN: one row per backtest x model x evaluation basis x level (all origins and horizons pooled).
-- skill_mae_vs_seasonal_naive = 1 - MAE(model) / MAE(seasonal_naive) of the same run, basis and level
-- (positive = better than the baseline, NULL when the run has no seasonal_naive rows).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_forecast_accuracy AS
WITH agg AS (
    SELECT m.backtest_run_id,
           m.model_name,
           m.evaluation_basis,
           m.level,
           COUNT(DISTINCT m.origin_date) AS n_origins,
           SUM(m.n_obs)                  AS n_obs,
           SUM(m.sum_actual)             AS sum_actual,
           SUM(m.sum_pred)               AS sum_pred,
           SUM(m.sum_abs_error)          AS sum_abs_error,
           SUM(m.sum_sq_error)           AS sum_sq_error
    FROM dwh.forecast_backtest_metric m
    GROUP BY m.backtest_run_id, m.model_name, m.evaluation_basis, m.level
)
SELECT a.backtest_run_id,
       a.model_name,
       a.evaluation_basis,
       a.level,
       a.n_origins,
       a.n_obs,
       a.sum_actual,
       a.sum_pred,
       a.sum_abs_error,
       a.sum_sq_error,
       1.0 * a.sum_abs_error / NULLIF(a.n_obs, 0)                  AS mae,
       1.0 * a.sum_sq_error / NULLIF(a.n_obs, 0)                   AS mse,
       1.0 * a.sum_abs_error / NULLIF(a.sum_actual, 0)             AS wape,
       1.0 * (a.sum_pred - a.sum_actual) / NULLIF(a.sum_actual, 0) AS bias,
       1.0 - (1.0 * a.sum_abs_error / NULLIF(a.n_obs, 0))
             / NULLIF(1.0 * b.sum_abs_error / NULLIF(b.n_obs, 0), 0) AS skill_mae_vs_seasonal_naive
FROM agg a
LEFT JOIN agg b ON b.backtest_run_id = a.backtest_run_id
               AND b.evaluation_basis = a.evaluation_basis
               AND b.level = a.level
               AND b.model_name = 'seasonal_naive';

-- -----------------------------------------------------------------------------
-- dwh.v_forecast_accuracy_by_horizon
-- GRAIN: one row per backtest x model x evaluation basis x level x horizon bucket (all origins pooled).
-- Same columns as v_forecast_accuracy plus horizon_bucket (day step for series_day, week number for store_category_week).
-- -----------------------------------------------------------------------------
CREATE VIEW dwh.v_forecast_accuracy_by_horizon AS
WITH agg AS (
    SELECT m.backtest_run_id,
           m.model_name,
           m.evaluation_basis,
           m.level,
           m.horizon_bucket,
           COUNT(DISTINCT m.origin_date) AS n_origins,
           SUM(m.n_obs)                  AS n_obs,
           SUM(m.sum_actual)             AS sum_actual,
           SUM(m.sum_pred)               AS sum_pred,
           SUM(m.sum_abs_error)          AS sum_abs_error,
           SUM(m.sum_sq_error)           AS sum_sq_error
    FROM dwh.forecast_backtest_metric m
    GROUP BY m.backtest_run_id, m.model_name, m.evaluation_basis, m.level, m.horizon_bucket
)
SELECT a.backtest_run_id,
       a.model_name,
       a.evaluation_basis,
       a.level,
       a.horizon_bucket,
       a.n_origins,
       a.n_obs,
       a.sum_actual,
       a.sum_pred,
       a.sum_abs_error,
       a.sum_sq_error,
       1.0 * a.sum_abs_error / NULLIF(a.n_obs, 0)                  AS mae,
       1.0 * a.sum_sq_error / NULLIF(a.n_obs, 0)                   AS mse,
       1.0 * a.sum_abs_error / NULLIF(a.sum_actual, 0)             AS wape,
       1.0 * (a.sum_pred - a.sum_actual) / NULLIF(a.sum_actual, 0) AS bias,
       1.0 - (1.0 * a.sum_abs_error / NULLIF(a.n_obs, 0))
             / NULLIF(1.0 * b.sum_abs_error / NULLIF(b.n_obs, 0), 0) AS skill_mae_vs_seasonal_naive
FROM agg a
LEFT JOIN agg b ON b.backtest_run_id = a.backtest_run_id
               AND b.evaluation_basis = a.evaluation_basis
               AND b.level = a.level
               AND b.horizon_bucket = a.horizon_bucket
               AND b.model_name = 'seasonal_naive';
