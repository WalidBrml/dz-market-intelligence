-- Milestone 3 post-load validation of forecast runs and backtest metrics (sql/schema/005_forecast_metrics.sql).
-- Each statement returns ONE row: (violations). A check passes when violations = 0.
-- Metadata comment format:  -- check: <name> | <severity: error|warning> | <description>
-- All checks pass trivially on a database without forecast runs. Portable across PostgreSQL and SQLite.

-- check: fc_forecast_after_train_end | error | forecast rows whose target day is not strictly after the run train_end_date (leakage guard)
SELECT COUNT(*) AS violations
FROM dwh.fact_demand_forecast f
JOIN dwh.model_run r ON r.forecast_run_id = f.forecast_run_id
JOIN dwh.dim_date d  ON d.date_key = f.date_key
WHERE d.full_date <= r.train_end_date;

-- check: fc_horizon_step_matches_dates | error | forecast rows whose horizon_step is not the number of days between train_end_date and the target day
SELECT COUNT(*) AS violations
FROM dwh.fact_demand_forecast f
JOIN dwh.model_run r ON r.forecast_run_id = f.forecast_run_id
JOIN (SELECT date_key, ROW_NUMBER() OVER (ORDER BY date_key) AS day_seq FROM dwh.dim_date) t ON t.date_key = f.date_key
JOIN (SELECT full_date, ROW_NUMBER() OVER (ORDER BY date_key) AS day_seq FROM dwh.dim_date) e ON e.full_date = r.train_end_date
WHERE f.horizon_step <> t.day_seq - e.day_seq;

-- check: fc_horizon_within_run | error | forecast rows with horizon_step above the run horizon_days
SELECT COUNT(*) AS violations
FROM dwh.fact_demand_forecast f
JOIN dwh.model_run r ON r.forecast_run_id = f.forecast_run_id
WHERE f.horizon_step > r.horizon_days;

-- check: fc_forecast_grid_complete | error | forecast runs where rows differ from series x horizon_days (a series is missing a day)
SELECT COUNT(*) AS violations
FROM (SELECT f.forecast_run_id, r.horizon_days, COUNT(*) AS n_rows
      FROM dwh.fact_demand_forecast f
      JOIN dwh.model_run r ON r.forecast_run_id = f.forecast_run_id
      GROUP BY f.forecast_run_id, r.horizon_days) a
JOIN (SELECT p.forecast_run_id, COUNT(*) AS n_series
      FROM (SELECT DISTINCT forecast_run_id, store_key, product_key FROM dwh.fact_demand_forecast) p
      GROUP BY p.forecast_run_id) b ON b.forecast_run_id = a.forecast_run_id
WHERE a.n_rows <> b.n_series * a.horizon_days;

-- check: fc_forecast_run_has_rows | warning | model_run rows without any forecast (an interrupted run)
SELECT COUNT(*) AS violations
FROM dwh.model_run r
WHERE NOT EXISTS (SELECT 1 FROM dwh.fact_demand_forecast f WHERE f.forecast_run_id = r.forecast_run_id);

-- check: fc_backtest_run_has_metrics | warning | backtest_run rows without any metric (an interrupted run)
SELECT COUNT(*) AS violations
FROM dwh.backtest_run b
WHERE NOT EXISTS (SELECT 1 FROM dwh.forecast_backtest_metric m WHERE m.backtest_run_id = b.backtest_run_id);

-- check: fc_backtest_same_evaluation_set | error | backtests where models were scored on different numbers of observations (unfair comparison)
SELECT COUNT(*) AS violations
FROM (SELECT t.backtest_run_id, t.evaluation_basis, t.level
      FROM (SELECT backtest_run_id, model_name, evaluation_basis, level, SUM(n_obs) AS n
            FROM dwh.forecast_backtest_metric
            GROUP BY backtest_run_id, model_name, evaluation_basis, level) t
      GROUP BY t.backtest_run_id, t.evaluation_basis, t.level
      HAVING MIN(t.n) <> MAX(t.n)) x;

-- check: fc_backtest_has_baseline | warning | backtests without the seasonal_naive baseline (skill scores cannot be computed)
SELECT COUNT(*) AS violations
FROM dwh.backtest_run b
WHERE NOT EXISTS (SELECT 1 FROM dwh.forecast_backtest_metric m
                  WHERE m.backtest_run_id = b.backtest_run_id AND m.model_name = 'seasonal_naive');

-- check: fc_backtest_sums_consistent | error | metric rows that break sum|e| >= |sum e| or (sum|e|)^2 <= n * sum e^2
SELECT COUNT(*) AS violations
FROM dwh.forecast_backtest_metric
WHERE sum_abs_error + 0.01 < ABS(sum_pred - sum_actual)
   OR sum_abs_error * sum_abs_error > n_obs * sum_sq_error * 1.000001 + 0.01;
