"""Catalogue of the Milestone 3 SQL views (sql/schema/005_forecast_metrics.sql) for `dzmi query`."""
from __future__ import annotations

FORECAST_VIEWS: dict[str, tuple[str, list[str]]] = {
    "v_forecast_accuracy": ("one row per backtest x model x evaluation basis x level",
                            ["backtest_run_id", "evaluation_basis", "level", "mae"]),
    "v_forecast_accuracy_by_horizon": ("one row per backtest x model x basis x level x horizon bucket",
                                       ["backtest_run_id", "evaluation_basis", "level", "model_name", "horizon_bucket"]),
}
