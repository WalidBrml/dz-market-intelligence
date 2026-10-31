"""Console and Markdown reports of a backtest. ASCII only (safe in any Windows console code page)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .metrics import summarise
from .models import MODEL_DESCRIPTIONS

BASIS_TITLES = {
    "observed": "Observed sales (what a retailer can see)",
    "observed_uncensored": "Observed sales, days without a stockout only",
    "latent_simulation_only": "Latent demand (SIMULATION ONLY: unobservable in real life)",
}
LEVEL_TITLES = {"series_day": "series x day", "store_category_week": "store x category x week"}
DISCLAIMER = ("SYNTHETIC DATA. These scores describe how the methods behave on the simulator, not on real Algerian "
              "retail demand, and no business value is implied.")


def _table(df: pd.DataFrame) -> str:
    cols = ["model_name", "n_obs", "mae", "rmse", "wape", "bias", "skill_vs_seasonal_naive"]
    out = df.sort_values("mae")[cols].copy()
    out["n_obs"] = out["n_obs"].map(lambda v: f"{int(v):,}")
    for c in ("mae", "rmse"):
        out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    for c in ("wape", "bias", "skill_vs_seasonal_naive"):
        out[c] = out[c].map(lambda v: "n/a" if pd.isna(v) else f"{v * 100:+.1f}%" if c != "wape" else f"{v * 100:.1f}%")
    out.columns = ["model", "n_obs", "MAE", "RMSE", "WAPE", "bias", "skill_vs_sn"]
    return out.to_string(index=False)


def zero_reference(sub: pd.DataFrame) -> str:
    """MAE of the trivial forecast 'no demand anywhere' on the same observations: sum_actual / n_obs (WAPE 100%)."""
    r = sub.iloc[0]
    if not r["n_obs"]:
        return "Reference: no observations."
    return (f"Reference: forecasting 0 everywhere would give MAE {r['sum_actual'] / r['n_obs']:.3f} (WAPE 100.0%). "
            "A model above that is worse than predicting no demand on this measure.")


def format_timing(result: BacktestResult) -> str:
    return "Seconds spent per model (all origins): " + ", ".join(f"{m} {t:.1f}" for m, t in result.seconds.items())


def format_summary(result: BacktestResult) -> str:
    """Text tables: one per (evaluation basis, level), models sorted by MAE (lower is better)."""
    s = summarise(result.metrics)
    parts = []
    for basis in ("observed", "observed_uncensored", "latent_simulation_only"):
        for level in ("series_day", "store_category_week"):
            sub = s[(s["evaluation_basis"] == basis) & (s["level"] == level)]
            if sub.empty:
                continue
            parts.append(f"{BASIS_TITLES[basis]} - {LEVEL_TITLES[level]}\n{_table(sub)}\n{zero_reference(sub)}")
    return "\n\n".join(parts + [format_timing(result)])


def markdown_report(result: BacktestResult, run_id: str, saved: bool) -> str:
    st = result.settings
    lines = [f"# Backtest report `{run_id}`", "", f"> {DISCLAIMER}", "",
             f"* Profile: `{st.profile}`; series: {result.n_series:,}; horizon: {st.horizon_days} days",
             f"* Origins ({len(result.origins)}): " + ", ".join(d.strftime("%Y-%m-%d") for d in result.origins),
             f"* Censoring policy: `{st.censor_policy}` (days that ended with zero stock "
             f"{'are ignored as training/level information' if st.censor_policy == 'exclude' else 'are used as ordinary sales'})",
             f"* Seed: {st.seed}; models: {', '.join(st.models)}; saved to the database: {'yes' if saved else 'no'}", "",
             "Lower MAE / RMSE / WAPE is better. `bias` > 0 means over-forecasting. `skill_vs_sn` = 1 - MAE / MAE of the "
             "seasonal-naive baseline (positive = better than the baseline).", ""]
    s = summarise(result.metrics)
    for basis in ("observed", "observed_uncensored", "latent_simulation_only"):
        for level in ("series_day", "store_category_week"):
            sub = s[(s["evaluation_basis"] == basis) & (s["level"] == level)]
            if sub.empty:
                continue
            lines += [f"## {BASIS_TITLES[basis]} - {LEVEL_TITLES[level]}", "", "```", _table(sub), "```", "",
                      zero_reference(sub), ""]
    lines += ["## Timing", "", format_timing(result), ""]
    lines += ["## Models", ""] + [f"* `{m}`: {MODEL_DESCRIPTIONS[m]}" for m in st.models]
    lines += ["", "## How to read this", "",
              "* Forecasts use only data up to each origin (see docs/forecasting.md). Accuracy is measured on OBSERVED sales, "
              "which stockouts truncate: a model can look accurate while forecasting censored sales.",
              "* With intermittent demand (mostly zero days) the daily MAE and WAPE are dominated by the zeros: compare with the "
              "'forecasting 0' reference line, and prefer the store x category x week level to judge the models.",
              "* The latent-demand section, when present, is a simulation-only diagnostic and was never used for training.", ""]
    return "\n".join(lines)


def write_report(result: BacktestResult, run_id: str, saved: bool, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"forecast_backtest_{run_id}.md"
    path.write_text(markdown_report(result, run_id, saved), encoding="utf-8", newline="\n")
    return path
