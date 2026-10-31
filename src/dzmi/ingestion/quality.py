"""Pandas-based data-quality checks on the generated tables (run BEFORE database load).

Every check returns a CheckResult. Severity 'error' blocks the pipeline; 'warning' is reported only.
The same business rules are re-checked inside the database by sql/validation/001_integrity_checks.sql.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import SYNTHETIC_DISCLAIMER
from ..errors import DataQualityError


@dataclass
class CheckResult:
    name: str
    severity: str
    passed: bool
    violations: int
    description: str
    hint: str = ""
    samples: list = field(default_factory=list)


def _res(name, desc, bad: pd.Series | np.ndarray | int, hint: str, severity="error", sample_df=None) -> CheckResult:
    n = int(bad.sum()) if hasattr(bad, "sum") else int(bad)
    samples = []
    if n and sample_df is not None:
        samples = sample_df[bad].head(3).astype(str).to_dict("records") if hasattr(bad, "sum") else []
    return CheckResult(name, severity, n == 0, n, desc, hint if n else "", samples)


def run_quality_checks(raw: dict[str, pd.DataFrame], dwh: dict[str, pd.DataFrame]) -> list[CheckResult]:
    R: list[CheckResult] = []
    fs, fi, dd = dwh["fact_sales"], dwh["fact_inventory_daily"], dwh["dim_date"]

    # ---- primary keys / duplicates
    pk = {
        "dim_date": ["date_key"], "dim_store": ["store_key"], "dim_product": ["product_key"],
        "dim_customer": ["customer_key"], "dim_promotion": ["promotion_key"], "fact_sales": ["sales_line_id"],
        "fact_inventory_daily": ["store_key", "product_key", "date_key"],
        "fact_promotions": ["store_key", "product_key", "date_key"],
    }
    for t, cols in pk.items():
        R.append(_res(f"pk_unique_{t}", f"Primary key {cols} unique in {t}", dwh[t].duplicated(cols).sum(),
                      f"Duplicate keys in {t}: regenerate the data or inspect the transform."))
    R.append(_res("unique_transaction_line", "(transaction_id, line_no) unique",
                  fs.duplicated(["transaction_id", "line_no"]).sum(), "Duplicate transaction lines found."))
    R.append(_res("duplicate_product_in_basket", "A product appears at most once per transaction",
                  fs.duplicated(["transaction_id", "product_key"]).sum(),
                  "Same product repeated in a basket: merge the lines in the generator."))

    # ---- referential integrity
    def orphans(child, col, parent, pcol):
        return (~child[col].isin(parent[pcol])).sum()
    for name, child, col, parent, pcol in [
        ("fk_sales_store", fs, "store_key", dwh["dim_store"], "store_key"),
        ("fk_sales_product", fs, "product_key", dwh["dim_product"], "product_key"),
        ("fk_sales_date", fs, "date_key", dd, "date_key"),
        ("fk_inventory_store", fi, "store_key", dwh["dim_store"], "store_key"),
        ("fk_inventory_product", fi, "product_key", dwh["dim_product"], "product_key"),
        ("fk_inventory_date", fi, "date_key", dd, "date_key"),
        ("fk_promo_product", dwh["fact_promotions"], "product_key", dwh["dim_product"], "product_key"),
        ("fk_promo_promotion", dwh["fact_promotions"], "promotion_key", dwh["dim_promotion"], "promotion_key"),
        ("fk_line_transaction", raw["sales_transaction_line"], "transaction_id", raw["sales_transaction"], "transaction_id"),
        ("fk_line_product", raw["sales_transaction_line"], "product_id", raw["product"], "product_id"),
    ]:
        R.append(_res(name, f"Every {child.shape[0]:,}-row child key exists in parent ({col})",
                      orphans(child, col, parent, pcol), f"Orphan {col} values: parent dimension is missing rows."))
    cust = fs["customer_key"].dropna()
    R.append(_res("fk_sales_customer", "Customer keys exist in dim_customer",
                  (~cust.isin(dwh["dim_customer"]["customer_key"])).sum(), "Unknown customer keys."))

    # ---- sales arithmetic and revenue reconciliation
    tol = 0.005
    R.append(_res("sales_gross_arithmetic", "gross = quantity x unit_price",
                  ((fs["gross_sales"] - fs["quantity"] * fs["unit_price"]).abs() >= tol).sum(), "Gross sales mismatch."))
    R.append(_res("sales_net_arithmetic", "net = gross - discount",
                  ((fs["net_sales"] - (fs["gross_sales"] - fs["discount_amount"])).abs() >= tol).sum(), "Net sales mismatch."))
    R.append(_res("sales_profit_arithmetic", "gross_profit = net - cogs and cogs = quantity x unit_cost",
                  (((fs["gross_profit"] - (fs["net_sales"] - fs["cogs"])).abs() >= tol)
                   | ((fs["cogs"] - fs["quantity"] * fs["unit_cost"]).abs() >= tol)).sum(), "Profit/COGS mismatch."))
    ln = raw["sales_transaction_line"]
    ops_net = (ln["quantity"] * ln["unit_price"] - ln["discount_amount"]).sum()
    R.append(_res("revenue_reconciliation", "Net sales in fact_sales equals ops lines (within 0.01 DZD)",
                  int(abs(ops_net - fs["net_sales"].sum()) >= 0.01),
                  f"ops net {ops_net:,.2f} vs dwh net {fs['net_sales'].sum():,.2f}."))
    R.append(_res("units_reconciliation", "Units in fact_sales equal units in ops lines",
                  int(ln["quantity"].sum() != fs["quantity"].sum()), "Unit totals differ between ops and dwh."))
    R.append(_res("line_count_reconciliation", "Row counts of ops lines and fact_sales are equal",
                  int(len(ln) != len(fs)), "Line counts differ."))
    R.append(_res("non_positive_values", "Quantity, price and cost are positive; discount is non-negative",
                  ((fs["quantity"] <= 0) | (fs["unit_price"] <= 0) | (fs["unit_cost"] <= 0)
                   | (fs["discount_amount"] < 0)).sum(), "Non-positive quantities/prices/costs."))
    R.append(_res("cost_below_price", "Standard cost is below list price", (fs["unit_cost"] >= fs["unit_price"]).sum(),
                  "Negative unit margin at list price."))
    R.append(_res("null_required_fields", "No NULLs in required fact_sales columns",
                  fs[["date_key", "store_key", "product_key", "quantity", "unit_price", "unit_cost"]].isna().sum().sum(),
                  "Required fact columns contain NULLs."))

    # ---- transactions
    tx = raw["sales_transaction"]
    R.append(_res("transaction_without_lines", "Every transaction has at least one line",
                  (~tx["transaction_id"].isin(ln["transaction_id"])).sum(), "Empty baskets exist."))
    R.append(_res("transaction_date_consistency", "transaction_date equals the date part of transaction_ts",
                  (tx["transaction_ts"].dt.normalize() != tx["transaction_date"]).sum(), "Timestamp/date mismatch."))

    # ---- dates
    d = pd.to_datetime(dd["full_date"]).sort_values()
    R.append(_res("date_completeness", "dim_date is a gap-free daily series",
                  int((d.diff().dropna() != pd.Timedelta(days=1)).sum()), "Gaps in dim_date."))
    R.append(_res("date_key_matches_date", "date_key equals yyyymmdd of full_date",
                  (dd["date_key"] != (dd["full_date"].dt.year * 10000 + dd["full_date"].dt.month * 100
                                      + dd["full_date"].dt.day)).sum(), "date_key mismatch."))
    R.append(_res("sales_dates_in_range", "All sales fall inside dim_date range",
                  (~fs["date_key"].isin(dd["date_key"])).sum(), "Sales outside the calendar."))

    # ---- inventory
    R.append(_res("stock_balance", "closing = opening + received - sold",
                  (fi["closing_stock"] != fi["opening_stock"] + fi["received_qty"] - fi["units_sold"]).sum(),
                  "Stock balance violated."))
    R.append(_res("stock_non_negative", "Stock quantities are non-negative",
                  ((fi[["opening_stock", "received_qty", "units_sold", "closing_stock", "on_order_qty"]] < 0)
                   .any(axis=1)).sum(), "Negative stock values."))
    fi_s = fi.sort_values(["store_key", "product_key", "date_key"])
    prev_close = fi_s.groupby(["store_key", "product_key"])["closing_stock"].shift(1)
    R.append(_res("stock_continuity", "Opening stock equals previous day's closing stock",
                  ((prev_close.notna()) & (fi_s["opening_stock"] != prev_close)).sum(), "Stock roll-forward broken."))
    R.append(_res("stockout_flag_definition", "stockout_flag is true exactly when closing_stock = 0",
                  ((fi["closing_stock"] == 0) != fi["stockout_flag"]).sum(), "Inconsistent stockout flag."))
    exp_rows = len(raw["store_assortment"]) * len(dd)
    R.append(_res("inventory_completeness", "One inventory row per assorted store x product x day",
                  int(len(fi) != exp_rows), f"Expected {exp_rows:,} rows, found {len(fi):,}."))
    per_series = fi.groupby(["store_key", "product_key"]).size()
    R.append(_res("inventory_series_length", "Every store x product series has all days",
                  int((per_series != len(dd)).sum()), "Series with missing days."))

    # ---- sales vs inventory (no double counting)
    agg = fs.groupby(["store_key", "product_key", "date_key"], as_index=False)["quantity"].sum()
    m = fi.merge(agg, on=["store_key", "product_key", "date_key"], how="left")
    m["quantity"] = m["quantity"].fillna(0)
    R.append(_res("sales_vs_inventory_units", "Units sold per store/product/day match inventory units_sold",
                  (m["quantity"] != m["units_sold"]).sum(), "fact_sales and fact_inventory_daily disagree."))
    R.append(_res("sales_without_inventory_row", "Every sales day has an inventory row",
                  int(len(agg) - len(agg.merge(fi[["store_key", "product_key", "date_key"]],
                                                on=["store_key", "product_key", "date_key"]))),
                  "Sales exist for store/product/days without inventory."))
    naive_rows = fs.merge(fi[["store_key", "product_key", "date_key", "closing_stock"]],
                          on=["store_key", "product_key", "date_key"])
    R.append(CheckResult(
        "fanout_risk_info", "warning", True, 0,
        f"INFO: joining fact_sales lines directly to inventory repeats each inventory row; a naive "
        f"SUM(closing_stock) would be inflated {naive_rows['closing_stock'].sum() / max(fi['closing_stock'].sum(), 1):.1f}x. "
        "Use dwh.v_store_product_day.", ""))

    # ---- promotions
    pr = dwh["fact_promotions"].merge(dwh["dim_promotion"], on="promotion_key")
    pdate = pd.to_datetime(pr["date_key"].astype(str), format="%Y%m%d")
    R.append(_res("promotion_within_window", "Promotion days fall within promotion start/end",
                  ((pdate < pr["start_date"]) | (pdate > pr["end_date"])).sum(), "Promotion day outside its window."))
    R.append(_res("discount_has_promotion", "Discounted lines carry a promotion",
                  ((fs["discount_amount"] > 0) & fs["promotion_key"].isna()).sum(), "Discount without promotion."))
    pk3 = fs.dropna(subset=["promotion_key"]).merge(
        dwh["fact_promotions"][["store_key", "product_key", "date_key"]].assign(_p=1),
        on=["store_key", "product_key", "date_key"], how="left")
    R.append(_res("promo_sales_have_promo_fact", "Promotion lines match a fact_promotions day", pk3["_p"].isna().sum(),
                  "Promotional sale without a promotion day."))

    # ---- latent vs observed demand
    lat = raw["sim_latent_demand"].merge(
        raw["inventory_daily"][["store_id", "product_id", "inventory_date", "units_sold", "closing_stock"]],
        left_on=["store_id", "product_id", "demand_date"], right_on=["store_id", "product_id", "inventory_date"])
    R.append(_res("latent_ge_observed", "Latent demand >= observed sales and unmet = latent - observed",
                  ((lat["latent_demand_units"] < lat["units_sold"])
                   | (lat["unmet_demand_units"] != lat["latent_demand_units"] - lat["units_sold"])).sum(),
                  "Latent demand inconsistent with observed sales."))
    R.append(_res("unmet_implies_stockout", "Unmet demand only occurs when stock ran out",
                  ((lat["unmet_demand_units"] > 0) & (lat["closing_stock"] > 0)).sum(), "Unmet demand with stock left.",
                  severity="warning"))
    R.append(CheckResult("latent_rowcount", "error", len(lat) == len(raw["inventory_daily"]),
                         int(len(lat) != len(raw["inventory_daily"])), "Latent table covers every inventory row"))
    return R


def summarize(results: list[CheckResult]) -> dict:
    errs = [r for r in results if not r.passed and r.severity == "error"]
    warns = [r for r in results if not r.passed and r.severity == "warning"]
    return {"checks": len(results), "passed": sum(r.passed for r in results), "errors": len(errs),
            "warnings": len(warns), "ok": not errs}


def write_report(results: list[CheckResult], out_dir: Path, profile: str, context: dict | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = summarize(results)
    payload = {"profile": profile, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "disclaimer": SYNTHETIC_DISCLAIMER, "summary": s, "context": context or {},
               "checks": [asdict(r) for r in results]}
    jp = out_dir / f"data_quality_{profile}.json"
    jp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [f"# Data-quality report - profile `{profile}`", "", f"*{SYNTHETIC_DISCLAIMER}*", "",
             f"**{s['passed']}/{s['checks']} checks passed** - {s['errors']} error(s), {s['warnings']} warning(s).", "",
             "| Check | Severity | Result | Violations | Description |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.severity} | {'PASS' if r.passed else 'FAIL'} | {r.violations:,} | {r.description} |")
    fails = [r for r in results if not r.passed]
    if fails:
        lines += ["", "## Action required"] + [f"- **{r.name}**: {r.hint}" for r in fails]
    mp = out_dir / f"data_quality_{profile}.md"
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp


def assert_quality(results: list[CheckResult]) -> None:
    bad = [r for r in results if not r.passed and r.severity == "error"]
    if bad:
        msg = "\n".join(f"  - {r.name}: {r.violations:,} violation(s). {r.hint}" for r in bad)
        raise DataQualityError(f"{len(bad)} blocking data-quality check(s) failed:\n{msg}")
