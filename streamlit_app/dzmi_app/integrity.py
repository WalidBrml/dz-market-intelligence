"""Data-integrity checks run against a loaded Dataset.

Each check returns a CheckResult. Severity "error" checks gate the application (see
assert_dataset_ok); severity "info" checks record documented properties of the synthetic
data (for example, that no future forecasts are stored) and never block.

These checks validate the *exports* (keys, identities, reconciliation between tables).
They do not compute any business metric - that is the metrics layer (Milestone B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data_loader import Dataset, count_csv_rows
from .errors import DataIntegrityError
from .schema import APP_TABLES, EXPORT_ONLY_TABLES

TIER_ORDER = {
    "Stockout": 1, "Critical": 2, "Low": 3, "Healthy": 4, "Overstock": 5, "No recent sales": 6,
}
ADDITIVE = ["n_obs", "sum_actual", "sum_pred", "sum_abs_error", "sum_sq_error"]
MONEY_TOL = 1e-6


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" blocks the app, "info" is a recorded observation


def _res(check_id: str, name: str, ok: bool, detail: str, severity: str = "error") -> CheckResult:
    return CheckResult(check_id, name, bool(ok), detail, severity)


def _orphans(child: pd.Series, parent: pd.Series) -> int:
    return int((~child.dropna().isin(parent)).sum())


# --------------------------------------------------------------------------- checks


def check_manifest(ds: Dataset) -> list[CheckResult]:
    out = [
        _res("IC01", "Manifest declares the data as synthetic",
             ds.manifest.get("synthetic") is True, f"synthetic={ds.manifest.get('synthetic')!r}")
    ]
    expected = dict(ds.manifest.get("rows", {}))
    wanted = set(APP_TABLES) | set(EXPORT_ONLY_TABLES)
    problems = []
    if set(expected) != wanted:
        problems.append(f"table list differs (manifest-only={sorted(set(expected) - wanted)}, "
                        f"unlisted={sorted(wanted - set(expected))})")
    for name in sorted(set(expected) & wanted):
        actual = len(ds[name]) if name in ds.tables else count_csv_rows(name, ds.data_dir)
        if actual != expected[name]:
            problems.append(f"{name}: manifest={expected[name]} actual={actual}")
    out.append(_res("IC02", "Manifest row counts equal CSV row counts (all 22 tables)",
                    not problems, "; ".join(problems) or f"{len(expected)} tables match"))
    return out


def check_date_dimension(ds: Dataset) -> list[CheckResult]:
    d = ds["dim_date"].sort_values("full_date").reset_index(drop=True)
    unique = d["date_key"].is_unique and d["full_date"].is_unique
    contiguous = bool((d["full_date"].diff().dropna() == pd.Timedelta(days=1)).all())
    key_ok = bool(
        (d["full_date"].dt.year * 10000 + d["full_date"].dt.month * 100 + d["full_date"].dt.day
         == d["date_key"]).all()
    )
    seq_ok = bool((d["day_seq"] == np.arange(1, len(d) + 1)).all())
    last = d.loc[d["is_last_loaded_date"] == 1, "full_date"]
    last_ok = len(last) == 1 and last.iloc[0] == d["full_date"].max()
    ok = unique and contiguous and key_ok and seq_ok and last_ok
    out = [_res(
        "IC03", "dim_date: unique, contiguous days, date_key = yyyymmdd, day_seq 1..N, one last-loaded flag",
        ok, f"unique={unique} contiguous={contiguous} key={key_ok} seq={seq_ok} last_flag={last_ok}; "
            f"{d['full_date'].min():%Y-%m-%d} to {d['full_date'].max():%Y-%m-%d} ({len(d)} days)")]

    g = d.groupby("year_month").agg(
        n=("full_date", "count"), dim=("days_in_month", "first"),
        dim_n=("days_in_month", "nunique"), flag=("is_complete_month", "first"),
        flag_n=("is_complete_month", "nunique"))
    expected_flag = (g["n"] == g["dim"]).astype(int)
    month_ok = bool((g["flag"] == expected_flag).all() and (g["dim_n"] == 1).all()
                    and (g["flag_n"] == 1).all())
    detail = ", ".join(f"{ym}:{r.n}/{r.dim} flag={r.flag}" for ym, r in g.iterrows())
    out.append(_res("IC04", "dim_date.is_complete_month equals (days loaded == days in month)",
                    month_ok, detail))
    return out


def check_keys(ds: Dataset) -> list[CheckResult]:
    t = ds.tables
    specs = {
        "dim_store.store_key": t["dim_store"]["store_key"].is_unique,
        "dim_product.product_key": t["dim_product"]["product_key"].is_unique,
        "dim_product.sku": t["dim_product"]["sku"].is_unique,
        "dim_customer.customer_key": t["dim_customer"]["customer_key"].is_unique,
        "dim_promotion.promotion_key": t["dim_promotion"]["promotion_key"].is_unique,
        "fact_sales.sales_line_id": t["fact_sales"]["sales_line_id"].is_unique,
        "fact_sales(transaction_id,line_no)":
            not t["fact_sales"].duplicated(["transaction_id", "line_no"]).any(),
        "fact_inventory_daily(store,product,date)":
            not t["fact_inventory_daily"].duplicated(["store_key", "product_key", "date_key"]).any(),
        "v_inventory_risk(store,product)":
            not t["v_inventory_risk"].duplicated(["store_key", "product_key"]).any(),
        "forecast_backtest_metric grain":
            not t["forecast_backtest_metric"].duplicated(
                ["backtest_run_id", "model_name", "evaluation_basis", "level",
                 "origin_date", "horizon_bucket"]).any(),
    }
    bad = [k for k, v in specs.items() if not v]
    return [_res("IC05", "Primary keys and grains are unique", not bad,
                 "violations: " + ", ".join(bad) if bad else f"{len(specs)} key sets unique")]


def check_foreign_keys(ds: Dataset) -> list[CheckResult]:
    t, fd = ds.tables, ds.forecast_dims
    links = [
        ("fact_sales.date_key", t["fact_sales"]["date_key"], t["dim_date"]["date_key"]),
        ("fact_sales.store_key", t["fact_sales"]["store_key"], t["dim_store"]["store_key"]),
        ("fact_sales.product_key", t["fact_sales"]["product_key"], t["dim_product"]["product_key"]),
        ("fact_sales.customer_key", t["fact_sales"]["customer_key"], t["dim_customer"]["customer_key"]),
        ("fact_sales.promotion_key", t["fact_sales"]["promotion_key"], t["dim_promotion"]["promotion_key"]),
        ("fact_inventory_daily.date_key", t["fact_inventory_daily"]["date_key"], t["dim_date"]["date_key"]),
        ("fact_inventory_daily.store_key", t["fact_inventory_daily"]["store_key"], t["dim_store"]["store_key"]),
        ("fact_inventory_daily.product_key", t["fact_inventory_daily"]["product_key"], t["dim_product"]["product_key"]),
        ("fact_promotions.date_key", t["fact_promotions"]["date_key"], t["dim_date"]["date_key"]),
        ("fact_promotions.store_key", t["fact_promotions"]["store_key"], t["dim_store"]["store_key"]),
        ("fact_promotions.product_key", t["fact_promotions"]["product_key"], t["dim_product"]["product_key"]),
        ("fact_promotions.promotion_key", t["fact_promotions"]["promotion_key"], t["dim_promotion"]["promotion_key"]),
        ("v_inventory_risk.store_key", t["v_inventory_risk"]["store_key"], t["dim_store"]["store_key"]),
        ("v_inventory_risk.product_key", t["v_inventory_risk"]["product_key"], t["dim_product"]["product_key"]),
    ]
    for table in ("forecast_backtest_metric", "v_forecast_accuracy", "v_forecast_accuracy_by_horizon"):
        m = t[table]
        links += [
            (f"{table}.model_name", m["model_name"], fd.model["model_name"]),
            (f"{table}.evaluation_basis", m["evaluation_basis"], fd.basis["evaluation_basis"]),
            (f"{table}.level", m["level"], fd.level["level"]),
            (f"{table}.backtest_run_id", m["backtest_run_id"], t["backtest_run"]["backtest_run_id"]),
        ]
    for table in ("forecast_backtest_metric", "v_forecast_accuracy_by_horizon"):
        links.append((f"{table}.horizon_key", t[table]["horizon_key"], fd.horizon["horizon_key"]))
    links.append(("forecast_backtest_metric.origin_date",
                  t["forecast_backtest_metric"]["origin_date"], fd.origin["origin_date"]))
    bad = [f"{name}={n}" for name, child, parent in links if (n := _orphans(child, parent))]
    return [_res("IC06", "No orphan foreign keys (every relationship of the semantic model)",
                 not bad, "orphans: " + ", ".join(bad) if bad else f"{len(links)} relationships clean")]


def check_sales(ds: Dataset) -> list[CheckResult]:
    s = ds["fact_sales"]
    diffs = {
        "gross=qty*price": (s["gross_sales"] - s["quantity"] * s["unit_price"]).abs().max(),
        "net=gross-discount": (s["net_sales"] - (s["gross_sales"] - s["discount_amount"])).abs().max(),
        "cogs=qty*cost": (s["cogs"] - s["quantity"] * s["unit_cost"]).abs().max(),
        "gp=net-cogs": (s["gross_profit"] - (s["net_sales"] - s["cogs"])).abs().max(),
    }
    bad = {k: v for k, v in diffs.items() if v > MONEY_TOL}
    positive = bool((s["quantity"] > 0).all() and (s["net_sales"] >= 0).all()
                    and (s["discount_amount"] >= 0).all())
    out = [_res("IC07", "fact_sales arithmetic identities hold and quantities are positive",
                not bad and positive,
                f"max abs diffs={ {k: float(v) for k, v in diffs.items()} } positive={positive}")]

    g = s.groupby("transaction_id").agg(
        stores=("store_key", "nunique"), dates=("date_key", "nunique"),
        customers=("customer_key", lambda x: x.fillna(-1).nunique()))
    ok = bool((g["stores"] == 1).all() and (g["dates"] == 1).all() and (g["customers"] == 1).all())
    out.append(_res("IC08", "Each basket belongs to one store, one date and one customer (or none)",
                    ok, f"{len(g)} baskets; max stores={g['stores'].max()} dates={g['dates'].max()} "
                        f"customers={g['customers'].max()}"))

    promo = s["promotion_key"].notna()
    disc = s["discount_amount"] > 0
    out.append(_res("IC09", "Discounts occur exactly on promotional lines",
                    bool((promo == disc).all()),
                    f"promo lines={int(promo.sum())}, discounted lines={int(disc.sum())}", "info"))
    neg = s[s["gross_profit"] < 0]
    out.append(_res("IC10", "Negative-margin lines are all promotional (documented in the audit)",
                    bool(neg["promotion_key"].notna().all()),
                    f"{len(neg)} negative-margin lines, GP total={neg['gross_profit'].sum():.0f}", "info"))
    return out


def check_inventory(ds: Dataset) -> list[CheckResult]:
    inv, s = ds["fact_inventory_daily"], ds["fact_sales"]
    identity = bool(((inv["opening_stock"] + inv["received_qty"] - inv["units_sold"])
                     == inv["closing_stock"]).all())
    non_negative = bool((inv[["opening_stock", "received_qty", "units_sold", "closing_stock",
                              "on_order_qty"]] >= 0).all().all())
    flag_ok = bool((inv["stockout_flag"] == (inv["closing_stock"] == 0)).all())
    out = [_res("IC11", "Inventory identity (opening + received - sold = closing), no negatives, "
                        "stockout_flag = (closing_stock = 0)",
                identity and non_negative and flag_ok,
                f"identity={identity} non_negative={non_negative} flag={flag_ok}")]

    keys = ["store_key", "product_key", "date_key"]
    sold = s.groupby(keys, as_index=False)["quantity"].sum()
    left = sold.merge(inv[keys + ["units_sold"]], on=keys, how="left")
    no_inventory_row = int(left["units_sold"].isna().sum())
    mismatch = int((left["quantity"] != left["units_sold"]).sum())
    right = inv[keys + ["units_sold"]].merge(sold, on=keys, how="left")
    silent_sales = int(((right["quantity"].isna()) & (right["units_sold"] != 0)).sum())
    out.append(_res("IC12", "Inventory units sold reconcile with fact_sales by store x product x day",
                    no_inventory_row == mismatch == silent_sales == 0,
                    f"sales rows without inventory row={no_inventory_row}, quantity mismatches={mismatch}, "
                    f"inventory sales without a sales row={silent_sales}; "
                    f"units inventory={int(inv['units_sold'].sum())} sales={int(s['quantity'].sum())}"))
    return out


def check_inventory_risk(ds: Dataset) -> list[CheckResult]:
    vr, inv, dd = ds["v_inventory_risk"], ds["fact_inventory_daily"], ds["dim_date"]
    last_key = int(inv["date_key"].max())
    last_loaded = int(dd.loc[dd["is_last_loaded_date"] == 1, "date_key"].iloc[0])
    single = vr["snapshot_date_key"].nunique() == 1 and int(vr["snapshot_date_key"].iloc[0]) == last_key
    last = inv[inv["date_key"] == last_key][["store_key", "product_key", "closing_stock", "on_order_qty"]]
    m = vr.merge(last, on=["store_key", "product_key"], how="outer", suffixes=("", "_inv"), indicator=True)
    same_rows = bool((m["_merge"] == "both").all())
    same_values = same_rows and bool(
        (m["closing_stock"] == m["closing_stock_inv"]).all()
        and (m["on_order_qty"] == m["on_order_qty_inv"]).all())
    out = [_res("IC13", "v_inventory_risk is a single latest-day snapshot equal to fact_inventory_daily",
                single and last_key == last_loaded and same_values,
                f"snapshot={vr['snapshot_date_key'].iloc[0]} last_inventory_day={last_key} "
                f"last_loaded_flag={last_loaded} rows={len(vr)} same_rows={same_rows} same_values={same_values}")]

    unknown = sorted(set(vr["risk_tier"]) - set(TIER_ORDER))
    order_ok = bool((vr["risk_tier"].map(TIER_ORDER) == vr["risk_tier_order"]).all())
    cover = vr["days_of_cover"]
    expected = np.select(
        [vr["closing_stock"] == 0, vr["units_sold_14d"] == 0, cover < 3, cover < 7, cover > 60],
        ["Stockout", "No recent sales", "Critical", "Low", "Overstock"], default="Healthy")
    rule_ok = bool((expected == vr["risk_tier"].to_numpy()).all())
    out.append(_res("IC14", "Risk tiers are known, ordered, and follow the documented thresholds "
                            "(0 stock, no sales, <3, <7, >60 days)",
                    not unknown and order_ok and rule_ok,
                    f"unknown={unknown} order_ok={order_ok} threshold_rule_ok={rule_ok}; "
                    f"counts={vr['risk_tier'].value_counts().to_dict()}"))
    return out


def check_forecast(ds: Dataset) -> list[CheckResult]:
    t = ds.tables
    m, run = t["forecast_backtest_metric"], t["backtest_run"]
    out = []
    one_run = len(run) == 1 and set(m["backtest_run_id"]) == set(run["backtest_run_id"])
    r = run.iloc[0]
    horizon_days, n_origins = int(r["horizon_days"]), int(r["n_origins"])
    models_declared = set(str(r["models"]).split(","))
    origins_ok = m["origin_date"].nunique() == n_origins
    models_ok = set(m["model_name"]) == models_declared
    bounds = m.groupby("level")["horizon_bucket"].agg(["min", "max"])
    expected_max = {"series_day": horizon_days, "store_category_week": -(-horizon_days // 7)}
    horizon_ok = all(
        lvl in expected_max and row["min"] == 1 and row["max"] == expected_max[lvl]
        for lvl, row in bounds.iterrows())
    values_ok = bool((m["n_obs"] > 0).all() and (m[["sum_actual", "sum_pred", "sum_abs_error",
                                                    "sum_sq_error"]] >= 0).all().all())
    out.append(_res("IC15", "Backtest run metadata matches the metric table (run, origins, models, horizons)",
                    one_run and origins_ok and models_ok and horizon_ok and values_ok,
                    f"one_run={one_run} origins={m['origin_date'].nunique()}/{n_origins} "
                    f"models_ok={models_ok} horizon_ok={horizon_ok} non_negative_sums={values_ok}"))

    problems = []
    for view, extra in (("v_forecast_accuracy", []), ("v_forecast_accuracy_by_horizon", ["horizon_bucket"])):
        key = ["model_name", "evaluation_basis", "level"] + extra
        agg = m.groupby(key, as_index=False)[ADDITIVE].sum()
        agg["n_origins"] = m.groupby(key)["origin_date"].nunique().to_numpy()
        j = agg.merge(t[view], on=key, how="outer", suffixes=("", "_view"), indicator=True)
        if not (j["_merge"] == "both").all():
            problems.append(f"{view}: key sets differ")
            continue
        for col in ADDITIVE:
            if not np.allclose(j[col], j[f"{col}_view"], rtol=1e-9, atol=1e-6):
                problems.append(f"{view}.{col}")
        if not (j["n_origins"] == j["n_origins_view"]).all():
            problems.append(f"{view}.n_origins")
    out.append(_res("IC16", "SQL accuracy views equal the sums of forecast_backtest_metric",
                    not problems, "; ".join(problems) or "additive columns and n_origins reconcile"))

    n_series_ok = int(r["n_series"]) == len(t["v_inventory_risk"])
    out.append(_res("IC17", "backtest_run.n_series equals the tracked store x product pairs",
                    n_series_ok, f"n_series={int(r['n_series'])} tracked={len(t['v_inventory_risk'])}", "info"))
    empty = {n: len(t[n]) for n in ("fact_demand_forecast", "model_run")}
    out.append(_res("IC18", "No stored forecasts: fact_demand_forecast and model_run are empty "
                            "(the app produces no future forecasts)",
                    all(v == 0 for v in empty.values()), f"rows={empty}", "info"))
    return out


_CHECKS: tuple[Callable[[Dataset], list[CheckResult]], ...] = (
    check_manifest, check_date_dimension, check_keys, check_foreign_keys, check_sales,
    check_inventory, check_inventory_risk, check_forecast,
)


def run_integrity_checks(ds: Dataset) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in _CHECKS:
        results.extend(check(ds))
    return results


def failed_errors(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.severity == "error" and not r.passed]


def assert_dataset_ok(ds: Dataset) -> list[CheckResult]:
    """Run every check; raise DataIntegrityError if any blocking check fails."""
    results = run_integrity_checks(ds)
    bad = failed_errors(results)
    if bad:
        raise DataIntegrityError(
            "Dataset failed integrity checks: "
            + " | ".join(f"{r.check_id} {r.name}: {r.detail}" for r in bad))
    return results
