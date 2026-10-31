"""CSV persistence for raw (ops) tables, with a manifest."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import GENERATOR_VERSION, SYNTHETIC_DISCLAIMER
from ..errors import DZMIError

DATE_COLS = {
    "product_price_history": ["valid_from"], "store": ["opened_date"], "customer": ["signup_date"],
    "promotion": ["start_date", "end_date"], "sales_transaction": ["transaction_ts", "transaction_date"],
    "inventory_daily": ["inventory_date"], "sim_latent_demand": ["demand_date"],
}
NULLABLE_INT = {"sales_transaction": ["customer_id"], "sales_transaction_line": ["promotion_id"]}
TABLE_NAMES = ["data_provenance", "category", "product", "product_price_history", "store", "store_assortment",
               "customer", "promotion", "promotion_scope", "sales_transaction", "sales_transaction_line",
               "inventory_daily", "sim_latent_demand"]


def write_raw(raw: dict[str, pd.DataFrame], out_dir: Path, cfg=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in raw.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    manifest = {
        "synthetic": True, "disclaimer": SYNTHETIC_DISCLAIMER, "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": getattr(cfg, "profile", None), "seed": getattr(cfg, "seed", None),
        "config_fingerprint": cfg.fingerprint() if cfg else None,
        "rows": {k: int(len(v)) for k, v in raw.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def read_raw(in_dir: Path) -> dict[str, pd.DataFrame]:
    if not (in_dir / "manifest.json").exists():
        raise DZMIError(f"No generated data in {in_dir}. Run: dzmi generate --profile <profile>")
    out = {}
    for name in TABLE_NAMES:
        path = in_dir / f"{name}.csv"
        if not path.exists():
            raise DZMIError(f"Missing {path.name} in {in_dir}; regenerate the dataset.")
        dtypes = {c: "Int64" for c in NULLABLE_INT.get(name, [])}
        out[name] = pd.read_csv(path, parse_dates=DATE_COLS.get(name, []), dtype=dtypes)
    return out
