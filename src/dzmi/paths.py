"""Repository paths. Works from any working directory."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.environ.get("DZMI_CONFIG_DIR", REPO_ROOT / "config"))


def sql_dir() -> Path:
    return REPO_ROOT / "sql"


def raw_dir(profile: str) -> Path:
    return REPO_ROOT / "data" / "raw" / profile


def warehouse_dir(profile: str) -> Path:
    return REPO_ROOT / "data" / "warehouse" / profile


def reports_dir() -> Path:
    return REPO_ROOT / "reports"


def default_sqlite_path(profile: str) -> Path:
    return REPO_ROOT / "data" / f"dzmi_{profile}.sqlite"
def bi_export_dir(profile: str) -> Path:
    return REPO_ROOT / "data" / "bi_export" / profile

