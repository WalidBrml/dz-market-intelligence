"""Relative, OS-independent path resolution (decision D5).

Every path is derived from this file's own location, so the project works wherever it is
cloned. Nothing here (or anywhere in the app) may contain a drive letter or a user directory.

Expected layout::

    <project root>/
        data/bi_export/smoke_m3_20261025/*.csv
        streamlit_app/dzmi_app/paths.py   <- this file
"""

from __future__ import annotations

from pathlib import Path

from .constants import DATASET_NAME

PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = APP_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "bi_export"
DATA_DIR = DATA_ROOT / DATASET_NAME


def display_path(path: Path | str) -> str:
    """Return ``path`` relative to the project root, with forward slashes.

    Used in messages so that absolute local paths never leak into the UI or logs.
    Paths outside the project root are reduced to their final component.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name
