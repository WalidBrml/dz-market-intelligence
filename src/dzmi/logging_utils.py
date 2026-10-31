"""Logging setup shared by the CLI and scripts."""
from __future__ import annotations

import logging
import os


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("DZMI_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
