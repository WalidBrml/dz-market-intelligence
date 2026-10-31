"""Run sql/validation/*.sql against the database and summarise the results."""
from __future__ import annotations

from ..errors import DatabaseError
from ..paths import sql_dir
from .db import Database, parse_check_file
from .quality import CheckResult


def run_db_checks(db: Database) -> list[CheckResult]:
    results: list[CheckResult] = []
    for f in sorted((sql_dir() / "validation").glob("*.sql")):
        checks = parse_check_file(f.read_text(encoding="utf-8"))
        if not checks:
            raise DatabaseError(f"No checks parsed from {f.name}.")
        for c in checks:
            try:
                n = int(db.scalar(c["sql"]))
            except Exception as exc:
                hint = ""
                if c["name"].startswith(("an_", "fc_")):   # Milestone 2 / 3 checks need the Milestone 2 / 3 objects
                    hint = (" If this database was created before Milestone 2 or 3, add the new views and tables (no "
                            "data is lost) with: dzmi init-db --backend <postgres|sqlite> [--profile <profile>]")
                raise DatabaseError(f"Validation query '{c['name']}' failed: {exc}.{hint}") from exc
            results.append(CheckResult(f"db_{c['name']}", c["severity"], n == 0, n, c["description"],
                                       f"See sql/validation/{f.name} -> {c['name']}"))
    return results
