"""Database access: PostgreSQL (target, via SQLAlchemy + psycopg) and SQLite (optional fallback).

The SQLite backend emulates the two schemas (ops, dwh) with ATTACH DATABASE so the SAME
schema-qualified SQL runs on both backends. SQLite is a development fallback only.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DatabaseError
from ..paths import REPO_ROOT, sql_dir

log = logging.getLogger(__name__)
SCHEMAS = ("ops", "dwh")


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    p = path or REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------------- SQL text helpers
def strip_comments(sql: str) -> str:
    return "\n".join(ln for ln in sql.splitlines() if not ln.strip().startswith("--"))


def split_statements(sql: str) -> list[str]:
    return [s.strip() for s in strip_comments(sql).split(";") if s.strip()]


def to_sqlite_sql(sql: str) -> str:
    """Rewrite PostgreSQL-flavoured, schema-qualified DDL for SQLite with attached schemas."""
    out = []
    for stmt in split_statements(sql):
        if re.match(r"CREATE SCHEMA", stmt, re.I):
            continue
        stmt = re.sub(r"REFERENCES\s+(?:ops|dwh)\.", "REFERENCES ", stmt, flags=re.I)
        m = re.match(r"(CREATE (?:UNIQUE )?INDEX (?:IF NOT EXISTS )?)(\w+)(\s+ON\s+)(\w+)\.(\w+)(.*)$", stmt, re.I | re.S)
        if m:
            stmt = f"{m.group(1)}{m.group(4)}.{m.group(2)}{m.group(3)}{m.group(5)}{m.group(6)}"
        if re.match(r"CREATE VIEW", stmt, re.I):
            head, body = re.split(r"\bAS\b", stmt, maxsplit=1, flags=re.I)
            stmt = head + "AS" + re.sub(r"\b(?:ops|dwh)\.", "", body)
        out.append(stmt)
    return ";\n".join(out) + ";"


def parse_check_file(sql: str) -> list[dict[str, str]]:
    """Parse '-- check: name | severity | description' blocks followed by one SELECT."""
    checks, meta = [], None
    buf: list[str] = []
    for line in sql.splitlines():
        m = re.match(r"--\s*check:\s*(.+)", line)
        if m:
            parts = [p.strip() for p in m.group(1).split("|")]
            meta = {"name": parts[0], "severity": parts[1] if len(parts) > 1 else "error",
                    "description": parts[2] if len(parts) > 2 else ""}
            buf = []
            continue
        if line.strip().startswith("--"):
            continue
        buf.append(line)
        if line.rstrip().endswith(";") and meta:
            checks.append({**meta, "sql": "\n".join(buf).strip().rstrip(";")})
            meta, buf = None, []
    return checks


# ----------------------------------------------------------------------------- backends
class Database:
    def __init__(self, backend: str, handle: Any, description: str):
        self.backend, self._h, self.description = backend, handle, description

    # --- constructors
    @classmethod
    def sqlite(cls, path: str | Path | None = None) -> "Database":
        conn = sqlite3.connect(":memory:")
        try:
            for schema in SCHEMAS:
                target = ":memory:" if path is None else str(Path(path).with_name(f"{Path(path).stem}_{schema}.sqlite"))
                if path is not None:
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                conn.execute(f"ATTACH DATABASE '{target}' AS {schema}")
            conn.execute("PRAGMA foreign_keys = ON")
        except BaseException:
            conn.close()   # never leak a half-configured connection (it would lock the files on Windows)
            raise
        desc = "SQLite (in-memory)" if path is None else f"SQLite files {Path(path).stem}_{{ops,dwh}}.sqlite"
        return cls("sqlite", conn, desc)

    @classmethod
    def postgres(cls, url: str | None = None) -> "Database":
        load_dotenv()
        url = url or os.environ.get("DATABASE_URL")
        if not url:
            raise DatabaseError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in, or use --backend sqlite "
                "for the development fallback.")
        try:
            from sqlalchemy import create_engine
            engine = create_engine(url, future=True, pool_pre_ping=True)
            try:
                with engine.connect():
                    pass
            except BaseException:
                engine.dispose()
                raise
        except ImportError as exc:
            raise DatabaseError("SQLAlchemy/psycopg not installed: run `pip install -r requirements.txt`.") from exc
        except Exception as exc:  # connection failures surface as various driver errors
            raise DatabaseError(
                f"Cannot connect to PostgreSQL ({type(exc).__name__}: {exc}). Check that the service is running, "
                "the database exists, and DATABASE_URL credentials are correct (see README troubleshooting).") from exc
        return cls("postgres", engine, "PostgreSQL")

    # --- operations
    def run_script(self, sql: str) -> None:
        try:
            if self.backend == "sqlite":
                self._h.executescript(to_sqlite_sql(sql))
            else:
                with self._h.begin() as conn:
                    for stmt in split_statements(sql):
                        conn.exec_driver_sql(stmt)
        except Exception as exc:
            raise DatabaseError(f"SQL script failed: {exc}") from exc

    def apply_schema(self) -> list[str]:
        files = sorted((sql_dir() / "schema").glob("*.sql"))
        for f in files:
            log.info("Applying %s", f.name)
            self.run_script(f.read_text(encoding="utf-8"))
        return [f.name for f in files]

    def query_df(self, sql: str) -> pd.DataFrame:
        if self.backend == "sqlite":
            return pd.read_sql_query(sql, self._h)
        from sqlalchemy import text
        with self._h.connect() as conn:
            return pd.read_sql_query(text(sql), conn)

    def iter_query_df(self, sql: str, chunksize: int = 100_000):
        """Yield query results as DataFrames, bounded by chunksize."""
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1")
        if self.backend == "sqlite":
            yield from pd.read_sql_query(sql, self._h, chunksize=chunksize)
        else:
            from sqlalchemy import text
            with self._h.connect() as conn:
                yield from pd.read_sql_query(text(sql), conn, chunksize=chunksize)
    def scalar(self, sql: str) -> Any:
        return self.query_df(sql).iloc[0, 0]

    def row_count(self, schema: str, table: str) -> int:
        return int(self.scalar(f"SELECT COUNT(*) FROM {schema}.{table}"))

    def write_df(self, df: pd.DataFrame, schema: str, table: str) -> int:
        if df.empty:
            return 0
        if self.backend == "sqlite":
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    dt = out[c]
                    date_only = (dt.dropna() == dt.dropna().dt.normalize()).all()
                    out[c] = dt.dt.strftime("%Y-%m-%d" if date_only else "%Y-%m-%d %H:%M:%S")
            obj = out.astype(object).where(out.notna(), None)
            cols = ", ".join(out.columns)
            marks = ", ".join("?" for _ in out.columns)
            cur = self._h.cursor()
            try:
                cur.executemany(f"INSERT INTO {schema}.{table} ({cols}) VALUES ({marks})",
                                (tuple(_py(v) for v in r) for r in obj.itertuples(index=False, name=None)))
                self._h.commit()
            except sqlite3.Error as exc:
                self._h.rollback()
                raise DatabaseError(f"Load into {schema}.{table} failed: {exc}") from exc
            finally:
                cur.close()
        else:
            try:
                df.to_sql(table, self._h, schema=schema, if_exists="append", index=False, chunksize=10000)
            except Exception as exc:
                raise DatabaseError(f"Load into {schema}.{table} failed: {exc}") from exc
        return len(df)

    def reset(self) -> None:
        """DESTRUCTIVE: drop the ops and dwh schemas (and everything in them)."""
        if self.backend == "postgres":
            self.run_script("DROP SCHEMA IF EXISTS dwh CASCADE;\nDROP SCHEMA IF EXISTS ops CASCADE;")
        else:
            self._h.execute("PRAGMA foreign_keys = OFF")   # tables are dropped in arbitrary order
            for schema in SCHEMAS:
                names = self.query_df(f"SELECT name, type FROM {schema}.sqlite_master "
                                      "WHERE type IN ('view','table') AND name NOT LIKE 'sqlite_%'")
                for n, t in zip(names["name"], names["type"]):
                    if t == "view":
                        self._h.execute(f"DROP VIEW IF EXISTS {schema}.{n}")
                for n, t in zip(names["name"], names["type"]):
                    if t == "table":
                        self._h.execute(f"DROP TABLE IF EXISTS {schema}.{n}")
            self._h.commit()
            self._h.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        """Release the connection/engine. Safe to call more than once."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.backend == "sqlite":
            self._h.close()   # also releases every ATTACHed file (ops, dwh)
        else:
            self._h.dispose()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _py(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bytes)):
        return v
    if hasattr(v, "item"):
        return v.item()
    return v

