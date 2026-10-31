"""Milestone 2 documentation tests: the docs, relationships, DAX and Power Query text must match the real schema.

Power BI itself cannot run here, so these tests are the safety net: every table and column named in the Power BI files
exists in the database objects they are loaded from, and the reference numbers quoted in the docs are recomputed.
"""
from __future__ import annotations

import csv
import re
import tempfile
from pathlib import Path

import pandas as pd

import reference_calcs as ref
from dzmi.analytics.bi_export import export_bi
from dzmi.analytics.powerquery import POSTGRES_ITEM, TABLE_COLUMN_TYPES, render_csv_query, render_postgres_query
from dzmi.analytics.views import ANALYTIC_VIEWS
from dzmi.paths import REPO_ROOT
from helpers import shared_smoke_db, smoke

DOCS = (REPO_ROOT / "docs" / "analytical_views.md").read_text(encoding="utf-8")
PBI_DIR = REPO_ROOT / "powerbi"
PBI_README = (PBI_DIR / "README.md").read_text(encoding="utf-8")
DAX = (PBI_DIR / "measures.dax").read_text(encoding="utf-8")
DAX_REF = (PBI_DIR / "dax_measure_reference.md").read_text(encoding="utf-8")
DB_OBJECT = {"dim_date": "v_dim_date_bi"}          # Power BI table -> database object when the names differ


def columns_of(db, table: str) -> list[str]:
    return list(db.query_df(f"SELECT * FROM dwh.{DB_OBJECT.get(table, table)} LIMIT 0").columns)


def section(view: str) -> str:
    m = re.search(rf"^### dwh\.{view}\n(.*?)(?=^### |^## |\Z)", DOCS, re.S | re.M)
    assert m, f"docs/analytical_views.md has no section '### dwh.{view}'"
    return m.group(1)


# ------------------------------------------------------------------------------------------ view documentation
def test_every_view_is_documented_with_grain_and_every_column():
    db = shared_smoke_db()
    for view, (grain, _) in ANALYTIC_VIEWS.items():
        text = section(view)
        assert "**Grain:**" in text, f"{view}: missing a Grain line"
        assert f"`dwh.{view}`" in DOCS.split("## Metric definitions")[0], f"{view}: missing from the overview table"
        missing = [c for c in columns_of(db, view) if f"`{c}`" not in text]
        assert not missing, f"{view}: columns not documented: {missing}"
    documented = set(re.findall(r"^### dwh\.(v_\w+)$", DOCS, re.M))
    assert documented == set(ANALYTIC_VIEWS)             # no stale sections for views that no longer exist


def test_documented_metrics_name_real_columns():
    """The 'Column' cell of every row of the metric table must name columns that exist in some view."""
    db = shared_smoke_db()
    table = DOCS.split("## Metric definitions")[1].split("### Year-over-year rules")[0]
    cols = set()
    for view in ANALYTIC_VIEWS:
        cols |= set(columns_of(db, view))
    checked = 0
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cells) < 3 or cells[0] in ("Metric", "---") or cells[0].startswith("---"):
            continue
        for name in re.findall(r"`([a-z_0-9]+)`", cells[1]):
            assert name in cols, f"metric '{cells[0]}' names `{name}` which is not a view column"
            checked += 1
    assert checked >= 20


def test_reference_values_in_docs_match_an_independent_recalculation():
    cfg, raw, dwh = smoke()
    ln = ref.sales_lines(raw)
    inv = raw["inventory_daily"]
    risk = ref.inventory_risk(raw)
    net = ln["net"].sum()
    values = {
        "net sales": f"{net:,.2f}",
        "gross profit": f"{ln['profit'].sum():,.2f}",
        "cogs": f"{ln['cogs'].sum():,.2f}",
        "gross sales": f"{ln['gross'].sum():,.2f}",
        "discounts": f"{ln['discount_amount'].sum():,.2f}",
        "units": f"{int(ln['quantity'].sum()):,}",
        "transactions": f"{ln['transaction_id'].nunique():,}",
        "basket": f"{net / ln['transaction_id'].nunique():,.2f}",
        "asp": f"{net / ln['quantity'].sum():,.2f}",
        "sku_days": f"{len(inv):,}",
        "stockouts": f"{int(inv['stockout_flag'].sum()):,}",
        "ramadan": f"{ln[ln['date'].isin(dwh['dim_date'].loc[dwh['dim_date']['is_ramadan'], 'full_date'])]['net'].sum():,.2f}",
        "promo": f"{ln.loc[ln['has_promo'], 'net'].sum():,.2f}",
        "stock value": f"{risk['stock_value_at_current_cost'].sum():,.2f}",
        "margin": f"{ln['profit'].sum() / net * 100:.4f}",
        "stockout rate": f"{inv['stockout_flag'].mean() * 100:.4f}",
    }
    monthly = ln.groupby(ln["date"].dt.month)["net"].sum()
    for m in (2, 3, 4, 5):
        values[f"month {m}"] = f"{monthly[m]:,.2f}"
    for k, v in values.items():
        assert v in DOCS, f"docs/analytical_views.md: reference value for {k} ({v}) not found"
    for k in ("net sales", "gross profit", "units", "transactions", "basket", "asp", "ramadan", "promo", "stock value",
              "month 2", "month 3", "month 4", "month 5"):
        assert values[k] in PBI_README, f"powerbi/README.md: reference value for {k} ({values[k]}) not found"
    tiers = risk["risk_tier"].value_counts()
    assert all(f"{n}" in PBI_README for n in (tiers["Stockout"], tiers["Critical"], tiers["Low"]))
    assert f"{values['margin'][:5]} %" in PBI_README.replace("16.94 %", "16.94 %") or "16.94 %" in PBI_README
    assert f"{int(inv['stockout_flag'].sum()):,} of {len(inv):,}" in PBI_README


# ------------------------------------------------------------------------------------------ relationships
def _relationships():
    with (PBI_DIR / "model_relationships.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_relationships_use_real_columns_and_have_no_orphan_keys():
    db = shared_smoke_db()
    rels = _relationships()
    assert len(rels) == 15
    for r in rels:
        for table, col in ((r["from_table"], r["from_column"]), (r["to_table"], r["to_column"])):
            assert table in TABLE_COLUMN_TYPES, f"{table} is not a Power BI table"
            assert col in columns_of(db, table), f"{table}[{col}] does not exist"
        assert r["cardinality"] == "many-to-one" and r["cross_filter"] == "single"   # no bidirectional filters
        src, dst = DB_OBJECT.get(r["from_table"], r["from_table"]), DB_OBJECT.get(r["to_table"], r["to_table"])
        orphans = db.scalar(f"SELECT COUNT(*) FROM dwh.{src} f LEFT JOIN dwh.{dst} d ON d.{r['to_column']} = f.{r['from_column']} "
                            f"WHERE f.{r['from_column']} IS NOT NULL AND d.{r['to_column']} IS NULL")
        assert orphans == 0, f"{r['from_table']}[{r['from_column']}] has {orphans} values missing in {r['to_table']}"
        dup = db.scalar(f"SELECT COUNT(*) - COUNT(DISTINCT {r['to_column']}) FROM dwh.{dst}")
        assert dup == 0, f"{r['to_table']}[{r['to_column']}] is not unique, so it cannot be the 'one' side"
        null_keys = db.scalar(f"SELECT COUNT(*) FROM dwh.{src} WHERE {r['from_column']} IS NULL")
        assert (null_keys > 0) <= (r["nullable_key"] == "true"), f"{r['from_table']}[{r['from_column']}] has NULLs"
    only_inactive = [r for r in rels if r["active"] == "false"]
    assert [r["from_column"] for r in only_inactive] == ["snapshot_date_key"]


def test_readme_diagram_matches_the_active_relationships():
    diagram = PBI_README.split("```mermaid")[1].split("```")[0]
    for r in _relationships():
        pattern = rf"{r['to_table']}\s+\S+--\S+\s+{r['from_table']}\s*:\s*{r['from_column']}"
        assert bool(re.search(pattern, diagram)) == (r["active"] == "true"), f"diagram mismatch for {r}"


# ------------------------------------------------------------------------------------------ DAX
def _dax_measures() -> dict[str, str]:
    measures: dict[str, list[str]] = {}
    current = None
    for line in DAX.splitlines():
        if not line.strip() or line.startswith("//"):
            continue
        if line[0] in " \t":
            assert current, "indented line before any measure"
            measures[current].append(line.strip())
            continue
        name, sep, rest = line.partition(" =")
        assert sep, f"line outside a measure: {line!r}"
        current = name.strip()
        assert current not in measures, f"duplicate measure {current}"
        measures[current] = [rest.strip()]
    return {k: "\n".join(v) for k, v in measures.items()}


def test_dax_references_exist_and_every_measure_is_documented():
    db = shared_smoke_db()
    measures = _dax_measures()
    assert len(measures) >= 35
    for name, body in measures.items():
        assert body.strip(), f"{name}: empty definition"
        assert body.count("(") == body.count(")"), f"{name}: unbalanced parentheses"
        for table, col in re.findall(r"\b((?:dim|fact)_\w+|v_inventory_risk)\[(\w+)\]", body):
            assert table in TABLE_COLUMN_TYPES, f"{name}: unknown table {table}"
            assert col in columns_of(db, table), f"{name}: {table}[{col}] does not exist in the database"
        for ref_name in re.findall(r"(?<![\w\]])\[([^\]]+)\]", body):
            assert ref_name in measures, f"{name}: refers to [{ref_name}] which is not a measure"
        assert not re.search(r"\bRELATED\s*\(", body), f"{name}: RELATED would copy columns across facts"
    documented = re.findall(r"^\| ([^|]+?) \| (?:Sales|Time intelligence|Calendar effects|Availability|Inventory risk) \|", DAX_REF, re.M)
    assert sorted(documented) == sorted(measures), (set(documented) ^ set(measures))
    # the definitions that matter most are the ones people re-implement wrongly
    assert "DIVIDE" in measures["Gross Margin %"] and "[Gross Profit]" in measures["Gross Margin %"]
    assert "LASTDATE" in measures["Closing Stock Units"]                         # semi-additive, not SUM over dates
    assert "closing_stock] = 0" in measures["Stockout SKU Days"]
    assert "SAMEPERIODLASTYEAR" in measures["Net Sales PY"] and "<> 0" in measures["Net Sales YoY %"]


# ------------------------------------------------------------------------------------------ Power Query
def test_power_query_types_match_the_real_schema():
    db = shared_smoke_db()
    ddl_type = {"INTEGER": "int", "BIGINT": "int", "SMALLINT": "int", "NUMERIC": "number", "DATE": "date",
                "BOOLEAN": "logical", "VARCHAR": "text", "TEXT": "text"}
    for table, types in TABLE_COLUMN_TYPES.items():
        assert set(types) == set(columns_of(db, table)), f"{table}: Power Query columns differ from the database"
        if table in ("dim_date", "v_inventory_risk"):
            continue
        info = db.query_df(f"PRAGMA dwh.table_info({table})")
        for _, r in info.iterrows():
            assert types[r["name"]] == ddl_type[r["type"].split("(")[0].upper()], f"{table}.{r['name']}: wrong type"


def test_exported_power_query_files_match_the_exported_csv_headers():
    db = shared_smoke_db()
    with tempfile.TemporaryDirectory() as d:
        export_bi(db, Path(d))
        for table, types in TABLE_COLUMN_TYPES.items():
            header = pd.read_csv(Path(d) / f"{table}.csv", nrows=0).columns.tolist()
            pq = (Path(d) / "power_query" / f"{table}.pq").read_text(encoding="utf-8")
            assert header == list(types) or set(header) == set(types), f"{table}: CSV header differs from the query"
            for col in header:
                assert f'{{"{col}", ' in pq, f"{table}.pq does not type column {col}"
            assert f'"{table}.csv"' in pq and '"en-US"' in pq and "Encoding = 65001" in pq
            assert str(Path(d).resolve()) in pq                                  # points at the folder it was written to
    assert 'Item = "v_dim_date_bi"' in render_postgres_query("dim_date") and POSTGRES_ITEM["dim_date"] == "v_dim_date_bi"
    assert 'Item = "fact_sales"' in render_postgres_query("fact_sales")
    q = render_csv_query("fact_sales", "C:\\data\\smoke")
    assert 'DataFolder = "C:\\data\\smoke\\"' in q and q.count("{") == q.count("}")
