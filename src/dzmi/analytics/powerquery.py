"""Power Query (M) text for loading the exported CSV files into Power BI with exact column types.

Why: Power BI guesses CSV types from a sample and uses the Windows regional settings. In a locale that writes decimals
with a comma that can turn 123.45 into text or a wrong number, and integer keys can arrive as decimals. These queries
state every type and read numbers with the invariant en-US culture, matching the CSV written by `dzmi export-bi`.

The M code is generated, not executed here: it has not been run in Power BI Desktop by the author.
"""
from __future__ import annotations

from pathlib import Path

M_TYPES = {"int": "Int64.Type", "number": "type number", "text": "type text", "date": "type date",
           "logical": "type logical"}

_DATE_FLAGS = ["is_weekend", "is_ramadan", "ramadan_day", "is_eid_al_fitr", "is_eid_al_adha", "is_pre_eid_al_fitr",
               "is_pre_eid_al_adha"]

# table (as named in the Power BI model) -> {column: type}. Base tables follow sql/schema/001_schema.sql
# (tests/test_analytics_docs.py compares them with the real DDL). The two views are typed by hand.
TABLE_COLUMN_TYPES: dict[str, dict[str, str]] = {
    "dim_date": {**{c: "int" for c in ["date_key", "year", "quarter", "month", "day_of_month", "day_of_week", "iso_week",
                                       "year_month", "days_in_month", "is_complete_month", "prior_year_date_key",
                                       "day_seq", "is_last_loaded_date"] + _DATE_FLAGS},
                 "full_date": "date", "month_name": "text", "day_name": "text", "year_month_label": "text",
                 "year_quarter_label": "text"},
    "dim_store": {"store_key": "int", "store_id": "int", "store_code": "text", "store_name": "text", "wilaya": "text",
                  "city": "text", "store_format": "text", "floor_area_sqm": "int", "opened_date": "date"},
    "dim_product": {"product_key": "int", "product_id": "int", "sku": "text", "product_name": "text",
                    "category_name": "text", "is_perishable": "logical", "brand": "text", "pack_size": "int",
                    "current_list_price": "number", "current_unit_cost": "number"},
    "dim_customer": {"customer_key": "int", "customer_id": "int", "wilaya": "text", "household_type": "text",
                     "signup_date": "date"},
    "dim_promotion": {"promotion_key": "int", "promotion_name": "text", "promotion_type": "text",
                      "discount_pct": "number", "start_date": "date", "end_date": "date"},
    "fact_sales": {"sales_line_id": "int", "transaction_id": "int", "line_no": "int", "date_key": "int",
                   "store_key": "int", "product_key": "int", "customer_key": "int", "promotion_key": "int",
                   "quantity": "int", "unit_price": "number", "unit_cost": "number", "gross_sales": "number",
                   "discount_amount": "number", "net_sales": "number", "cogs": "number", "gross_profit": "number"},
    "fact_inventory_daily": {"store_key": "int", "product_key": "int", "date_key": "int", "opening_stock": "int",
                             "received_qty": "int", "units_sold": "int", "closing_stock": "int", "on_order_qty": "int",
                             "stockout_flag": "logical"},
    "fact_promotions": {"store_key": "int", "product_key": "int", "date_key": "int", "promotion_key": "int",
                        "discount_pct": "number"},
    "v_inventory_risk": {"store_key": "int", "store_code": "text", "store_name": "text", "wilaya": "text",
                         "product_key": "int", "sku": "text", "product_name": "text", "category_name": "text",
                         "is_perishable": "int", "snapshot_date_key": "int", "snapshot_date": "date",
                         "closing_stock": "int", "on_order_qty": "int", "units_sold_14d": "int",
                         "days_observed_14d": "int", "avg_daily_units_14d": "number", "days_of_cover": "number",
                         "days_of_cover_incl_on_order": "number", "stockout_days_14d": "int",
                         "stock_value_at_current_cost": "number", "risk_tier": "text", "risk_tier_order": "int"},
}

# Power BI table -> CSV file written by export_bi (identical names) and database object for the PostgreSQL connector.
POSTGRES_ITEM = {"dim_date": "v_dim_date_bi"}


def render_csv_query(table: str, data_folder: str) -> str:
    """M query that loads `<data_folder><table>.csv` with explicit types. Name the Power BI query `table`."""
    types = ",\n            ".join(f'{{"{c}", {M_TYPES[t]}}}' for c, t in TABLE_COLUMN_TYPES[table].items())
    folder = data_folder if data_folder.endswith(("\\", "/")) else data_folder + "\\"
    return (f"// DZ Market Intelligence - Power BI query for the table '{table}' (SYNTHETIC DATA).\n"
            f"// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: {table}\n"
            f"let\n"
            f'    DataFolder = "{folder}",\n'
            f'    Source = Csv.Document(File.Contents(DataFolder & "{table}.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),\n'
            f"    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n"
            f"    Typed = Table.TransformColumnTypes(\n"
            f"        Promoted,\n"
            f"        {{\n"
            f"            {types}\n"
            f"        }},\n"
            f'        "en-US"\n'
            f"    )\n"
            f"in\n"
            f"    Typed\n")


def render_postgres_query(table: str, host: str = "localhost:5432", database: str = "dzmi") -> str:
    item = POSTGRES_ITEM.get(table, table)
    return (f"// DZ Market Intelligence - Power BI query for the table '{table}' from PostgreSQL (SYNTHETIC DATA).\n"
            f"let\n"
            f'    Source = PostgreSQL.Database("{host}", "{database}"),\n'
            f'    Data = Source{{[Schema = "dwh", Item = "{item}"]}}[Data]\n'
            f"in\n"
            f"    Data\n")


def write_power_query_files(out_dir: Path, data_folder: str) -> list[Path]:
    """Write one .pq file per Power BI table into out_dir (the CSV flavour)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in TABLE_COLUMN_TYPES:
        p = out_dir / f"{table}.pq"
        p.write_text(render_csv_query(table, data_folder), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths
