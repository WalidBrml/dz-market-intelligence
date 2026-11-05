import pandas as pd
import pytest

from dzmi_app import data_loader
from dzmi_app.data_loader import clear_cache, load_dataset, read_table
from dzmi_app.errors import DataIntegrityError
from dzmi_app.schema import APP_TABLES, SCHEMAS
from oracle_values import MANIFEST_ROWS


def test_every_app_table_is_loaded(dataset):
    assert set(dataset.tables) == set(APP_TABLES)
    assert len(APP_TABLES) == 15


@pytest.mark.parametrize("table", list(APP_TABLES))
def test_columns_match_the_schema_exactly(dataset, table):
    expected = [c.name for c in SCHEMAS[table]]
    derived = {"dim_date": ["calendar_period", "week_part"], "dim_product": ["perishability"],
               "forecast_backtest_metric": ["horizon_key"],
               "v_forecast_accuracy_by_horizon": ["horizon_key"]}.get(table, [])
    assert list(dataset[table].columns) == expected + derived


@pytest.mark.parametrize("table", list(APP_TABLES))
def test_dtypes_follow_the_declared_kinds(dataset, table):
    df = dataset[table]
    for col in SCHEMAS[table]:
        s = df[col.name]
        if col.kind == "int":
            assert str(s.dtype) == "int64", (table, col.name, s.dtype)
        elif col.kind == "Int":
            assert str(s.dtype) == "Int64", (table, col.name, s.dtype)
        elif col.kind == "float":
            assert str(s.dtype) == "float64", (table, col.name, s.dtype)
        elif col.kind == "bool":
            assert str(s.dtype) == "bool", (table, col.name, s.dtype)
        elif col.kind in ("date", "datetime"):
            assert pd.api.types.is_datetime64_any_dtype(s), (table, col.name, s.dtype)
        else:
            assert pd.api.types.is_string_dtype(s), (table, col.name, s.dtype)


def test_row_counts_equal_the_manifest(dataset):
    for name, n in dataset.row_counts().items():
        assert n == MANIFEST_ROWS[name], name


def test_second_call_returns_the_cached_object_and_clear_cache_resets_it():
    clear_cache()
    first = load_dataset()
    assert load_dataset() is first
    assert load_dataset(None) is first
    clear_cache()
    second = load_dataset()
    assert second is not first
    assert second.row_counts() == first.row_counts()


def test_tables_mapping_is_read_only(dataset):
    with pytest.raises(TypeError):
        dataset.tables["new_table"] = pd.DataFrame()


def test_accented_text_is_read_as_utf8(dataset):
    assert "S\u00e9tif El Hidhab" in set(dataset["dim_store"]["store_name"])
    assert "S\u00e9tif" in set(dataset["dim_customer"]["wilaya"])


def test_empty_forecast_tables_keep_their_schema(dataset):
    for name in ("fact_demand_forecast", "model_run"):
        assert len(dataset[name]) == 0
        assert list(dataset[name].columns) == [c.name for c in SCHEMAS[name]]


def test_nullable_integer_keys_use_pandas_na(dataset):
    sales = dataset["fact_sales"]
    assert sales["customer_key"].isna().sum() == 31262
    assert sales["promotion_key"].isna().sum() == 45288
    assert dataset["dim_date"]["ramadan_day"].isna().sum() == 61


def test_power_query_derived_columns(dataset):
    dd = dataset["dim_date"].set_index("full_date")
    assert dd.loc[pd.Timestamp("2025-02-01"), "calendar_period"] == "Regular days"
    assert dd.loc[pd.Timestamp("2025-03-01"), "calendar_period"] == "Ramadan"
    assert dd.loc[pd.Timestamp("2025-03-29"), "calendar_period"] == "Ramadan"
    assert dd.loc[pd.Timestamp("2025-03-30"), "calendar_period"] == "Eid al-Fitr"
    assert dd.loc[pd.Timestamp("2025-03-31"), "calendar_period"] == "Eid al-Fitr"
    assert dd.loc[pd.Timestamp("2025-04-01"), "calendar_period"] == "Regular days"
    assert set(dd["calendar_period"]) == {"Regular days", "Ramadan", "Eid al-Fitr"}
    # Weekend = Friday and Saturday
    weekend_days = set(dd.loc[dd["week_part"] == "Weekend (Fri-Sat)", "day_name"])
    assert weekend_days == {"Friday", "Saturday"}
    assert set(dd.loc[dd["week_part"] == "Weekday", "day_name"]) == {
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"}
    prod = dataset["dim_product"]
    assert ((prod["perishability"] == "Perishable") == prod["is_perishable"]).all()
    assert set(prod["perishability"]) == {"Perishable", "Non-perishable"}
    for table in ("forecast_backtest_metric", "v_forecast_accuracy_by_horizon"):
        df = dataset[table]
        assert (df["horizon_key"] == df["level"] + "|" + df["horizon_bucket"].astype(str)).all()


def test_text_na_is_not_treated_as_missing(tmp_path):
    src = "product_key,product_id,sku,product_name,category_name,is_perishable,brand,pack_size,current_list_price,current_unit_cost\n"
    (tmp_path / "dim_product.csv").write_text(
        src + "1,1,NA-1,NA,None,False,Brand A,12,10,8\n", encoding="utf-8")
    df = read_table("dim_product", tmp_path)
    assert df.loc[0, "product_name"] == "NA" and df.loc[0, "category_name"] == "None"


# ------------------------------------------------------------------ failure modes


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataIntegrityError, match="Missing data file"):
        read_table("dim_store", tmp_path)


def test_unknown_table_raises(tmp_path):
    with pytest.raises(DataIntegrityError, match="Unknown table"):
        read_table("not_a_table", tmp_path)


def test_wrong_header_raises(tmp_path):
    (tmp_path / "dim_promotion.csv").write_text(
        "promotion_key,promotion_name,promotion_type,discount_pct,start_date\n1,A,B,10,2025-01-01\n",
        encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="header differs"):
        read_table("dim_promotion", tmp_path)


def test_non_numeric_value_in_integer_column_raises(tmp_path):
    (tmp_path / "dim_promotion.csv").write_text(
        "promotion_key,promotion_name,promotion_type,discount_pct,start_date,end_date\n"
        "1,A,B,ten,2025-01-01,2025-01-02\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="cannot parse"):
        read_table("dim_promotion", tmp_path)


def test_unexpected_null_in_required_column_raises(tmp_path):
    (tmp_path / "dim_promotion.csv").write_text(
        "promotion_key,promotion_name,promotion_type,discount_pct,start_date,end_date\n"
        "1,,B,10,2025-01-01,2025-01-02\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="unexpected null"):
        read_table("dim_promotion", tmp_path)


def test_bad_date_raises(tmp_path):
    (tmp_path / "dim_promotion.csv").write_text(
        "promotion_key,promotion_name,promotion_type,discount_pct,start_date,end_date\n"
        "1,A,B,10,01/02/2025,2025-01-02\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="cannot parse"):
        read_table("dim_promotion", tmp_path)


def test_missing_data_folder_raises(tmp_path):
    with pytest.raises(DataIntegrityError, match="Data folder not found"):
        load_dataset(tmp_path / "nope")


def test_error_messages_do_not_leak_absolute_paths(tmp_path):
    with pytest.raises(DataIntegrityError) as err:
        read_table("dim_store", tmp_path)
    assert str(tmp_path) not in str(err.value)
