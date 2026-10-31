// DZ Market Intelligence - Power BI query for the table 'dim_date' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: dim_date
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "dim_date.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"date_key", Int64.Type},
            {"year", Int64.Type},
            {"quarter", Int64.Type},
            {"month", Int64.Type},
            {"day_of_month", Int64.Type},
            {"day_of_week", Int64.Type},
            {"iso_week", Int64.Type},
            {"year_month", Int64.Type},
            {"days_in_month", Int64.Type},
            {"is_complete_month", Int64.Type},
            {"prior_year_date_key", Int64.Type},
            {"day_seq", Int64.Type},
            {"is_last_loaded_date", Int64.Type},
            {"is_weekend", Int64.Type},
            {"is_ramadan", Int64.Type},
            {"ramadan_day", Int64.Type},
            {"is_eid_al_fitr", Int64.Type},
            {"is_eid_al_adha", Int64.Type},
            {"is_pre_eid_al_fitr", Int64.Type},
            {"is_pre_eid_al_adha", Int64.Type},
            {"full_date", type date},
            {"month_name", type text},
            {"day_name", type text},
            {"year_month_label", type text},
            {"year_quarter_label", type text}
        },
        "en-US"
    )
in
    Typed
