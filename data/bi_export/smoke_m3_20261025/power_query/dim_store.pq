// DZ Market Intelligence - Power BI query for the table 'dim_store' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: dim_store
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "dim_store.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"store_key", Int64.Type},
            {"store_id", Int64.Type},
            {"store_code", type text},
            {"store_name", type text},
            {"wilaya", type text},
            {"city", type text},
            {"store_format", type text},
            {"floor_area_sqm", Int64.Type},
            {"opened_date", type date}
        },
        "en-US"
    )
in
    Typed
