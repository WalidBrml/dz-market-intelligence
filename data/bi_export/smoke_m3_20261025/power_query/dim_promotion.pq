// DZ Market Intelligence - Power BI query for the table 'dim_promotion' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: dim_promotion
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "dim_promotion.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"promotion_key", Int64.Type},
            {"promotion_name", type text},
            {"promotion_type", type text},
            {"discount_pct", type number},
            {"start_date", type date},
            {"end_date", type date}
        },
        "en-US"
    )
in
    Typed
