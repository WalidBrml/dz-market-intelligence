// DZ Market Intelligence - Power BI query for the table 'fact_promotions' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: fact_promotions
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "fact_promotions.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"store_key", Int64.Type},
            {"product_key", Int64.Type},
            {"date_key", Int64.Type},
            {"promotion_key", Int64.Type},
            {"discount_pct", type number}
        },
        "en-US"
    )
in
    Typed
