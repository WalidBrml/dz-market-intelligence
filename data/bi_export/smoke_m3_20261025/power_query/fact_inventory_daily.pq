// DZ Market Intelligence - Power BI query for the table 'fact_inventory_daily' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: fact_inventory_daily
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "fact_inventory_daily.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"store_key", Int64.Type},
            {"product_key", Int64.Type},
            {"date_key", Int64.Type},
            {"opening_stock", Int64.Type},
            {"received_qty", Int64.Type},
            {"units_sold", Int64.Type},
            {"closing_stock", Int64.Type},
            {"on_order_qty", Int64.Type},
            {"stockout_flag", type logical}
        },
        "en-US"
    )
in
    Typed
