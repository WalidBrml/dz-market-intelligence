// DZ Market Intelligence - Power BI query for the table 'fact_sales' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: fact_sales
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "fact_sales.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"sales_line_id", Int64.Type},
            {"transaction_id", Int64.Type},
            {"line_no", Int64.Type},
            {"date_key", Int64.Type},
            {"store_key", Int64.Type},
            {"product_key", Int64.Type},
            {"customer_key", Int64.Type},
            {"promotion_key", Int64.Type},
            {"quantity", Int64.Type},
            {"unit_price", type number},
            {"unit_cost", type number},
            {"gross_sales", type number},
            {"discount_amount", type number},
            {"net_sales", type number},
            {"cogs", type number},
            {"gross_profit", type number}
        },
        "en-US"
    )
in
    Typed
