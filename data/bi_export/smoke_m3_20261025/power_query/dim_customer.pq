// DZ Market Intelligence - Power BI query for the table 'dim_customer' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: dim_customer
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "dim_customer.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"customer_key", Int64.Type},
            {"customer_id", Int64.Type},
            {"wilaya", type text},
            {"household_type", type text},
            {"signup_date", type date}
        },
        "en-US"
    )
in
    Typed
