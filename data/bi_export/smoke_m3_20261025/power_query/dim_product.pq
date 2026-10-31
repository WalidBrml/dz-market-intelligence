// DZ Market Intelligence - Power BI query for the table 'dim_product' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: dim_product
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "dim_product.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"product_key", Int64.Type},
            {"product_id", Int64.Type},
            {"sku", type text},
            {"product_name", type text},
            {"category_name", type text},
            {"is_perishable", type logical},
            {"brand", type text},
            {"pack_size", Int64.Type},
            {"current_list_price", type number},
            {"current_unit_cost", type number}
        },
        "en-US"
    )
in
    Typed
