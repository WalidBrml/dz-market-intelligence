// DZ Market Intelligence - Power BI query for the table 'v_inventory_risk' (SYNTHETIC DATA).
// Home > Transform data > New Source > Blank Query > Advanced Editor: paste this, then rename the query to: v_inventory_risk
let
    DataFolder = "C:\Users\DELL\dz-market-intelligence\dz-market-intelligence\data\bi_export\smoke_m3_20261025\",
    Source = Csv.Document(File.Contents(DataFolder & "v_inventory_risk.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    Typed = Table.TransformColumnTypes(
        Promoted,
        {
            {"store_key", Int64.Type},
            {"store_code", type text},
            {"store_name", type text},
            {"wilaya", type text},
            {"product_key", Int64.Type},
            {"sku", type text},
            {"product_name", type text},
            {"category_name", type text},
            {"is_perishable", Int64.Type},
            {"snapshot_date_key", Int64.Type},
            {"snapshot_date", type date},
            {"closing_stock", Int64.Type},
            {"on_order_qty", Int64.Type},
            {"units_sold_14d", Int64.Type},
            {"days_observed_14d", Int64.Type},
            {"avg_daily_units_14d", type number},
            {"days_of_cover", type number},
            {"days_of_cover_incl_on_order", type number},
            {"stockout_days_14d", Int64.Type},
            {"stock_value_at_current_cost", type number},
            {"risk_tier", type text},
            {"risk_tier_order", Int64.Type}
        },
        "en-US"
    )
in
    Typed
