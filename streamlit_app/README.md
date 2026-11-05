# DZ Market Intelligence - Streamlit companion (read-only)

Read-only Streamlit re-implementation of the Power BI report, built on the **synthetic**
`smoke_m3_20261025` export. Nothing here describes real Algerian retail demand.

**Status: Milestones B1-B6** - all six pages of the Power BI report are implemented, in the same order.
No future forecasts are produced: the forecast tables of this dataset are empty.

## Layout (inside the project root)

    data/bi_export/smoke_m3_20261025/   CSV exports (already in the repository)
    streamlit_app/
      app.py                            entry point: integrity gate + st.navigation (six pages)
      pages/p1_executive_overview.py    Page 1 (filters, KPI cards, charts)
      dzmi_app/
        paths, constants, schema, errors, data_loader   data layer (relative paths, cached CSV load)
        dims_forecast, integrity                        forecast dimensions, 18 integrity checks
        filters, overview                               Page 1 filter objects and result computation
        measures_sales, measures_inventory              ported DAX measures (pure pandas)
        formatting, theme, ui_html, charts, components, state   presentation layer
      tests/                            pytest suite + independent SQL oracle for Page 1

## Commands (run from the project root)

    python -m pip install -r streamlit_app/requirements.txt
    python -m pytest streamlit_app -v
    streamlit run streamlit_app/app.py

## Executive Overview: where it differs from the Power BI page (intentional)

| ID | Difference | Reason |
|----|------------|--------|
| V1 | Every filter starts at All; the saved Power BI slicer selections are not reproduced | Decision D1 |
| V2 | The risk-tier chart shows the real SKU count per tier (and all six tiers, zeros included) | `Tracked SKUs` used REMOVEFILTERS on the chart's own category and drew the same total on every bar |
| V4 | Date range is a range picker, default = full range | Usability (approved) |
| V9 | Healthy, Overstock and No recent sales get colours | The report only coloured three tiers |
| V13 | Blank values show as an em dash; blank SKU counts show as 0 | Presentation only |

Unchanged from Power BI: every KPI definition, the snapshot rule (risk tiers, SKU counts and stock
value ignore the date range and are never summed across days), the original titles and the
synthetic-data banner wording.

## Sales Intelligence: where it differs from the Power BI page (intentional)

| ID | Difference | Reason |
|----|------------|--------|
| V1 | Every filter starts at All (the report saved Wilaya = Alger); the page keeps its own filters | Decision D1 |
| V4 | Date range is a range picker, default = full range | Usability (approved) |
| S1 | The product grid is sortable by clicking a header; its default order is the one saved in the report (category A-Z), not "ranked by net sales" as its title says | Report title and saved sort disagree; both kept visible |
| S2 | Matrix header `year_month_label` reads "Month"; household group is labelled "(Blank)" as Power BI shows it | Readability |
| S3 | The grand-total rows of the matrix and the product table are the Power BI defaults (not set explicitly in the report) | Assumption, see notes |

Month-over-month follows the report's DAX (`Net Sales MoM %`), including the rule that it is blank unless
both months are complete. The behaviour of `DATEADD(-1 MONTH)` on month-end dates could not be executed
without Power BI Desktop and is implemented as documented DAX behaviour (assumption).

## Inventory Intelligence, Forecasting Performance, Forecast Accuracy by Horizon, Data & Method Notes

| Page | Difference from the Power BI report (intentional) | Reason |
|------|---------------------------------------------------|--------|
| Inventory | Two days-of-cover cards: overall (the report's measure) and median SKU | Decision D3 |
| Inventory | The risk-tier filter applies to the SKU list only; the At-Risk share ignores it | Approved V6 |
| Inventory | Availability uses the same filters as the stockout rate (the report ignored Category) | Approved V7 |
| Inventory | The category x store table is a heat map; the tier chart shows real SKU counts and all six tiers | Approved V8 / V2 |
| Forecasting | Benchmark MAE ignores the model filter (the intended meaning, approved spec). The literal DAX overrides `model_name` while the slicer filters `model_label`, so Power BI may show a blank benchmark for non-benchmark rows. This was NOT executed in Power BI Desktop. | Open point V12 |
| Forecasting / Horizon | No grand-total or subtotal rows in the matrices: at total level the measures would pool different models | Meaningless values |
| Horizon | "Observations behind each point" is the literal measure: observations summed over all selected models | Fidelity |
| Notes | The two text blocks are verbatim; coverage, files, integrity, reconciliation and this register are read from the dataset | Page brief |

Nothing in the dashboard produces or displays a forecast of the future: `fact_demand_forecast` and
`model_run` are empty in this dataset; every accuracy figure is a backtest on past origins.
