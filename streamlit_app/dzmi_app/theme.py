"""Design tokens (no Streamlit import). Colours come from the report's custom theme."""

PRIMARY = "#18A6A6"      # teal accent
BLUE = "#2878B5"
NAVY = "#17324D"
RED = "#D9534F"
DEEP_RED = "#A61C1C"
LOW_BLUE = "#5B9BD5"
AMBER = "#E6A23C"
GOOD = "#279B75"
GREY = "#6B7785"
TEXT = "#263238"
SURFACE = "#F5F7FA"
BORDER = "#E3E8EE"
GRID = "#EEF1F5"
WHITE = "#FFFFFF"

FONT_STACK = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

# Original report colours for Critical / Stockout / Low; Healthy, Overstock and
# No recent sales had no custom colour in Power BI and are added here (deviation V9).
TIER_COLORS = {
    "Stockout": DEEP_RED,
    "Critical": RED,
    "Low": LOW_BLUE,
    "Healthy": GOOD,
    "Overstock": AMBER,
    "No recent sales": GREY,
}
