import streamlit as st

from sections.shared import kpi_card, section


def render_insights(df, data, file):
    section("Insights", "Data context, review prompts, and demo talking points.")

    date_range = data["transaction_date"].dropna()
    start_date = date_range.min().strftime("%b %d, %Y") if not date_range.empty else "N/A"
    end_date = date_range.max().strftime("%b %d, %Y") if not date_range.empty else "N/A"
    provinces = data["province"].nunique()
    products = data["item"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Visible rows", f"{len(data):,}", "After filters")
    with c2:
        kpi_card("Date range", end_date, f"Starts {start_date}")
    with c3:
        kpi_card("Provinces", f"{provinces:,}", "In current view")
    with c4:
        kpi_card("Products", f"{products:,}", "In current view")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("What to review", "Useful checks while reading the dashboard.")
        st.markdown('<div class="placeholder">Check whether sales are concentrated in one province or spread across regions.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Compare top products with the busiest days to plan staffing and stock.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Use the forecast gap to see whether the forecast is close enough for planning.</div>', unsafe_allow_html=True)

    with right, st.container(border=True):
        section("Presentation notes", "Simple points for the demo.")
        st.markdown('<div class="placeholder">Overview gives the quick business picture: revenue, order mix, products, and strongest days.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Sales gives the detailed view: trends, regions, products, and activity patterns.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Forecast shows expected sales and how they compare with actuals when available.</div>', unsafe_allow_html=True)

    with st.expander("Data quality snapshot"):
        st.write(f"Source file: `{file.name}`")
        st.write(f"{len(df):,} rows, {len(df.columns):,} features")
        st.write(f"{df['transaction_date'].notna().sum():,} rows with valid dates")
        st.write(f"{df['total_spent'].notna().sum():,} rows with valid revenue")
        st.write(f"Visible after filters: {len(data):,} rows")
