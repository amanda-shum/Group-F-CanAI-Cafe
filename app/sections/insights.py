import streamlit as st
from streamlit_echarts import st_echarts

from sections.shared import kpi_card, section

def render_insights(df, data, file):
    section("Insights", "Short notes and action items for the current view.")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Notes", "Pending", "Summary not connected yet")
    with c2:
        kpi_card("Actions", "Pending", "Recommendations not connected yet")
    with c3:
        kpi_card("Data snapshot", f"{len(data):,}", "Visible rows")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Notes")
        st.markdown('<div class="placeholder">Top region, strongest product, and main sales pattern will appear here.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Forecast summary and data notes will appear here.</div>', unsafe_allow_html=True)
    with right, st.container(border=True):
        section("Actions")
        st.markdown('<div class="placeholder">Inventory, staffing, campaign timing, and region ideas will appear here.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder">Forecasting and reporting next steps will appear here.</div>', unsafe_allow_html=True)

    with st.expander("Data quality snapshot"):
        st.write(f"Source file: `{file.name}`")
        st.write(f"{len(df):,} rows, {len(df.columns):,} features")
        st.write(f"{df['transaction_date'].notna().sum():,} rows with valid dates")
        st.write(f"{df['total_spent'].notna().sum():,} rows with valid revenue")
        st.write(f"Visible after filters: {len(data):,} rows")
