import streamlit as st
from streamlit_echarts import st_echarts

from charts import (
    bar_option,
    donut_option,
    #forecast_option,
    #heatmap_option,
    money,
    monthly_option,
    #weekday_option,
)

from sections.shared import kpi_card, section, show_chart

def kpi_strip(metrics):
    for col, args in zip(st.columns(5), [
        ("Revenue", money(metrics["total"]), "Filtered sales"),
        ("Orders", f"{metrics['transactions']:,.0f}", "Unique transactions"),
        ("Avg order", money(metrics["aov"]), "Revenue per transaction"),
        ("Top province", metrics["top_province"], "Best region"),
        ("Top item", metrics["top_item"], "Best product"),
    ]):
        with col:
            kpi_card(*args)


def render_overview(metrics, monthly, province, location):
    section("Overview", "Main results for the current filters.")
    kpi_strip(metrics)
    with st.container(border=True):
        section("Monthly revenue", "Revenue by month.")
        show_chart(monthly_option(monthly), "overview_monthly_revenue", "340px")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Revenue by province")
        show_chart(bar_option(province, "province", horizontal=False), "overview_province_bar")
    with right, st.container(border=True):
        section("Visit mode")
        show_chart(donut_option(location, "location"), "overview_location_donut", "300px")
