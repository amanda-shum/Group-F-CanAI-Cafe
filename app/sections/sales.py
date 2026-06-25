import streamlit as st
from streamlit_echarts import st_echarts

from charts import (
    bar_option,
    donut_option,
    #forecast_option,
    heatmap_option,
    #money,
    monthly_option,
    weekday_option,
)

from sections.shared import section, show_chart

def render_sales(monthly, weekday, province, products, location, activity):
    section("Sales", "Revenue, products, regions, and customer activity.")
    with st.container(border=True):
        section("Monthly revenue")
        show_chart(monthly_option(monthly), "sales_monthly_revenue", "340px")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Best sales days", "Revenue and order count by weekday.")
        show_chart(weekday_option(weekday), "sales_weekday_sales")
    with right, st.container(border=True):
        section("Top products")
        show_chart(bar_option(products, "label"), "sales_product_bar")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Revenue by province")
        show_chart(bar_option(province, "province"), "sales_province_bar")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Visit mode")
        show_chart(donut_option(location, "location"), "sales_location_donut", "300px")
    with right, st.container(border=True):
        section("Activity heatmap", "Order volume by weekday and month.")
        show_chart(heatmap_option(activity), "sales_activity_heatmap", "300px")
