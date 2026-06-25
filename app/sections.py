import streamlit as st
from streamlit_echarts import st_echarts

from charts import (
    bar_option,
    donut_option,
    forecast_option,
    heatmap_option,
    money,
    monthly_option,
    weekday_option,
)


ITEM_BADGES = {
    "Coffee": "C",
    "Tea": "T",
    "Donut": "D",
    "Sandwich": "S",
    "Cookie": "K",
    "Juice": "J",
    "Refresher": "R",
    "Salad": "G",
    "Unknown": "?",
}


def kpi_card(label, value, hint):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="note">{hint}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title, note=None):
    html = f'<div class="section-heading"><div class="section-title">{title}</div>'
    if note:
        html += f'<div class="note">{note}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def show_chart(options, key, height="320px"):
    """Small wrapper so every ECharts render stays consistent and easy to find."""
    st_echarts(options=options, height=height, key=key)


def kpi_strip(metrics):
    for col, args in zip(st.columns(5), [
        ("Revenue", money(metrics["total"]), "Filtered sales"),
        #("Transactions", f"{metrics['transactions']:,.0f}", "Unique orders"), doesn't fit nicely and is the least relevant?
        ("Items sold", f"{metrics['items_sold']:,.0f}", "Total quantity"),
        ("Avg order", money(metrics["aov"]), "Revenue per order"),
        ("Top province", metrics["top_province"], "Best region"),
        ("Top item", metrics["top_item"], "Best product"),
    ]):
        with col:
            kpi_card(*args)


def render_overview(metrics, monthly, province):
    section("Overview", "Key metrics")
    kpi_strip(metrics)
    with st.container(border=True):
        section("Monthly revenue", "Historical monthly sales")
        show_chart(monthly_option(monthly), "overview_monthly_revenue", "340px")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Revenue by province")
        show_chart(bar_option(province, "province", horizontal=False), "overview_province_bar")


def render_sales(monthly, weekday, province, products, location, activity):
    section("Sales", "Revenue trends, regional performance, product mix, and customer behavior.")
    with st.container(border=True):
        section("Monthly revenue")
        show_chart(monthly_option(monthly), "sales_monthly_revenue", "340px")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Best sales days", "Revenue bars plus order-count line by weekday.")
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
        section("Activity heatmap", "Which weekdays and months generate the most order volume.")
        show_chart(heatmap_option(activity), "sales_activity_heatmap", "300px")


def render_forecast(monthly, forecast):
    section("Forecast", "Actual monthly sales followed by a simple average forecast.")
    if forecast is None:
        st.info("At least three months of sales history are needed for this baseline forecast.")
        return

    with st.container(border=True):
        section("Forecast summary", "Will not look like this, will prob be line chart comparing actual vs forecast! or bar charts next to each other")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Latest Month", money(forecast["latest"]), "Most recent actual")
        with c2:
            kpi_card("Monthly Avg", money(forecast["recent_avg"]), "Recent run rate")
        with c3:
            kpi_card("Next Month", money(forecast["next_month"]), f"{forecast['change']:+.1%} vs latest")
        with c4:
            kpi_card("Next Quarter", money(forecast["next_quarter"]), "3-month total")
        show_chart(forecast_option(monthly, forecast), "forecast_vs_actual", "340px")

    with st.container(border=True):
        section("Forecast assumptions")
        st.write("Forecast method: TBD")
        st.write("more detailes to come...")


def render_insights(df, data, file):
    section("Insights", "Placeholder area for summaries and recommendations.")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Insights", "Summary Placeholder", "...")
    with c2:
        kpi_card("Recommendations", "Summary Placeholder", "Action planning slot")
    with c3:
        kpi_card("Data snapshot", f"{len(data):,}", "Visible rows")

    left, right = st.columns(2)
    with left, st.container(border=True):
        section("Insights")
        st.markdown('<div class="placeholder"> Top region, strongest product, seasonality pattern, and unusual customer behavior will appear here.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder"> Average forecast interpretation and data caveats will appear here.</div>', unsafe_allow_html=True)
    with right, st.container(border=True):
        section("Recommendations")
        st.markdown('<div class="placeholder"> Inventory, staffing, campaign timing, and regional targeting ideas will appear here.</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder"> Next forecasting/modeling steps will appear here.</div>', unsafe_allow_html=True)

    with st.expander("Data quality snapshot"):
        st.write(f"Source file: `{file.name}`")
        st.write(f"{len(df):,} rows, {len(df.columns):,} features")
        st.write(f"{df['transaction_date'].notna().sum():,} rows with valid dates")
        st.write(f"{df['total_spent'].notna().sum():,} rows with valid revenue")
        st.write(f"Visible after filters: {len(data):,} rows")
