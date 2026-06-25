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
    for col, args in zip(st.columns(5, gap="large"), [
        ("Revenue", money(metrics["total"]), "Filtered sales"),
        ("Orders", f"{metrics['transactions']:,.0f}", "Unique transactions"),
        ("Avg order", money(metrics["aov"]), "Revenue per transaction"),
        ("Top province", metrics["top_province"], "Best region"),
        ("Top item", metrics["top_item"], "Best product"),
    ]):
        with col:
            kpi_card(*args)


def render_overview(metrics, monthly, province,products, location, weekday):

    total_revenue = province["total_spent"].sum()

    section("Overview", "Main results for the current filters.")
    kpi_strip(metrics)
    with st.container(border=True):
        section("Monthly revenue", "Revenue by month.")
        show_chart(monthly_option(monthly), "overview_monthly_revenue", "340px")

    left, right = st.columns(2)

    top_regions = province.sort_values("total_spent", ascending=False).head(3)

    with left, st.container(border=True):
        section("Top Provinces", "Share of revenue from the top provinces!")

        for _, region in top_regions.iterrows():
            share = region["total_spent"] / total_revenue if total_revenue else 0
            st.markdown(
                f"""
                <div class="placeholder">
                    <div style="display:flex; justify-content:space-between; gap:1rem;">
                        <strong>{region["province"]}</strong>
                        <span>{share:.0%}</span>
                    </div>
                    <div style="color:inherit; opacity:.72; margin-top:.15rem;">
                        {money(region["total_spent"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    with right, st.container(border=True):
        section("Revenue by Province")
        show_chart(bar_option(province, "province", horizontal=False), "overview_province_bar")

    left,middle,right = st.columns([1,1,1])

    with left, st.container(border=True):
        section("Order Type")
        show_chart(donut_option(location, "location"), "overview_location_donut", "300px")

    with middle, st.container(border=True):
        section("Top Products", "Top Products by Revenue")
        top_products = products.head(6)
        for _, row in top_products.iterrows():
            st.write(f"{row['label']} - {money(row['total_spent'])}")

    best_days = weekday.sort_values("total_spent", ascending=False).head(3)
    with right, st.container(border=True):
        section("Best Sales Day", "Highest Revenue Weekdays")
        for _, day in best_days.iterrows():
            st.write(f"**{day['weekday']}** - {money(day['total_spent'])} . {day['transactions']:,.0f} orders")

    
