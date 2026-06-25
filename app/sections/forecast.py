import streamlit as st
from streamlit_echarts import st_echarts

from charts import forecast_option, money
from sections.shared import kpi_card, section, show_chart

def render_forecast(monthly, forecast):
    section("Forecast", "Actual revenue compared with placeholder forecast values.")
    if forecast is None:
        st.info("At least three months of sales history are needed for the forecast view.")
        return

    with st.container(border=True):
        section("Forecast summary", "Forecast values are placeholders until the model output is added.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Latest month", money(forecast["latest"]), "Most recent actual")
        with c2:
            kpi_card("Recent average", money(forecast["recent_avg"]), "Last three months")
        with c3:
            kpi_card("Next month", money(forecast["next_month"]), f"{forecast['change']:+.1%} vs latest")
        with c4:
            kpi_card("Next quarter", money(forecast["next_quarter"]), "Three-month total")
        show_chart(forecast_option(monthly, forecast), "forecast_vs_actual", "340px")

    with st.container(border=True):
        section("Forecast notes")
        st.write("Current method: placeholder values based on recent monthly revenue.")
        st.write("Next step: replace placeholders with the team's forecast output.")