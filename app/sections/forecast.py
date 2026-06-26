import streamlit as st

from charts import forecast_option, forecast_weekday_option, future_forecast_option, money
from sections.shared import kpi_grid, section, show_chart


def render_forecast(monthly, forecast):
    section("Forecast", "Expected sales for upcoming planning.")
    if forecast is None:
        st.info("Forecast files are not available yet. Run `python run_model.py test` to generate them.")
        return

    with st.container(border=True):
        section("Forecast summary", "Simple numbers to guide the next planning cycle.")
        kpi_grid([
            ("Next month", money(forecast["next_month"]), "Expected sales"),
            ("Next 3 months", money(forecast["next_quarter"]), "Expected sales"),
            ("6-month total", money(forecast["six_month_total"]), "Forecast period"),
        ], 3)

    with st.container(border=True):
        section("Actual vs forecast", "Monthly sales compared with the forecast.")
        show_chart(forecast_option(monthly, forecast), "forecast_vs_actual", "360px")

    left, right = st.columns(2)
    with left, st.container(border=True, height=370):
        section("Next 6 months", "Expected revenue by month.")
        show_chart(future_forecast_option(forecast), "future_forecast", "300px")
    with right, st.container(border=True, height=370):
        section("Forecast by weekday", "Expected average sales for each weekday.")
        show_chart(forecast_weekday_option(forecast), "forecast_weekday", "300px")
