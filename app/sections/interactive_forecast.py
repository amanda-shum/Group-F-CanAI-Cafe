import streamlit as st
from src.modeling import (
    fit_sarima_model,
    generate_sarima_forecast,
    split_time_series,
    clip_negative_forecasts
)
from run_model import load_daily_sales


def run():
    st.title("Interactive Forecasting Tool")

    # Sidebar controls
    st.sidebar.header("Controls")

    forecast_days = st.sidebar.slider("Forecast Horizon", 7, 180, 30)

    p = st.sidebar.selectbox("p", [0, 1, 2])
    d = st.sidebar.selectbox("d", [0, 1])
    q = st.sidebar.selectbox("q", [0, 1, 2])

    # Load data
    data = load_daily_sales()
    train, val, test = split_time_series(data)

    if st.button("Run Forecast"):
        model = fit_sarima_model(
            train["daily_total_sales"],
            order=(p, d, q)
        )

        forecast = generate_sarima_forecast(model, steps=forecast_days)
        forecast = clip_negative_forecasts(forecast)

        st.line_chart(forecast["forecast"])
