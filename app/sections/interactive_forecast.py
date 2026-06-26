import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.modeling import (
    fit_sarima_model,
    generate_sarima_forecast,
    split_time_series,
    clip_negative_forecasts,
    forecast_grouped_sales,
    aggregate_daily_sales_by_group,
    complete_grouped_daily_sales
)

from run_model import load_daily_sales, load_raw_transactions


def render_interactive_forecast():
    st.title("📈 Interactive Forecasting Tool")
    st.markdown("Explore future scenarios by adjusting model behavior and business assumptions.")

    # =====================================================
    # ✅ FORECAST SETTINGS
    # =====================================================
    st.subheader("Forecast Settings")

    col1, col2 = st.columns(2)

    with col1:
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 180, 30)

        trend_sensitivity = st.selectbox(
            "Trend Sensitivity",
            [0, 1, 2],
            index=1,
            help="How strongly forecasts react to recent trends"
        )

    with col2:
        trend_stability = st.selectbox(
            "Trend Stabilization",
            [0, 1],
            index=1,
            help="Helps smooth unstable trends"
        )

        noise_smoothing = st.selectbox(
            "Noise Smoothing",
            [0, 1, 2],
            index=1,
            help="Controls how much noise is reduced"
        )

    # =====================================================
    # ✅ SCENARIO SETTINGS
    # =====================================================
    st.subheader("Scenario Simulation")

    col3, col4 = st.columns(2)

    with col3:
        growth_adjustment = st.slider("Expected Demand Change (%)", -30, 30, 0)

    with col4:
        uncertainty_multiplier = st.slider("Risk / Uncertainty Level", 0.5, 2.0, 1.0)

    # =====================================================
    # ✅ FORECAST SCOPE
    # =====================================================
    st.subheader("Forecast Scope")

    scope = st.selectbox("Forecast Level", ["Overall", "Province", "Item"])

    data = load_daily_sales()

    # =====================================================
    # ✅ OVERALL FORECAST
    # =====================================================
    if scope == "Overall":
        train, val, test = split_time_series(data)

        model = fit_sarima_model(
            train["daily_total_sales"],
            order=(trend_sensitivity, trend_stability, noise_smoothing)
        )

        forecast = generate_sarima_forecast(model, steps=forecast_days)
        forecast = clip_negative_forecasts(forecast)

        forecast["adjusted"] = forecast["forecast"] * (1 + growth_adjustment / 100)
        forecast["lower_adj"] = forecast["lower_bound"] * uncertainty_multiplier
        forecast["upper_adj"] = forecast["upper_bound"] * uncertainty_multiplier

        st.subheader("Forecast Comparison")

        fig = go.Figure()

        # ✅ Historical
        fig.add_trace(go.Scatter(
            x=train.index,
            y=train["daily_total_sales"],
            name="Historical"
        ))

        # ✅ Last actual point (BONUS)
        fig.add_trace(go.Scatter(
            x=[train.index[-1]],
            y=[train["daily_total_sales"].iloc[-1]],
            mode="markers",
            marker=dict(size=10),
            name="Last Actual"
        ))

        # ✅ Base forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["forecast"],
            name="Base Forecast"
        ))

        # ✅ Scenario forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["adjusted"],
            name="Scenario Forecast"
        ))

        # ✅ Uncertainty band
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["upper_adj"],
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast["lower_adj"],
            fill='tonexty',
            name="Uncertainty Range",
            line=dict(width=0)
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ✅ Business outputs
        st.subheader("Business Summary")

        total = forecast["adjusted"].sum()
        avg = forecast["adjusted"].mean()

        colA, colB = st.columns(2)
        colA.metric("Total Forecast Revenue", f"${total:,.2f}")
        colB.metric("Average Daily Revenue", f"${avg:,.2f}")

        with st.expander("📄 View Forecast Data"):
            st.dataframe(forecast)

    # =====================================================
    # ✅ GROUPED FORECAST (PROVINCE / ITEM)
    # =====================================================
    else:
        st.subheader(f"📂 {scope} Forecast")

        raw_data = load_raw_transactions()
        group_col = ["Province"] if scope == "Province" else ["Item"]

        # ✅ FIXED PIPELINE
        grouped_daily = aggregate_daily_sales_by_group(
            raw_data,
            group_cols=group_col,
            date_col="Transaction Date",
            sales_col="Total Spent"
        )

        grouped_daily = complete_grouped_daily_sales(
            grouped_daily,
            group_cols=group_col,
            date_col="date"
        )

        grouped_forecast = forecast_grouped_sales(
            grouped_daily,
            group_cols=group_col,
            forecast_steps=forecast_days
        )

        # ✅ Select group
        selected = st.selectbox(
            f"Select {scope}",
            sorted(grouped_forecast[group_col[0]].unique())
        )

        # ✅ Historical slice
        history = grouped_daily[
            grouped_daily[group_col[0]] == selected
        ].sort_values("date")

        # ✅ Forecast slice
        forecast_filtered = grouped_forecast[
            grouped_forecast[group_col[0]] == selected
        ].copy()

        forecast_filtered["adjusted"] = (
            forecast_filtered["forecast"] * (1 + growth_adjustment / 100)
        )

        # =====================
        # CHART
        # =====================
        fig = go.Figure()

        # ✅ Historical
        fig.add_trace(go.Scatter(
            x=history["date"],
            y=history["daily_total_sales"],
            name="Historical"
        ))

        # ✅ Last actual point (BONUS)
        fig.add_trace(go.Scatter(
            x=[history["date"].iloc[-1]],
            y=[history["daily_total_sales"].iloc[-1]],
            mode="markers",
            marker=dict(size=10),
            name="Last Actual"
        ))

        # ✅ Base forecast
        fig.add_trace(go.Scatter(
            x=forecast_filtered["date"],
            y=forecast_filtered["forecast"],
            name="Base Forecast"
        ))

        # ✅ Scenario forecast
        fig.add_trace(go.Scatter(
            x=forecast_filtered["date"],
            y=forecast_filtered["adjusted"],
            name="Scenario Forecast"
        ))

        st.plotly_chart(fig, use_container_width=True)

        # =====================
        # BUSINESS OUTPUTS
        # =====================
        st.subheader("💼 Business Summary")

        total = forecast_filtered["adjusted"].sum()
        avg = forecast_filtered["adjusted"].mean()

        col1, col2 = st.columns(2)
        col1.metric("Total Forecast", f"${total:,.2f}")
        col2.metric("Average Daily", f"${avg:,.2f}")

        with st.expander("📄 View Data"):
            st.dataframe(forecast_filtered)