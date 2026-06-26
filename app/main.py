"""
CanAI Cafe dashboard!!.

This file is the main entry point for the dashboard web app. 
It:
- configures the Streamlit page and theme
- loads the cleaned cafe sales dataset
- applies sidebar filters (province and month)
- prepares aggregated data for dashboard views
- renders the Overview, Sales, and Forecast sections

The app depends on helper modules such as:
- charts.py for visual components and forecasting helpers
- data.py for loading the dataset
- sections.py for rendering dashboard pages

Run with:
    streamlit run app/main.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path
import sys
import charts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from charts import build_forecast_summary
from data import load_data, load_forecast_reports
from sections import (
    product_label,
    render_forecast,
    render_overview,
    render_sales,
    render_interactive_forecast
)

PAGE_TITLE = "CanAI Cafe Dashboard!"

LIGHT_COLORS = {
    "page_background": "#FFFDF8",
    "card_background": "#FFFFFF",
    "primary_text": "#2B2118",
    "secondary_text": "#756A5E",
    "chart_grid": "#EDE6DB",
    "card_border": "#E2D7C9",
    "chart_primary": "#6F8F72",
    "chart_secondary": "#C79A4B",
    "chart_tertiary": "#B86F52",
    "chart_soft": "#A7B99E",
    "chart_deep": "#3E2B20",
    "heatmap_low": "#FBF7EF",
    "sidebar_background": "#332211",
    "sidebar_border": "#34343D",
    "sidebar_text": "#ECE8E2",
    "sidebar_muted_text": "#AAA8B0",
    "sidebar_quiet_text": "#6F6E78",
    "sidebar_brand_background": "#D98C32",
    "sidebar_brand_text": "#FFFFFF",
    "sidebar_brand_subtitle": "#8E8D96",
    "sidebar_active_background": "#3A302D",
    "sidebar_active_text": "#FFFDD0",
    "sidebar_input_background": "#332211",
    "sidebar_input_border": "#555560",
    "soft_shadow": "rgba(0,0,0,.04)",
}

#streamlit supports dark theme!
DARK_COLORS = {
    "page_background": "#171512",
    "card_background": "#211E19",
    "primary_text": "#F6EFE3",
    "secondary_text": "#B9AB99",
    "chart_grid": "#3A332B",
    "card_border": "#443B31",
    "chart_primary": "#95B889",
    "chart_secondary": "#D4AA5E",
    "chart_tertiary": "#CF7D62",
    "chart_soft": "#71896C",
    "chart_deep": "#F0D9B5",
    "heatmap_low": "#2A261F",
    "sidebar_background": "#332211",
    "sidebar_border": "#332211",
    "sidebar_text": "#ECE8E2",
    "sidebar_muted_text": "#AAA8B0",
    "sidebar_quiet_text": "#6F6E78",
    "sidebar_brand_background": "#FFFDD0",
    "sidebar_brand_text": "#FFFFFF",
    "sidebar_brand_subtitle": "#8E8D96",
    "sidebar_active_background": "#3A302D",
    "sidebar_active_text": "#FFFDD0",
    "sidebar_input_background": "#332211",
    "sidebar_input_border": "#555560",
    "soft_shadow": "rgba(0,0,0,.18)",
}


def apply_theme(colors):
    st.markdown(f"""
    <style>
    header[data-testid="stHeader"] {{ background: {colors["sidebar_background"]}; }}
    [data-testid="stSidebarCollapsedControl"] {{ display:flex !important; visibility:visible; }}
    .stApp {{ background:{colors["page_background"]}; color:{colors["primary_text"]}; }}
    .block-container {{ max-width:1420px; padding-top:3.2rem; padding-bottom:2.5rem; }}
    [data-testid="stVerticalBlock"] {{ gap:1.35rem; }}
    [data-testid="stHorizontalBlock"] {{ gap:1.35rem; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ margin-top:.25rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        
        background:{colors["card_background"]};
        border-color:{colors["card_border"]};
        box-shadow:0 10px 24px {colors["soft_shadow"]};
    }}
    .title {{ font-size:2.1rem; font-weight:800; line-height:1.05; }}
    .subtitle {{ color:{colors["primary_text"]}; opacity:.72; font-size:1rem; margin:.3rem 0 1.25rem; }}
    [data-testid="stSidebar"] {{ background:{colors["sidebar_background"]}; border-right:1px solid {colors["sidebar_border"]}; }}
    [data-testid="stSidebar"] * {{ color:{colors["sidebar_text"]}; }}
    [data-testid="stSidebar"] .brand {{ display:flex; gap:.85rem; align-items:center; padding:.9rem .25rem 1.05rem; }}
    [data-testid="stSidebar"] .brand-mark {{
        width:2.6rem; height:2.6rem; border-radius:.75rem;
        display:flex; align-items:center; justify-content:center;
        background:{colors["sidebar_brand_background"]}; color:{colors["sidebar_brand_text"]}; font-weight:850; font-size:1.15rem;
    }}
    [data-testid="stSidebar"] .brand-title {{ font-size:1.1rem; font-weight:850; line-height:1.05; }}
    [data-testid="stSidebar"] .brand-subtitle {{ color:{colors["sidebar_brand_subtitle"]}; font-size:.78rem; font-weight:800; letter-spacing:.16em; margin-top:.25rem; }}
    [data-testid="stSidebar"] hr {{ border-color:{colors["sidebar_border"]}; margin:1rem 0; }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.35rem; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding:.78rem .85rem; border-radius:.45rem; margin:.12rem 0;
        color:{colors["sidebar_muted_text"]}; font-weight:760;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background:{colors["sidebar_active_background"]}; border-left:3px solid {colors["sidebar_brand_background"]};
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {{ color:{colors["sidebar_active_text"]}; }}
    [data-testid="stSidebar"] div[role="radiogroup"] input {{ display:none; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background:{colors["sidebar_input_background"]}; border-color:{colors["sidebar_input_border"]}; border-radius:.42rem;
    }}
    [data-testid="stSidebar"] label {{ color:{colors["sidebar_muted_text"]} !important; font-weight:760; }}
    .sidebar-label {{ color:{colors["sidebar_quiet_text"]}; font-size:.8rem; font-weight:850; letter-spacing:.18em; margin:.3rem 0 .75rem; }}
    .sidebar-footer {{ color:{colors["sidebar_quiet_text"]}; font-family:monospace; font-weight:700; font-size:.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid {colors["sidebar_border"]}; }}
    .section-heading {{ margin:.2rem 0 .65rem; }}
    .section-title {{ font-size:1.06rem; font-weight:760; margin-bottom:.22rem; }}
    .note {{ color:{colors["primary_text"]}; opacity:.68; font-size:.86rem; margin-bottom:.45rem; }}
    .kpi-grid {{ display:grid; gap:.75rem; width:100%; margin:.15rem 0 .35rem; }}
    .kpi {{ border:1px solid {colors["card_border"]}; border-radius:8px; padding:.55rem .6rem; background:{colors["card_background"]}; min-height:4.8rem; overflow:hidden; }}
    .kpi-label {{ color:{colors["primary_text"]}; opacity:.68; font-size:.68rem; font-weight:750; text-transform:uppercase; letter-spacing:.035em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .kpi-value {{ font-size:clamp(.8rem, .9vw, 1.02rem); font-weight:820; line-height:1.12; margin-top:.18rem; overflow-wrap:anywhere; }}
    .placeholder {{ border:1px dashed {colors["card_border"]}; border-radius:8px; background:{colors["page_background"]}; padding:.75rem .85rem; margin:.45rem 0; font-size:.96rem; }}
    .summary-row {{ border:1px solid {colors["card_border"]}; border-radius:8px; background:{colors["page_background"]}; padding:.62rem .72rem; margin:.45rem 0; }}
    .summary-line {{ display:flex; justify-content:space-between; gap:.8rem; align-items:center; font-weight:760; }}
    .summary-meta {{ color:{colors["secondary_text"]}; font-size:.82rem; margin-top:.14rem; }}
    .summary-bar {{ height:.38rem; border-radius:99px; background:{colors["chart_grid"]}; margin-top:.45rem; overflow:hidden; }}
    .summary-fill {{ height:100%; border-radius:99px; background:{colors["chart_primary"]}; }}
    .rank-table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
    .rank-table th {{ color:{colors["secondary_text"]}; font-size:.72rem; font-weight:850; text-align:left; text-transform:uppercase; letter-spacing:.04em; padding:.2rem .1rem .45rem; border-bottom:1px solid {colors["card_border"]}; }}
    .rank-table th:last-child {{ text-align:right; }}
    .rank-table td {{ padding:.58rem .1rem; border-bottom:1px solid {colors["card_border"]}; }}
    .rank-table tr:last-child td {{ border-bottom:none; }}
    .rank-number {{ color:{colors["secondary_text"]}; font-weight:800; width:2rem; }}
    .rank-value {{ text-align:right; font-weight:800; }}
    .rank-share {{ text-align:right; color:{colors["secondary_text"]}; font-weight:760; }}
    .podium-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:.7rem; align-items:end; margin-top:.6rem; }}
    .podium-card {{ border:1px solid {colors["card_border"]}; border-radius:8px; background:{colors["page_background"]}; padding:.75rem .65rem; text-align:center; }}
    .podium-rank {{ color:{colors["chart_secondary"]}; font-size:.8rem; font-weight:850; text-transform:uppercase; letter-spacing:.04em; }}
    .podium-day {{ font-size:1.08rem; font-weight:850; margin-top:.25rem; }}
    .podium-value {{ margin-top:.35rem; font-weight:800; }}
    .podium-meta {{ color:{colors["secondary_text"]}; font-size:.8rem; margin-top:.2rem; }}
    .equal-card-short {{ min-height:320px; }}
    .equal-card-medium {{ min-height:360px; }}
    </style>
    """, unsafe_allow_html=True)


def sidebar_controls(df):
    """Render side navigation and simple filters."""
    valid_months = sorted(df["month"].dropna().unique())
    month_lookup = {pd.Timestamp(m).strftime("%b %Y"): pd.Timestamp(m) for m in valid_months}

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">C</div>
              <div>
                <div class="brand-title">CanAI Cafe</div>
                <div class="brand-subtitle">DASHBOARD</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        labels = {"Overview": "Overview", "Sales": "Sales", "Forecast": "Forecast", "Interactive Forecast": "Interative Forecast"}
        page = st.radio("Menu", list(labels), format_func=lambda option: labels[option], label_visibility="collapsed")
        st.divider()
        st.markdown('<div class="sidebar-label">FILTERS</div>', unsafe_allow_html=True)
        province = st.selectbox("Province", ["All"] + sorted(df["province"].dropna().unique()))
        month = st.selectbox("Month", ["All"] + list(month_lookup))
        st.markdown('<div class="sidebar-footer">Updated Jun 25 2026</div>', unsafe_allow_html=True)

    filtered = df.copy()
    if province != "All":
        filtered = filtered[filtered["province"] == province]
    if month != "All":
        filtered = filtered[filtered["month"] == month_lookup[month]]
    return page, filtered


def prepare_dashboard_data(data):
    """Create the small aggregated tables each dashboard section needs."""
    revenue = data.dropna(subset=["total_spent"])
    trend = revenue.dropna(subset=["transaction_date"])
    if trend.empty:
        return None

    metrics = {
        "total": revenue["total_spent"].sum(),
        "transactions": data["transaction_id"].nunique(),
        "top_province": revenue.groupby("province")["total_spent"].sum().idxmax() if not revenue.empty else "N/A",
        "top_item": revenue.groupby("item")["total_spent"].sum().idxmax() if not revenue.empty else "N/A",
    }
    metrics["aov"] = metrics["total"] / metrics["transactions"] if metrics["transactions"] else 0
    active_days = trend["transaction_date"].dt.date.nunique()
    metrics["average_daily"] = metrics["total"] / active_days if active_days else 0

    monthly = trend.groupby("month", as_index=False)["total_spent"].sum().sort_values("month")
    weekday = (
        trend.assign(weekday=pd.Categorical(trend["transaction_date"].dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ordered=True))
        .groupby("weekday", observed=False)
        .agg(total_spent=("total_spent", "sum"), transactions=("transaction_id", "nunique"))
        .reset_index()
    )
    province = revenue.groupby("province", as_index=False)["total_spent"].sum().sort_values("total_spent", ascending=False)
    products = revenue.groupby("item", as_index=False)["total_spent"].sum().sort_values("total_spent", ascending=False)
    products["label"] = products["item"].map(product_label)
    location = revenue.groupby("location", as_index=False)["total_spent"].sum()
    activity = (
        trend.assign(
            weekday=pd.Categorical(trend["transaction_date"].dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            month=trend["transaction_date"].dt.strftime("%b"),
            month_num=trend["transaction_date"].dt.month,
        )
        .groupby(["weekday", "month", "month_num"], observed=False)
        .size()
        .reset_index(name="transactions")
        .sort_values(["month_num", "weekday"])
        .drop(columns="month_num")
    )
    return metrics, monthly, weekday, province, products, location, activity


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="B", layout="wide", initial_sidebar_state="expanded")
    colors = DARK_COLORS if st.context.theme.get("type") == "dark" else LIGHT_COLORS
    charts.set_colors(colors)
    apply_theme(colors)

    df, file, error = load_data()
    if error:
        st.error(error)
        st.stop()

    monthly_forecast, six_month_forecast, daily_forecast, forecast_error = load_forecast_reports()
    if forecast_error:
        st.warning(forecast_error)

    st.markdown(f'<div class="title">{PAGE_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sales, customer activity, and forecast views for cafe performance.</div>', unsafe_allow_html=True)

    page, data = sidebar_controls(df)
    prepared = prepare_dashboard_data(data)
    if prepared is None:
        st.warning("No dated revenue rows are available for trend charts.")
        return

    metrics, monthly, weekday, province, products, location, activity = prepared
    forecast = build_forecast_summary(monthly, monthly_forecast, six_month_forecast, daily_forecast)
    if page == "Overview":
        render_overview(metrics, monthly, province, products, location, weekday)
    elif page == "Sales":
        render_sales(metrics, monthly, weekday, province, products, location, activity)
    elif page == "Forecast":
        render_forecast(monthly, forecast)
    elif page == "Interactive Forecast":
        render_interactive_forecast()
    #else:
        #render_insights(df, data, file)
    # Insights is intentionally hidden for now.

 
if __name__ == "__main__":
    main()
