import pandas as pd
import streamlit as st

import charts
from charts import build_average_forecast
from data import load_data
from sections import (
    ITEM_BADGES,
    render_forecast,
    render_insights,
    render_overview,
    render_sales,
)

PAGE_TITLE = "CanAI Cafe Dashboard!"

LIGHT_COLORS = {
    "bg": "#FFFDF8",
    "panel": "#FFFFFF",
    "text": "#2B2118",
    "muted": "#756A5E",
    "grid": "#EDE6DB",
    "border": "#E2D7C9",
    "green": "#6F8F72",
    "sage": "#A7B99E",
    "gold": "#C79A4B",
    "rust": "#B86F52",
    "espresso": "#3E2B20",
}
DARK_COLORS = {
    "bg": "#171512",
    "panel": "#211E19",
    "text": "#F6EFE3",
    "muted": "#B9AB99",
    "grid": "#3A332B",
    "border": "#443B31",
    "green": "#95B889",
    "sage": "#71896C",
    "gold": "#D4AA5E",
    "rust": "#CF7D62",
    "espresso": "#F0D9B5",
}


def apply_theme(colors):
    """Keep Streamlit native theme and then add some style"""
    st.markdown(f"""
    <style>
    .stApp {{ background:var(--background-color); color:var(--text-color); }}
    .block-container {{ max-width:1420px; padding-top:1.4rem; padding-bottom:2.5rem; }}
    [data-testid="stVerticalBlock"] {{ gap:1.35rem; }}
    [data-testid="stHorizontalBlock"] {{ gap:1.35rem; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{ margin-top:.25rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:var(--secondary-background-color);
        border-color:{colors["border"]};
        box-shadow:0 10px 24px rgba(0,0,0,.04);
    }}
    .title {{ font-size:2.1rem; font-weight:800; line-height:1.05; }}
    .subtitle {{ color:var(--text-color); opacity:.72; font-size:1rem; margin:.3rem 0 1.25rem; }}
    [data-testid="stSidebar"] {{ background:#26262E; border-right:1px solid #34343D; }}
    [data-testid="stSidebar"] * {{ color:#ECE8E2; }}
    [data-testid="stSidebar"] .brand {{ display:flex; gap:.85rem; align-items:center; padding:.9rem .25rem 1.05rem; }}
    [data-testid="stSidebar"] .brand-mark {{
        width:2.6rem; height:2.6rem; border-radius:.75rem;
        display:flex; align-items:center; justify-content:center;
        background:#D98C32; color:#FFFFFF; font-weight:850; font-size:1.15rem;
    }}
    [data-testid="stSidebar"] .brand-title {{ font-size:1.1rem; font-weight:850; line-height:1.05; }}
    [data-testid="stSidebar"] .brand-subtitle {{ color:#8E8D96; font-size:.78rem; font-weight:800; letter-spacing:.16em; margin-top:.25rem; }}
    [data-testid="stSidebar"] hr {{ border-color:#383841; margin:1rem 0; }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.35rem; }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding:.78rem .85rem; border-radius:.45rem; margin:.12rem 0;
        color:#AAA8B0; font-weight:760;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background:#3A302D; border-left:3px solid #D98C32;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {{ color:#F0A13A; }}
    [data-testid="stSidebar"] div[role="radiogroup"] input {{ display:none; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background:#34343D; border-color:#555560; border-radius:.42rem;
    }}
    [data-testid="stSidebar"] label {{ color:#AAA8B0 !important; font-weight:760; }}
    .sidebar-label {{ color:#6F6E78; font-size:.8rem; font-weight:850; letter-spacing:.18em; margin:.3rem 0 .75rem; }}
    .sidebar-footer {{ color:#6F6E78; font-family:monospace; font-weight:700; font-size:.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #383841; }}
    .section-heading {{ margin:.2rem 0 .65rem; }}
    .section-title {{ font-size:1.06rem; font-weight:760; margin-bottom:.22rem; }}
    .note {{ color:var(--text-color); opacity:.68; font-size:.86rem; margin-bottom:.45rem; }}
    .kpi {{ border:1px solid {colors["border"]}; border-radius:8px; padding:.72rem .78rem; background:var(--secondary-background-color); }}
    .kpi-label {{ color:var(--text-color); opacity:.68; font-size:.72rem; font-weight:750; text-transform:uppercase; letter-spacing:.04em; }}
    .kpi-value {{ font-size:1.34rem; font-weight:820; margin-top:.18rem; }}
    .placeholder {{ border:1px dashed {colors["border"]}; border-radius:8px; background:var(--background-color); padding:.75rem .85rem; margin:.45rem 0; font-size:.96rem; }}
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
        labels = {"Overview": "Overview", "Sales": "Sales", "Forecast": "Forecast", "Insights": "Insights"}
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
        "items_sold": len(data),
        "top_province": revenue.groupby("province")["total_spent"].sum().idxmax() if not revenue.empty else "N/A",
        "top_item": revenue.groupby("item")["total_spent"].sum().idxmax() if not revenue.empty else "N/A",
    }
    metrics["aov"] = metrics["total"] / metrics["transactions"] if metrics["transactions"] else 0

    monthly = trend.groupby("month", as_index=False)["total_spent"].sum().sort_values("month")
    weekday = (
        trend.assign(weekday=pd.Categorical(trend["transaction_date"].dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ordered=True))
        .groupby("weekday", observed=False)
        .agg(total_spent=("total_spent", "sum"), transactions=("transaction_id", "nunique"))
        .reset_index()
    )
    province = revenue.groupby("province", as_index=False)["total_spent"].sum().sort_values("total_spent", ascending=False)
    products = revenue.groupby("item", as_index=False)["total_spent"].sum().sort_values("total_spent", ascending=False)
    products["label"] = products["item"].map(lambda x: f"[{ITEM_BADGES.get(x, '?')}] {x}")
    location = revenue.groupby("location", as_index=False)["total_spent"].sum()
    activity = (
        trend.assign(
            weekday=pd.Categorical(trend["transaction_date"].dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            month=trend["transaction_date"].dt.strftime("%b"),
        )
        .groupby(["weekday", "month"], observed=False)
        .size()
        .reset_index(name="transactions")
    )
    return metrics, monthly, build_average_forecast(monthly), weekday, province, products, location, activity


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="B", layout="wide")
    colors = DARK_COLORS if st.context.theme.get("type") == "dark" else LIGHT_COLORS
    charts.set_colors(colors)
    apply_theme(colors)

    df, file, error = load_data()
    if error:
        st.error(error)
        st.stop()

    st.markdown(f'<div class="title">{PAGE_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Summary placeholder :)</div>', unsafe_allow_html=True)

    page, data = sidebar_controls(df)
    prepared = prepare_dashboard_data(data)
    if prepared is None:
        st.warning("No dated revenue rows are available for trend charts.")
        return

    metrics, monthly, forecast, weekday, province, products, location, activity = prepared
    if page == "Overview":
        render_overview(metrics, monthly, province)
    elif page == "Sales":
        render_sales(monthly, weekday, province, products, location, activity)
    elif page == "Forecast":
        render_forecast(monthly, forecast)
    else:
        render_insights(df, data, file)


if __name__ == "__main__":
    main()
