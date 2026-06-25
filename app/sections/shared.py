"""shared helpers!"""
import streamlit as st
from streamlit_echarts import st_echarts

from charts import money

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

