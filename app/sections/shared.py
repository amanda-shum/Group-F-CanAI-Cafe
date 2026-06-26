"""shared helpers!"""
import streamlit as st
from streamlit_echarts import st_echarts

from charts import money

ITEM_BADGES = {
    "Coffee": "☕",
    "Tea": "🍵",
    "Donut": "🍩",
    "Sandwich": "🥪",
    "Cookie": "🍪",
    "Juice": "🧃",
    "Refresher": "🧊",
    "Salad": "🥗",
    "Unknown": "•",
}


def product_label(item):
    return f"{ITEM_BADGES.get(item, '•')} {item}"


def kpi_grid(cards, columns):
    items = []
    for label, value, hint in cards:
        items.append(
            '<div class="kpi">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="note">{hint}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="kpi-grid" style="grid-template-columns:repeat({columns}, minmax(0, 1fr));">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )

def kpi_card(label, value, hint):
    st.markdown(
        (
            '<div class="kpi">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="note">{hint}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def section(title, note=None):
    html = f'<div class="section-heading"><div class="section-title">{title}</div>'
    if note:
        html += f'<div class="note">{note}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def show_chart(options, key, height="320px"):
    """Small wrapper so every ECharts render stays consistent and easy to find."""
    st_echarts(options=options, height=height, key=key)

