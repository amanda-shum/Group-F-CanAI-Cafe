import streamlit as st

from charts import donut_option, money, monthly_option
from sections.shared import kpi_grid, product_label, section, show_chart


def kpi_strip(metrics):
    kpi_grid([
        ("Revenue", money(metrics["total"]), "Filtered sales"),
        ("Orders", f"{metrics['transactions']:,.0f}", "Unique transactions"),
        ("Avg order", money(metrics["aov"]), "Revenue per transaction"),
        ("Top province", metrics["top_province"], "Best region"),
        ("Top item", product_label(metrics["top_item"]), "Best product"),
    ], 5)


def summary_row(title, value, note, share=None):
    bar = ""
    if share is not None:
        width = max(0, min(100, share * 100))
        bar = f'<div class="summary-bar"><div class="summary-fill" style="width:{width:.0f}%"></div></div>'
    st.markdown(
        (
            '<div class="summary-row">'
            '<div class="summary-line">'
            f"<span>{title}</span>"
            f"<span>{value}</span>"
            "</div>"
            f'<div class="summary-meta">{note}</div>'
            f"{bar}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def product_rank_table(products):
    rows = []
    for index, (_, product) in enumerate(products.iterrows(), start=1):
        rows.append(
            "<tr>"
            f'<td class="rank-number">{index}</td>'
            f"<td>{product_label(product['item'])}</td>"
            f'<td class="rank-value">{money(product["total_spent"])}</td>'
            "</tr>"
        )
    st.markdown(f'<table class="rank-table"><tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)


def best_days_podium(best_days):
    cards = []
    labels = ["First", "Second", "Third"]
    for label, (_, day) in zip(labels, best_days.iterrows()):
        cards.append(
            '<div class="podium-card">'
            f'<div class="podium-rank">{label}</div>'
            f'<div class="podium-day">{day["weekday"]}</div>'
            f'<div class="podium-value">{money(day["total_spent"])}</div>'
            f'<div class="podium-meta">{day["transactions"]:,.0f} orders</div>'
            "</div>"
        )
    st.markdown(f'<div class="podium-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def best_day_card(best_day):
    st.markdown(
        (
            '<div class="podium-card">'
            '<div class="podium-rank">Best day</div>'
            f'<div class="podium-day">{best_day["weekday"]}</div>'
            f'<div class="podium-value">{money(best_day["total_spent"])}</div>'
            f'<div class="podium-meta">{best_day["transactions"]:,.0f} orders</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_overview(metrics, monthly, province, products, location, weekday):
    section("Overview", "Main results for the current filters.")
    kpi_strip(metrics)

    with st.container(border=True):
        section("Monthly revenue", "Revenue by month.")
        show_chart(monthly_option(monthly), "overview_monthly_revenue", "340px")

    left, middle, right = st.columns(3)
    with left, st.container(border=True, height=320):
        section("Top products", "Top 3 products by revenue.")
        top_products = products.sort_values("total_spent", ascending=False).head(3)
        product_rank_table(top_products)

    with middle, st.container(border=True, height=320):
        section("Order type", "Revenue by visit mode.")
        show_chart(donut_option(location, "location"), "overview_location_donut", "250px")

    with right, st.container(border=True, height=320):
        section("Best sales day", "Highest revenue weekday.")
        best_day = weekday.sort_values("total_spent", ascending=False).iloc[0]
        best_day_card(best_day)
