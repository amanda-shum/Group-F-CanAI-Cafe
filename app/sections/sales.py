import streamlit as st

from charts import (
    bar_option,
    heatmap_option,
    money,
    monthly_with_average_option,
    sales_matrix_heatmap_option,
    weekday_option,
)
from sections.shared import kpi_grid, product_label, section, show_chart


def product_rank_table(products):
    rows = []
    total_revenue = products["total_spent"].sum()
    for index, (_, product) in enumerate(products.sort_values("total_spent", ascending=False).iterrows(), start=1):
        share = product["total_spent"] / total_revenue if total_revenue else 0
        rows.append(
            "<tr>"
            f'<td class="rank-number">{index}</td>'
            f"<td>{product_label(product['item'])}</td>"
            f'<td class="rank-value">{money(product["total_spent"])}</td>'
            f'<td class="rank-share">{share:.0%}</td>'
            "</tr>"
        )
    st.markdown(
        (
            '<table class="rank-table">'
            "<thead><tr><th></th><th>Product</th><th>Revenue</th><th>Share</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        ),
        unsafe_allow_html=True,
    )



def render_sales(metrics, monthly, weekday, province, products, location, activity, sales_heatmaps):
    section("Sales", "Detailed revenue, product, region, and activity charts.")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    product_order = products.sort_values("total_spent", ascending=False)["item"].tolist()
    province_order = province.sort_values("total_spent", ascending=False)["province"].tolist()

    top_product = products.iloc[0] if not products.empty else None
    top_region = province.iloc[0] if not province.empty else None
    best_day = weekday.sort_values("total_spent", ascending=False).iloc[0] if not weekday.empty else None

    kpi_grid([
        ("Top product", money(top_product["total_spent"]) if top_product is not None else "N/A", product_label(top_product["item"]) if top_product is not None else "No product"),
        ("Top province", money(top_region["total_spent"]) if top_region is not None else "N/A", top_region["province"] if top_region is not None else "No region"),
        ("Strongest weekday", best_day["weekday"] if best_day is not None else "N/A", money(best_day["total_spent"]) if best_day is not None else "No revenue"),
        ("Avg daily sales", money(metrics["average_daily"]), "Actual sales per active day"),
    ], 4)


    with st.container(border=True):
        section("Sales trend", "Monthly revenue with the filtered-period average.")
        show_chart(monthly_with_average_option(monthly), "sales_monthly_revenue", "360px")

    left, right = st.columns(2)
    with left, st.container(border=True, height=360):
        section("Best sales days", "Revenue and order count by weekday.")
        show_chart(weekday_option(weekday), "sales_weekday_sales", "300px")
    with right, st.container(border=True, height=360):
        section("Activity heatmap", "Order volume by weekday and month.")
        show_chart(heatmap_option(activity), "sales_activity_heatmap", "300px")

    left, right = st.columns(2)
    with left, st.container(border=True, height=360):
        section("Top products", "Revenue by product.")
        product_rank_table(products)
    with right, st.container(border=True, height=360):
        section("Revenue by province", "Regions ranked by revenue.")
        show_chart(bar_option(province, "province", horizontal=False), "sales_province_bar", "300px")

        section("Sales heatmaps", "Compare product, province, month, and weekday sales patterns.")

    with st.container(border=True):
        section(
            "Item popularity by province",
            "Each cell shows the item share of that province's total sales.",
        )
        show_chart(
            sales_matrix_heatmap_option(
                sales_heatmaps["province_item"],
                x_col="item",
                y_col="province",
                value_col="province_sales_share",
                value_name="Share of province sales",
                value_format="percent",
                x_order=product_order,
                y_order=province_order,
                sort_x_by_total=False,
                sort_y_by_total=False,
                y_sort_col="province_total_sales",
            ),
            "sales_item_popularity_by_province_heatmap",
            "360px",
        )

    left, right = st.columns(2)

    with left, st.container(border=True):
        section("Item sales by month", "Monthly sales split by product.")
        show_chart(
            sales_matrix_heatmap_option(
                sales_heatmaps["item_month"],
                x_col="month_label",
                y_col="item",
                value_col="total_spent",
                value_name="Item sales by month",
                value_format="money",
                x_order=month_order,
                sort_x_by_total=False,
                sort_y_by_total=True,
            ),
            "sales_item_by_month_heatmap",
            "360px",
        )

    with right, st.container(border=True):
        section("Province sales by month", "Monthly sales split by province.")
        show_chart(
            sales_matrix_heatmap_option(
                sales_heatmaps["province_month"],
                x_col="month_label",
                y_col="province",
                value_col="total_spent",
                value_name="Province sales by month",
                value_format="money",
                x_order=month_order,
                sort_x_by_total=False,
                sort_y_by_total=True,
            ),
            "sales_province_by_month_heatmap",
            "360px",
        )

    left, right = st.columns(2)

    with left, st.container(border=True):
        section("Item sales by day of week", "Weekday sales split by product.")
        show_chart(
            sales_matrix_heatmap_option(
                sales_heatmaps["item_weekday"],
                x_col="weekday_full",
                y_col="item",
                value_col="total_spent",
                value_name="Item sales by weekday",
                value_format="money",
                x_order=weekday_order,
                sort_x_by_total=False,
                sort_y_by_total=True,
            ),
            "sales_item_by_weekday_heatmap",
            "340px",
        )

    with right, st.container(border=True):
        section("Province sales by day of week", "Weekday sales split by province.")
        show_chart(
            sales_matrix_heatmap_option(
                sales_heatmaps["province_weekday"],
                x_col="weekday_full",
                y_col="province",
                value_col="total_spent",
                value_name="Province sales by weekday",
                value_format="money",
                x_order=weekday_order,
                sort_x_by_total=False,
                sort_y_by_total=True,
            ),
            "sales_province_by_weekday_heatmap",
            "340px",
        )
