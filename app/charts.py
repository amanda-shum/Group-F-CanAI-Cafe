"""ECharts option builders for the dashboard."""
import pandas as pd

COLORS = {}


def set_colors(colors):
    global COLORS
    COLORS = colors


def money(value):
    return f"${value:,.0f}" if pd.notna(value) else "$0"


def base_chart():
    return {
        "color": [COLORS["chart_primary"], COLORS["chart_secondary"], COLORS["chart_tertiary"], COLORS["chart_soft"], COLORS["chart_deep"]],
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": "Arial", "color": COLORS["primary_text"]},
        "tooltip": {"trigger": "axis", "backgroundColor": COLORS["card_background"], "borderColor": COLORS["card_border"]},
        "grid": {"left": 42, "right": 18, "top": 32, "bottom": 34, "containLabel": True},
    }


def monthly_option(monthly):
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "xAxis": {"type": "category", "data": monthly["month"].dt.strftime("%b").tolist(), "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
        "series": [{
            "name": "Revenue",
            "type": "line",
            "smooth": True,
            "symbolSize": 8,
            "data": monthly["total_spent"].round(2).tolist(),
            "lineStyle": {"width": 3, "color": COLORS["chart_primary"]},
            "areaStyle": {"opacity": 0.08},
        }],
    })
    return opt


def weekday_option(weekday):
    data = weekday.to_dict("records")
    opt = base_chart()
    opt.update({
        "legend": {"top": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "xAxis": {"type": "category", "data": [d["weekday"] for d in data], "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": [
            {"type": "value", "name": "Revenue", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
            {"type": "value", "name": "Orders", "axisLabel": {"color": COLORS["secondary_text"]}, "splitLine": {"show": False}},
        ],
        "series": [
            {"name": "Revenue", "type": "bar", "data": [round(d["total_spent"], 2) for d in data], "barWidth": "52%", "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_soft"]}},
            {"name": "Orders", "type": "line", "yAxisIndex": 1, "smooth": True, "symbolSize": 8, "data": [int(d["transactions"]) for d in data], "lineStyle": {"width": 3, "color": COLORS["chart_secondary"]}},
        ],
    })
    return opt


def bar_option(df, label_col, value_col="total_spent", horizontal=True):
    labels = df[label_col].tolist()
    values = df[value_col].round(2).tolist()
    opt = base_chart()
    if horizontal:
        opt.update({
            "xAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
            "yAxis": {"type": "category", "data": labels[::-1], "axisLabel": {"color": COLORS["secondary_text"], "interval": 0}},
            "series": [{"type": "bar", "data": values[::-1], "barWidth": "58%", "itemStyle": {"borderRadius": [0, 6, 6, 0]}}],
        })
    else:
        opt.update({
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["secondary_text"], "interval":0, "rotate":25}},
            "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
            "series": [{"type": "bar", "data": values, "barWidth": "55%", "itemStyle": {"borderRadius": [6, 6, 0, 0]}}],
        })
    return opt


def donut_option(df, name_col):
    data = [{"name": r[name_col], "value": round(r["total_spent"], 2)} for _, r in df.iterrows()]
    return {
        "color": [COLORS["chart_primary"], COLORS["chart_secondary"], COLORS["chart_tertiary"], COLORS["chart_soft"]],
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "series": [{
            "type": "pie",
            "radius": ["48%", "72%"],
            "center": ["50%", "44%"],
            "avoidLabelOverlap": True,
            "label": {"show": False},
            "emphasis": {"label": {"show": True, "fontSize": 14, "fontWeight": 700, "formatter": "{b}\n{d}%"}},
            "data": data,
        }],
    }


def heatmap_option(activity):
    x = activity["month"].drop_duplicates().tolist()
    y = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    data = [[x.index(r["month"]), y.index(r["weekday"]), int(r["transactions"])] for _, r in activity.iterrows()]
    return {
        "tooltip": {"position": "top"},
        "grid": {"left": 42, "right": 18, "top": 20, "bottom": 40},
        "xAxis": {"type": "category", "data": x, "splitArea": {"show": True}, "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "category", "data": y, "splitArea": {"show": True}, "axisLabel": {"color": COLORS["secondary_text"]}},
        "visualMap": {"show": False, "min": 0, "max": max([point[2] for point in data] or [1]), "inRange": {"color": [COLORS["heatmap_low"], COLORS["chart_soft"], COLORS["chart_primary"], COLORS["chart_deep"]]}},
        "series": [{"type": "heatmap", "data": data, "label": {"show": False}}],
    }


def build_forecast_summary(monthly, monthly_forecast, six_month_forecast, daily_forecast=None):
    """Prepare user-facing forecast metrics from reports generated by run_model.py."""
    if monthly_forecast is None or six_month_forecast is None or monthly_forecast.empty or six_month_forecast.empty:
        return None

    future_monthly = (
        six_month_forecast.groupby("month", as_index=False)["forecast"]
        .sum()
        .sort_values("month")
    )
    forecast_months = pd.concat([
        monthly_forecast[["month", "monthly_forecast"]],
        future_monthly.rename(columns={"forecast": "monthly_forecast"}),
    ], ignore_index=True)
    forecast_months = (
        forecast_months
        .dropna(subset=["month"])
        .drop_duplicates(subset=["month"], keep="first")
        .sort_values("month")
    )
    monthly_comparison = (
        forecast_months
        .merge(monthly[["month", "total_spent"]], on="month", how="left")
        .sort_values("month")
    )

    forecast_weekday = None
    if daily_forecast is not None and not daily_forecast.empty:
        forecast_weekday = (
            daily_forecast.dropna(subset=["weekday"])
            .groupby("weekday", observed=False)
            .agg(forecast=("forecast", "mean"), total_forecast=("forecast", "sum"))
            .reset_index()
        )

    next_month = float(future_monthly["forecast"].iloc[0]) if not future_monthly.empty else 0
    next_quarter = float(future_monthly.head(3)["forecast"].sum()) if not future_monthly.empty else 0
    six_month_total = float(future_monthly["forecast"].sum())
    average_daily = float(six_month_forecast["forecast"].mean())

    return {
        "monthly_comparison": monthly_comparison,
        "future_monthly": future_monthly,
        "next_month": next_month,
        "next_quarter": next_quarter,
        "six_month_total": six_month_total,
        "average_daily": average_daily,
        "forecast_weekday": forecast_weekday,
    }


def forecast_option(monthly, forecast):
    """Compare actual revenue with forecast revenue for overlapping months."""
    comparison = forecast["monthly_comparison"]
    labels = comparison["month"].dt.strftime("%b").tolist()
    actual_values = [
        None if pd.isna(value) else round(float(value), 2)
        for value in comparison["total_spent"].tolist()
    ]
    forecast_values = [
        None if pd.isna(value) else round(float(value), 2)
        for value in comparison["monthly_forecast"].tolist()
    ]
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
        "series": [
            {"name": "Actual", "type": "bar", "barWidth": "36%", "data": actual_values, "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_primary"]}},
            {"name": "Forecast", "type": "bar", "barWidth": "36%", "data": forecast_values, "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_secondary"]}},
        ],
    })
    return opt


def monthly_with_average_option(monthly):
    """Show monthly revenue with the filtered-period average as context."""
    average = float(monthly["total_spent"].mean()) if not monthly.empty else 0
    opt = monthly_option(monthly)
    opt["series"][0]["name"] = "Monthly revenue"
    opt["series"].append({
        "name": "Average",
        "type": "line",
        "symbol": "none",
        "data": [round(average, 2)] * len(monthly),
        "lineStyle": {"width": 2, "type": "dashed", "color": COLORS["chart_secondary"]},
    })
    return opt


def future_forecast_option(forecast):
    """Show the forecast outlook by month."""
    future = forecast["future_monthly"]
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "xAxis": {"type": "category", "data": future["month"].dt.strftime("%b").tolist(), "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
        "series": [{
            "name": "Forecast",
            "type": "bar",
            "barWidth": "52%",
            "data": future["forecast"].round(2).tolist(),
            "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_secondary"]},
        }],
    })
    return opt


def forecast_weekday_option(forecast):
    """Show expected average daily sales by weekday."""
    weekday = forecast.get("forecast_weekday")
    if weekday is None:
        weekday = pd.DataFrame({"weekday": [], "forecast": []})

    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 42, "right": 18, "top": 26, "bottom": 34, "containLabel": True},
        "xAxis": {"type": "category", "data": weekday["weekday"].astype(str).tolist(), "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
        "series": [{
            "name": "Average forecast",
            "type": "bar",
            "data": weekday["forecast"].round(2).tolist(),
            "barWidth": "52%",
            "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_soft"]},
        }],
    })
    return opt
def sales_matrix_heatmap_option(
    df,
    x_col,
    y_col,
    value_col,
    value_name="Total Sales",
    value_format="money",
    x_order=None,
    y_order=None,
    sort_x_by_total=False,
    sort_y_by_total=True,
    x_sort_col=None,
    y_sort_col=None,
):
    """
    Generic sorted heatmap builder.

    Sorting rules:
    - y-axis defaults to highest total at the top.
    - x-axis can either keep a logical order, like Jan-Dec or Mon-Sun,
      or sort by total sales.
    - y_sort_col is useful when the heatmap value is a percentage.
      Example: province-item share rows all add to 100%, so we sort provinces
      by actual province revenue instead.
    """

    if df is None or df.empty:
        x_labels = []
        y_labels = []
        data = []
        max_value = 1
    else:
        working = df.copy()
        working[x_col] = working[x_col].astype(str)
        working[y_col] = working[y_col].astype(str)
        working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0)

        matrix = working.pivot_table(
            values=value_col,
            index=y_col,
            columns=x_col,
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )

        def apply_order(existing_labels, order):
            if order is None:
                return existing_labels
            order = [str(label) for label in order]
            ordered = [label for label in order if label in existing_labels]
            remaining = [label for label in existing_labels if label not in ordered]
            return ordered + remaining

        if x_sort_col and x_sort_col in working.columns:
            x_labels = (
                working.groupby(x_col)[x_sort_col]
                .sum()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
        elif sort_x_by_total:
            x_labels = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
        else:
            x_labels = working[x_col].drop_duplicates().tolist()
            x_labels = apply_order(x_labels, x_order)

        if y_sort_col and y_sort_col in working.columns:
            y_labels = (
                working.groupby(y_col)[y_sort_col]
                .max()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
        elif sort_y_by_total:
            y_labels = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
        else:
            y_labels = working[y_col].drop_duplicates().tolist()
            y_labels = apply_order(y_labels, y_order)

        matrix = matrix.reindex(index=y_labels, columns=x_labels, fill_value=0)

        max_value = float(matrix.to_numpy().max()) if matrix.size else 1
        max_value = max(max_value, 1)

        data = []
        for y_index, y_label in enumerate(y_labels):
            for x_index, x_label in enumerate(x_labels):
                value = float(matrix.loc[y_label, x_label])

                if value_format == "percent":
                    label = f"{value:.1f}%"
                elif value_format == "money":
                    label = f"${value:,.0f}"
                else:
                    label = f"{value:,.0f}"

                data.append({
                    "value": [x_index, y_index, round(value, 2)],
                    "name": f"{y_label} / {x_label}: {label}",
                    "label": {"formatter": label},
                })

    return {
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": "Arial", "color": COLORS["primary_text"]},
        "tooltip": {
            "trigger": "item",
            "backgroundColor": COLORS["card_background"],
            "borderColor": COLORS["card_border"],
            "textStyle": {"color": COLORS["primary_text"]},
        },
        "grid": {
            "left": 105,
            "right": 20,
            "top": 18,
            "bottom": 52,
            "containLabel": True,
        },
        "xAxis": {
            "type": "category",
            "data": x_labels,
            "splitArea": {"show": True},
            "axisLabel": {
                "color": COLORS["secondary_text"],
                "interval": 0,
                "rotate": 35,
                "fontSize": 11,
            },
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "category",
            "data": y_labels,
            "inverse": True,
            "splitArea": {"show": True},
            "axisLabel": {
                "color": COLORS["secondary_text"],
                "fontSize": 11,
            },
            "axisTick": {"show": False},
        },
        "visualMap": {
            "show": False,
            "min": 0,
            "max": max_value,
            "inRange": {
                "color": [
                    COLORS["heatmap_low"],
                    COLORS["chart_soft"],
                    COLORS["chart_primary"],
                    COLORS["chart_deep"],
                ]
            },
        },
        "series": [{
            "name": value_name,
            "type": "heatmap",
            "data": data,
            "label": {
                "show": True,
                "fontSize": 10,
                "color": COLORS["primary_text"],
            },
        }],
    }