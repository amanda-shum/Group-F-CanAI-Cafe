"""
Each function returns a python dictionary that streamlit-echarts can render. 

WILL NEED TO UPDATE both forecast functions to take in data from the forecast model results!!!
"""
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
            "yAxis": {"type": "category", "data": labels[::-1], "axisLabel": {"color": COLORS["secondary_text"]}},
            "series": [{"type": "bar", "data": values[::-1], "barWidth": "58%", "itemStyle": {"borderRadius": [0, 6, 6, 0]}}],
        })
    else:
        opt.update({
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["secondary_text"]}},
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
            "label": {"fontSize": 14, "fontWeight": 700, "formatter": "{b}\n{d}%"},
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
        "visualMap": {"min": 0, "max": max([point[2] for point in data] or [1]), "orient": "horizontal", "left": "center", "bottom": 0, "inRange": {"color": [COLORS["heatmap_low"], COLORS["chart_soft"], COLORS["chart_primary"], COLORS["chart_deep"]]}},
        "series": [{"type": "heatmap", "data": data, "label": {"show": True, "color": COLORS["primary_text"]}}],
    }


def build_average_forecast(monthly):
    """Build PLACEHOLDER forecast values until model output is connected."""
    if len(monthly) < 3:
        return None
    recent_avg = float(monthly.tail(3)["total_spent"].mean())
    latest = float(monthly["total_spent"].iloc[-1])
    history = monthly.tail(6).copy()
    placeholder_values = history["total_spent"].rolling(3, min_periods=1).mean().round(2).tolist()
    return {
        "latest": latest,
        "recent_avg": recent_avg,
        "next_month": recent_avg,
        "next_quarter": recent_avg * 3,
        "forecast_months": history["month"].tolist(),
        "forecast_values": placeholder_values,
        "change": (recent_avg - latest) / latest if latest else 0,
    }


def forecast_option(monthly, forecast):
    """Compare actual revenue with placeholder forecast revenue for each month."""
    history = monthly.tail(6).copy()
    labels = history["month"].dt.strftime("%b").tolist()
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": COLORS["secondary_text"]}},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["secondary_text"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["secondary_text"]}, "splitLine": {"lineStyle": {"color": COLORS["chart_grid"]}}},
        "series": [
            {"name": "Actual", "type": "bar", "barWidth": "36%", "data": history["total_spent"].round(2).tolist(), "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_primary"]}},
            {"name": "Forecast", "type": "bar", "barWidth": "36%", "data": forecast["forecast_values"], "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["chart_secondary"]}},
        ],
    })
    return opt
