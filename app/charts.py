"""
Each function returns a python dictionary that streamlit-echarts can render. 

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
        "color": [COLORS["green"], COLORS["gold"], COLORS["rust"], COLORS["sage"], COLORS["espresso"]],
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": "Arial", "color": COLORS["text"]},
        "tooltip": {"trigger": "axis", "backgroundColor": COLORS["panel"], "borderColor": COLORS["border"]},
        "grid": {"left": 42, "right": 18, "top": 32, "bottom": 34, "containLabel": True},
    }


def monthly_option(monthly):
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0, "textStyle": {"color": COLORS["muted"]}},
        "xAxis": {"type": "category", "data": monthly["month"].dt.strftime("%b").tolist(), "axisLabel": {"color": COLORS["muted"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["muted"]}, "splitLine": {"lineStyle": {"color": COLORS["grid"]}}},
        "series": [{
            "name": "Revenue",
            "type": "line",
            "smooth": True,
            "symbolSize": 8,
            "data": monthly["total_spent"].round(2).tolist(),
            "lineStyle": {"width": 3, "color": COLORS["green"]},
            "areaStyle": {"opacity": 0.08},
        }],
    })
    return opt


def weekday_option(weekday):
    data = weekday.to_dict("records")
    opt = base_chart()
    opt.update({
        "legend": {"top": 0, "textStyle": {"color": COLORS["muted"]}},
        "xAxis": {"type": "category", "data": [d["weekday"] for d in data], "axisLabel": {"color": COLORS["muted"]}},
        "yAxis": [
            {"type": "value", "name": "Revenue", "axisLabel": {"formatter": "${value}", "color": COLORS["muted"]}, "splitLine": {"lineStyle": {"color": COLORS["grid"]}}},
            {"type": "value", "name": "Orders", "axisLabel": {"color": COLORS["muted"]}, "splitLine": {"show": False}},
        ],
        "series": [
            {"name": "Revenue", "type": "bar", "data": [round(d["total_spent"], 2) for d in data], "barWidth": "52%", "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["sage"]}},
            {"name": "Orders", "type": "line", "yAxisIndex": 1, "smooth": True, "symbolSize": 8, "data": [int(d["transactions"]) for d in data], "lineStyle": {"width": 3, "color": COLORS["gold"]}},
        ],
    })
    return opt


def bar_option(df, label_col, value_col="total_spent", horizontal=True):
    labels = df[label_col].tolist()
    values = df[value_col].round(2).tolist()
    opt = base_chart()
    if horizontal:
        opt.update({
            "xAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["muted"]}, "splitLine": {"lineStyle": {"color": COLORS["grid"]}}},
            "yAxis": {"type": "category", "data": labels[::-1], "axisLabel": {"color": COLORS["muted"]}},
            "series": [{"type": "bar", "data": values[::-1], "barWidth": "58%", "itemStyle": {"borderRadius": [0, 6, 6, 0]}}],
        })
    else:
        opt.update({
            "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["muted"]}},
            "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["muted"]}, "splitLine": {"lineStyle": {"color": COLORS["grid"]}}},
            "series": [{"type": "bar", "data": values, "barWidth": "55%", "itemStyle": {"borderRadius": [6, 6, 0, 0]}}],
        })
    return opt


def donut_option(df, name_col):
    data = [{"name": r[name_col], "value": round(r["total_spent"], 2)} for _, r in df.iterrows()]
    return {
        "color": [COLORS["green"], COLORS["gold"], COLORS["rust"], COLORS["sage"]],
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": COLORS["muted"]}},
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
        "xAxis": {"type": "category", "data": x, "splitArea": {"show": True}, "axisLabel": {"color": COLORS["muted"]}},
        "yAxis": {"type": "category", "data": y, "splitArea": {"show": True}, "axisLabel": {"color": COLORS["muted"]}},
        "visualMap": {"min": 0, "max": max([point[2] for point in data] or [1]), "orient": "horizontal", "left": "center", "bottom": 0, "inRange": {"color": ["#FBF7EF", COLORS["sage"], COLORS["green"], COLORS["espresso"]]}},
        "series": [{"type": "heatmap", "data": data, "label": {"show": True, "color": COLORS["text"]}}],
    }


def build_average_forecast(monthly):
    """Use recent monthly average as a simple, explainable forecast baseline."""
    if len(monthly) < 3:
        return None
    recent_avg = float(monthly.tail(3)["total_spent"].mean())
    latest = float(monthly["total_spent"].iloc[-1])
    future_months = pd.date_range(monthly["month"].max() + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    return {
        "latest": latest,
        "recent_avg": recent_avg,
        "next_month": recent_avg,
        "next_quarter": recent_avg * 3,
        "future_months": future_months,
        "future_values": [recent_avg] * 3,
        "change": (recent_avg - latest) / latest if latest else 0,
    }


def forecast_option(monthly, forecast):
    """Compare recent actual months with the next three forecast months."""
    history = monthly.tail(6).copy()
    labels = history["month"].dt.strftime("%b").tolist() + [m.strftime("%b") for m in forecast["future_months"]]
    opt = base_chart()
    opt.update({
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": COLORS["muted"]}},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"color": COLORS["muted"]}},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "color": COLORS["muted"]}, "splitLine": {"lineStyle": {"color": COLORS["grid"]}}},
        "series": [
            {"name": "Actual", "type": "bar", "barWidth": "48%", "data": history["total_spent"].round(2).tolist() + [None] * 3, "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["green"]}},
            {"name": "Forecast", "type": "bar", "barWidth": "48%", "data": [None] * len(history) + [round(v, 2) for v in forecast["future_values"]], "itemStyle": {"borderRadius": [6, 6, 0, 0], "color": COLORS["gold"]}, "label": {"show": True, "position": "top", "formatter": "${c}", "color": COLORS["text"]}},
        ],
    })
    return opt
