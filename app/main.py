import streamlit as st
import pandas as pd

st.set_page_config(page_title="CanAI Coffee Shop - Dashboard", layout="wide")

# Title
st.title("☕ CanAI Coffee Shop - Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
date = st.sidebar.date_input("Select Date Range")
province = st.sidebar.selectbox("Select Province", ["All", "BC", "MB", "SK", "NF"])

# KPI Row
col1, col2, col3, col4 = st.columns(4)

#placeholder values for now
col1.metric("Total Revenue", "$12,430", "+5%") 
col2.metric("Transactions", "1,240", "+3%")
col3.metric("Avg Order Value", "$10.02", "+2%")
col4.metric("Top Province", "BC")

st.divider()

# Charts placeholder
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Trend")
    st.line_chart([10, 20, 15, 30, 40])

with col2:
    st.subheader("Forecast")
    st.line_chart([40, 45, 50, 55, 60])

st.divider()

# Bottom section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regional Performance")
    st.bar_chart([100, 200, 150])

with col2:
    st.subheader("Product Performance")
    st.bar_chart([50, 80, 30])

# Insights
st.subheader("💡 Insights")
st.write("- BC generates the highest revenue")
st.write("- Coffee is the top-selling item")

st.subheader("🚀 Recommendations")
st.write("- Increase inventory in BC")
st.write("- Promote low-performing items")