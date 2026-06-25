"""
This file reads the selected sales csv/xlsx file :)"
"""

from pathlib import Path
import re
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = {
    "transaction_id",
    "item",
    "total_spent",
    "location",
    "transaction_date",
    "province"
}

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_transactions.csv"


def clean_column_name(column):
    """Make clean team exports easier to connect without forcing exact header casing."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def read_data_file(file):
    if file.suffix.lower() == ".csv":
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_data():
    """Load the first CSV/Excel file found and prepare chart-ready typed columns."""
    file = DATA_FILE
    if not file.exists():
        return None, file, f"Data file not found: {file.name}. Update DATA_FILE in data.py"

    df = read_data_file(file).copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return None, file, (
            f"Missing required columns after alias mapping: {', '.join(missing)}. "
            
        )

    df["total_spent"] = pd.to_numeric(df["total_spent"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    for col in ["item", "province", "location"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    df["month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    return df, file, None
