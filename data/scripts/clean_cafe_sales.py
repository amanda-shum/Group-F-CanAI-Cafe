#!/usr/bin/env python3
"""
Clean CanAI Cafe 2023 sales transactions.

This script consolidates the cleaning logic from the uploaded notebooks:
- data_cleaning_v1(2).ipynb
- data_cleaning_standardisation.ipynb
- data_cleaning_handling_missings.ipynb

Example:
    python clean_cafe_sales.py \
        --input "data/raw/CanAI Cafe 2023 Sales Information.csv" \
        --output "data/processed/cleaned_transactions.csv"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROVINCE_MAPPING = {
    # British Columbia
    "british columbia": "British Columbia",
    "britishcolumbia": "British Columbia",
    "british columba": "British Columbia",
    "british columbi": "British Columbia",
    "bc": "British Columbia",
    "b c": "British Columbia",
    # Newfoundland and Labrador
    "newfoundland": "Newfoundland and Labrador",
    "new foundland": "Newfoundland and Labrador",
    "newfoundland and labrador": "Newfoundland and Labrador",
    "nfld": "Newfoundland and Labrador",
    "nl": "Newfoundland and Labrador",
    "newfoundlan": "Newfoundland and Labrador",
    # Manitoba
    "manitoba": "Manitoba",
    "manitba": "Manitoba",
    "manitobaa": "Manitoba",
    "manitob": "Manitoba",
    "mb": "Manitoba",
    # Saskatchewan
    "saskatchewan": "Saskatchewan",
    "saskatchewn": "Saskatchewan",
    "sasktchewan": "Saskatchewan",
    "saskatchewa": "Saskatchewan",
    "sask": "Saskatchewan",
    "sk": "Saskatchewan",
    # Ontario
    "ontario": "Ontario",
    "ontaroi": "Ontario",
    "ontairo": "Ontario",
    "ont": "Ontario",
    "on": "Ontario",
}

ITEM_MAPPING = {
    # Coffee
    "coffee": "Coffee",
    "cofee": "Coffee",
    "coffe": "Coffee",
    "c0ffee": "Coffee",
    # Tea
    "tea": "Tea",
    "tee": "Tea",
    # Sandwich
    "sandwich": "Sandwich",
    "sandwhich": "Sandwich",
    # Cookie
    "cookie": "Cookie",
    # Donut
    "donut": "Donut",
    "doughnut": "Donut",
    "donutt": "Donut",
    # Other items
    "refresher": "Refresher",
    "salad": "Salad",
    "juice": "Juice",
    "juic": "Juice",
    "juicee": "Juice",
}

EXPECTED_INPUT_COLUMNS = {
    "Transaction ID",
    "Item",
    "Quantity",
    "Price Per Unit",
    "Total Spent",
    "Payment Method",
    "Location",
    "Transaction Date",
    "Province",
}


def replace_blank_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty or whitespace-only cells to NaN."""
    return df.replace(r"^\s*$", np.nan, regex=True)


def clean_province(value: Any) -> Any:
    """Standardise province names and common misspellings/abbreviations."""
    if pd.isna(value):
        return np.nan

    cleaned = str(value).strip().lower().replace(".", "")
    cleaned = " ".join(cleaned.split())
    return PROVINCE_MAPPING.get(cleaned, cleaned.title())


def clean_item(value: Any) -> Any:
    """Standardise item names and common misspellings."""
    if pd.isna(value):
        return np.nan

    cleaned = str(value).strip().lower()
    if cleaned == "":
        return np.nan

    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace("-", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return ITEM_MAPPING.get(cleaned, cleaned.title())


def random_sample_from_series(series: pd.Series, rng: np.random.Generator) -> Any:
    """Randomly sample one observed value from a Series using its empirical distribution."""
    observed = series.dropna()
    if observed.empty:
        return np.nan

    probs = observed.value_counts(normalize=True)
    return rng.choice(probs.index.to_numpy(), p=probs.values)


def conditional_random_impute(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Impute target_col by random sampling from observed values within group_col.

    If the row's group has no observed values for target_col, fall back to the
    overall observed distribution for target_col.
    """
    df = df.copy()
    missing_mask = df[target_col].isna()

    for idx in df.loc[missing_mask].index:
        group_value = df.at[idx, group_col]

        if pd.notna(group_value):
            group_series = df.loc[
                (df[group_col] == group_value) & df[target_col].notna(),
                target_col,
            ]
            sampled_value = random_sample_from_series(group_series, rng)
            if pd.notna(sampled_value):
                df.at[idx, target_col] = sampled_value
                continue

        df.at[idx, target_col] = random_sample_from_series(df[target_col], rng)

    return df


def make_json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serialisable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def clean_transactions(raw_df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run the full cleaning workflow and return cleaned data, audit summary, and dropped rows."""
    missing_columns = sorted(EXPECTED_INPUT_COLUMNS - set(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    df = raw_df.copy()
    rows_initial = len(df)

    # Step 1: convert blanks to NaN.
    df = replace_blank_strings(df)
    missing_after_blank_conversion = df.isna().sum().sort_values(ascending=False).to_dict()

    # Step 2: standardise categorical columns.
    df["Province_cleaned"] = df["Province"].apply(clean_province)
    df["Item_cleaned"] = df["Item"].apply(clean_item)

    # Step 3: coerce numeric columns and impute Quantity from Total Spent / Price Per Unit.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").astype(float)
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")

    df["Quantity_imputed_flag"] = df["Quantity"].isna().astype(int)
    quantity_impute_mask = (
        df["Quantity"].isna()
        & df["Price Per Unit"].notna()
        & df["Total Spent"].notna()
        & (df["Price Per Unit"] != 0)
    )
    df.loc[quantity_impute_mask, "Quantity"] = (
        df.loc[quantity_impute_mask, "Total Spent"] / df.loc[quantity_impute_mask, "Price Per Unit"]
    )

    # Step 4: drop unused/invalid fields and rows that cannot support time-series modelling.
    df = df.drop(columns=["Payment Method"], errors="ignore")
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    rows_to_drop = df["Transaction Date"].isna() | df["Item_cleaned"].isna()
    dropped_rows = df.loc[rows_to_drop].copy()
    df = df.loc[~rows_to_drop].copy()

    # Step 5: retain cleaned categorical columns and impute remaining Province/Location gaps.
    df = df.drop(columns=["Province", "Item"], errors="ignore")
    df["Location_missing_flag"] = df["Location"].isna().astype(int)
    df["Province_missing_flag"] = df["Province_cleaned"].isna().astype(int)

    rng = np.random.default_rng(seed=seed)
    df = conditional_random_impute(df, "Province_cleaned", "Location", rng)
    df = conditional_random_impute(df, "Location", "Province_cleaned", rng)

    df = df.rename(columns={"Province_cleaned": "Province", "Item_cleaned": "Item"})

    # Step 6: sanity check item prices before dropping them.
    price_check = (
        df.groupby("Item")["Price Per Unit"]
        .agg(
            unique_price_count="nunique",
            unique_prices=lambda x: [make_json_safe(v) for v in sorted(x.dropna().unique())],
            row_count="count",
        )
        .reset_index()
        .sort_values(by="unique_price_count", ascending=False)
    )

    # Step 7: final modelling dataset keeps Total Spent but drops Quantity and Price Per Unit.
    df = df.drop(columns=["Quantity", "Price Per Unit", "Quantity_imputed_flag"], errors="ignore")

    # Match notebook-style date output in the CSV.
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.date

    audit_summary = {
        "rows_initial": rows_initial,
        "rows_final": int(len(df)),
        "rows_dropped_missing_date_or_item": int(rows_to_drop.sum()),
        "percentage_dropped": round(float(rows_to_drop.mean() * 100), 2),
        "quantity_values_imputed": int(quantity_impute_mask.sum()),
        "location_values_imputed": int(df["Location_missing_flag"].sum()),
        "province_values_imputed": int(df["Province_missing_flag"].sum()),
        "missing_after_blank_conversion": {k: make_json_safe(v) for k, v in missing_after_blank_conversion.items()},
        "final_missing_values": {k: make_json_safe(v) for k, v in df.isna().sum().sort_values(ascending=False).items()},
        "final_shape": [int(df.shape[0]), int(df.shape[1])],
        "final_columns": list(df.columns),
        "location_distribution": {k: float(v) for k, v in df["Location"].value_counts(normalize=True).items()},
        "province_distribution": {k: float(v) for k, v in df["Province"].value_counts(normalize=True).items()},
        "price_consistency_by_item": price_check.to_dict(orient="records"),
    }

    return df, audit_summary, dropped_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean CanAI Cafe sales transaction data.")
    parser.add_argument("--input", required=True, help="Path to the raw sales CSV file.")
    parser.add_argument("--output", required=True, help="Path where cleaned_transactions.csv should be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for conditional random imputation.")
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional path for a JSON audit summary. Defaults to <output_stem>_audit.json.",
    )
    parser.add_argument(
        "--dropped-rows-output",
        default=None,
        help="Optional path for rows dropped because of missing Transaction Date or Item.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit_output) if args.audit_output else output_path.with_name(f"{output_path.stem}_audit.json")

    raw_df = pd.read_csv(input_path)
    cleaned_df, audit_summary, dropped_rows = clean_transactions(raw_df, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

    if args.dropped_rows_output:
        dropped_path = Path(args.dropped_rows_output)
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_rows.to_csv(dropped_path, index=False)

    print(f"Cleaned CSV written to: {output_path}")
    print(f"Audit summary written to: {audit_path}")
    print(f"Final dataset shape: {cleaned_df.shape}")
    print(f"Rows dropped because of missing Transaction Date or Item: {len(dropped_rows)}")


if __name__ == "__main__":
    main()
