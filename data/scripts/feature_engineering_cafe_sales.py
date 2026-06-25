#!/usr/bin/env python3
"""
Feature engineer CanAI Cafe sales transactions.

This script consolidates the feature engineering logic from feature_eng(2).ipynb.
It creates a complete daily Province-Item panel and adds calendar, lag, and rolling
sales features.

Example:
    python feature_engineering_cafe_sales.py \
        --input "data/processed/cleaned_transactions.csv" \
        --output "data/processed/feature_engineered_transactions.csv"

Optional model reproduction from the notebook:
    python feature_engineering_cafe_sales.py \
        --input "data/processed/cleaned_transactions.csv" \
        --output "data/processed/feature_engineered_transactions.csv" \
        --train-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Total Spent",
    "Transaction Date",
    "Province",
    "Item",
    "Location_missing_flag",
    "Province_missing_flag",
}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def build_feature_panel(
    cleaned_df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the daily Province-Item panel and time-series features."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(cleaned_df.columns))
    if missing_columns:
        raise ValueError(f"Input file is missing required columns: {missing_columns}")

    df = cleaned_df.copy()

    # The notebook drops Transaction ID before modelling/feature engineering.
    df = df.drop(columns=["Transaction ID"], errors="ignore")

    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    invalid_dates = int(df["Transaction Date"].isna().sum())
    df = df.dropna(subset=["Transaction Date", "Province", "Item"]).copy()
    df["Date"] = df["Transaction Date"].dt.normalize()

    if start_date is None:
        start = df["Date"].min()
    else:
        start = pd.to_datetime(start_date)

    if end_date is None:
        end = df["Date"].max()
    else:
        end = pd.to_datetime(end_date)

    all_dates = pd.date_range(start=start, end=end, freq="D")
    provinces = sorted(df["Province"].dropna().unique())
    items = sorted(df["Item"].dropna().unique())

    daily_sales = (
        df.groupby(["Date", "Province", "Item"], as_index=False)
        .agg(
            total_spent=("Total Spent", "sum"),
            transaction_count=("Total Spent", "size"),
            avg_transaction_value=("Total Spent", "mean"),
            location_missing_count=("Location_missing_flag", "sum"),
            province_missing_count=("Province_missing_flag", "sum"),
        )
    )

    full_index = pd.MultiIndex.from_product(
        [all_dates, provinces, items],
        names=["Date", "Province", "Item"],
    )
    full_panel = full_index.to_frame(index=False)

    df_panel = full_panel.merge(
        daily_sales,
        on=["Date", "Province", "Item"],
        how="left",
    )

    # Explicitly represent no-sale Province-Item-Date combinations.
    df_panel["had_transaction"] = df_panel["total_spent"].notna().astype(int)

    numeric_fill_zero_cols = [
        "total_spent",
        "transaction_count",
        "avg_transaction_value",
        "location_missing_count",
        "province_missing_count",
    ]
    for col in numeric_fill_zero_cols:
        df_panel[col] = df_panel[col].fillna(0)

    count_cols = [
        "transaction_count",
        "location_missing_count",
        "province_missing_count",
        "had_transaction",
    ]
    df_panel[count_cols] = df_panel[count_cols].astype(int)

    # Calendar features.
    df_features = df_panel.copy()
    df_features["Date"] = pd.to_datetime(df_features["Date"])
    df_features = df_features.sort_values(["Date", "Province", "Item"]).reset_index(drop=True)
    df_features["day_of_week"] = df_features["Date"].dt.dayofweek
    df_features["day_of_month"] = df_features["Date"].dt.day
    df_features["day_of_year"] = df_features["Date"].dt.dayofyear
    df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)
    df_features["quarter"] = df_features["Date"].dt.quarter
    df_features["month"] = df_features["Date"].dt.month

    # Lag features must be calculated within each Province-Item time series.
    df_features = df_features.sort_values(["Province", "Item", "Date"]).reset_index(drop=True)
    group = df_features.groupby(["Province", "Item"], sort=False)

    df_features["total_spent_lag_1"] = group["total_spent"].shift(1)
    df_features["total_spent_lag_7"] = group["total_spent"].shift(7)

    df_features["total_spent_lag_1_missing_flag"] = df_features["total_spent_lag_1"].isna().astype(int)
    df_features["total_spent_lag_7_missing_flag"] = df_features["total_spent_lag_7"].isna().astype(int)

    df_features["total_spent_lag_1"] = df_features["total_spent_lag_1"].fillna(0)
    df_features["total_spent_lag_7"] = df_features["total_spent_lag_7"].fillna(0)

    # Rolling averages use a shifted target so today's sales do not leak into today's feature value.
    df_features["total_spent_shifted"] = group["total_spent"].shift(1)
    df_features["total_spent_ma_3"] = group["total_spent_shifted"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df_features["total_spent_ma_7"] = group["total_spent_shifted"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    df_features["total_spent_ma_3_missing_flag"] = df_features["total_spent_ma_3"].isna().astype(int)
    df_features["total_spent_ma_7_missing_flag"] = df_features["total_spent_ma_7"].isna().astype(int)
    df_features["total_spent_ma_3"] = df_features["total_spent_ma_3"].fillna(0)
    df_features["total_spent_ma_7"] = df_features["total_spent_ma_7"].fillna(0)

    # Return to notebook display/model order.
    df_features = df_features.sort_values(["Date", "Province", "Item"]).reset_index(drop=True)

    expected_rows = len(all_dates) * len(provinces) * len(items)
    summary = {
        "invalid_dates_dropped": invalid_dates,
        "start_date": make_json_safe(pd.Timestamp(start)),
        "end_date": make_json_safe(pd.Timestamp(end)),
        "number_of_dates": int(len(all_dates)),
        "number_of_provinces": int(len(provinces)),
        "number_of_items": int(len(items)),
        "provinces": list(provinces),
        "items": list(items),
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(df_features)),
        "panel_row_count_match": bool(expected_rows == len(df_features)),
        "final_shape": [int(df_features.shape[0]), int(df_features.shape[1])],
        "final_columns": list(df_features.columns),
    }

    return df_features, summary


def train_lightgbm_validation_model(
    df_features: pd.DataFrame,
    output_dir: Path,
    drop_day_of_year_test: bool = True,
) -> dict[str, Any]:
    """Optionally reproduce the LightGBM validation experiment from the notebook."""
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
    except ImportError as exc:
        raise RuntimeError(
            "Model training requires lightgbm and scikit-learn. Install them or run without --train-model."
        ) from exc

    df_model = df_features.copy()
    df_model["Date"] = pd.to_datetime(df_model["Date"])
    df_model = df_model.sort_values(["Date", "Province", "Item"]).reset_index(drop=True)

    target_col = "total_spent"
    feature_cols = [
        "Province",
        "Item",
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "is_weekend",
        "quarter",
        "month",
        "total_spent_lag_1",
        "total_spent_lag_7",
        "total_spent_ma_3",
        "total_spent_ma_7",
    ]
    categorical_cols = ["Province", "Item"]
    for col in categorical_cols:
        df_model[col] = df_model[col].astype("category")

    max_date = df_model["Date"].max()
    validation_start_date = max_date - pd.Timedelta(days=14)
    train_df = df_model[df_model["Date"] < validation_start_date].copy()
    valid_df = df_model[df_model["Date"] >= validation_start_date].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    model = LGBMRegressor(
        objective="tweedie",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=categorical_cols,
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)],
    )

    y_pred = np.clip(model.predict(X_valid), 0, None)
    metrics = {
        "mae": float(mean_absolute_error(y_valid, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred))),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_valid, y_pred))),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "validation_start_date": validation_start_date.isoformat(),
    }

    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    validation_results = valid_df[["Date", "Province", "Item", "total_spent"]].copy()
    validation_results["predicted_total_spent"] = y_pred
    validation_results["error"] = validation_results["total_spent"] - validation_results["predicted_total_spent"]
    validation_results.to_csv(output_dir / "validation_predictions.csv", index=False)

    if drop_day_of_year_test:
        feature_cols_without_day_of_year = [col for col in feature_cols if col != "day_of_year"]
        model_no_doy = LGBMRegressor(
            objective="tweedie",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model_no_doy.fit(
            train_df[feature_cols_without_day_of_year],
            y_train,
            eval_set=[(valid_df[feature_cols_without_day_of_year], y_valid)],
            eval_metric="rmse",
            categorical_feature=categorical_cols,
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=50)],
        )
        y_pred_no_doy = np.clip(model_no_doy.predict(valid_df[feature_cols_without_day_of_year]), 0, None)
        metrics["without_day_of_year"] = {
            "mae": float(mean_absolute_error(y_valid, y_pred_no_doy)),
            "rmse": float(np.sqrt(mean_squared_error(y_valid, y_pred_no_doy))),
            "rmsle": float(np.sqrt(mean_squared_log_error(y_valid, y_pred_no_doy))),
            "best_iteration": int(getattr(model_no_doy, "best_iteration_", 0) or 0),
        }
        feature_importance_no_doy = pd.DataFrame(
            {"feature": feature_cols_without_day_of_year, "importance": model_no_doy.feature_importances_}
        ).sort_values("importance", ascending=False)
        feature_importance_no_doy.to_csv(output_dir / "feature_importance_without_day_of_year.csv", index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create feature-engineered CanAI Cafe sales data.")
    parser.add_argument("--input", required=True, help="Path to cleaned_transactions.csv.")
    parser.add_argument("--output", required=True, help="Path for the feature-engineered CSV.")
    parser.add_argument("--start-date", default=None, help="Optional panel start date, e.g. 2023-01-01.")
    parser.add_argument("--end-date", default=None, help="Optional panel end date, e.g. 2023-12-31.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for a JSON feature-engineering summary. Defaults to <output_stem>_summary.json.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Also reproduce the notebook's LightGBM validation experiment and save metrics/artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_name(f"{output_path.stem}_summary.json")

    cleaned_df = pd.read_csv(input_path)
    features_df, summary = build_feature_panel(cleaned_df, start_date=args.start_date, end_date=args.end_date)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    if args.train_model:
        model_output_dir = output_path.with_name(f"{output_path.stem}_model_artifacts")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        summary["model_metrics"] = train_lightgbm_validation_model(features_df, model_output_dir)
        summary["model_artifact_dir"] = str(model_output_dir)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Feature-engineered CSV written to: {output_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Final dataset shape: {features_df.shape}")
    print(f"Panel row count match: {summary['panel_row_count_match']}")


if __name__ == "__main__":
    main()
