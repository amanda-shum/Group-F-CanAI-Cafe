"""
Usage:
    py run_model.py train
    py run_model.py test

Commands:
    train - load data, train SARIMA, evaluate on validation, and optionally save validation forecasts.
    test  - retrain on train+validation, evaluate on test data, generate a 6-month forecast, and optionally save outputs.

This script loads daily sales from the Excel file, trains a SARIMA forecasting pipeline,
validates performance, evaluates on a held-out test set, and optionally saves results
including a 6-month forecast.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.modeling import (
    aggregate_forecast_to_monthly,
    build_calendar_features,
    build_forecast_output,
    calculate_forecast_metrics,
    clip_negative_forecasts,
    ensure_daily_index,
    extract_daily_sales_series,
    fit_sarima_model,
    generate_sarima_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    split_time_series,
)


PROJECT_ROOT = Path(__file__).resolve().parent  # root of the project directory
DATA_PATH = PROJECT_ROOT / "data" / "CanAI Cafe 2023 Sales Information.xlsx"  # source dataset path
OUTPUT_DIR = PROJECT_ROOT / "reports"  # output directory for forecast results


def load_daily_sales() -> pd.DataFrame:
    """
    This function loads raw transactions and aggregates them into daily sales totals.

    Parameters:
        None

    Returns:
        pd.DataFrame: Daily sales data with a complete daily calendar.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")  # fail fast if source data is missing

    # Read raw Excel data and preserve the original by working on a copy.
    raw = pd.read_excel(DATA_PATH, engine="openpyxl")
    raw = raw.copy()

    required_columns = {"Transaction Date", "Total Spent"}  # expected raw dataset columns
    missing_columns = required_columns.difference(raw.columns)

    if missing_columns:
        raise KeyError(
            "The dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    raw["Transaction Date"] = pd.to_datetime(
        raw["Transaction Date"],
        errors="coerce",
    )
    if raw["Transaction Date"].isna().all():
        raise ValueError("Transaction Date column contains no parseable dates.")

    # Convert transaction amounts to numeric values, invalid values become NaN.
    raw["Total Spent"] = pd.to_numeric(
        raw["Total Spent"],
        errors="coerce",
    )

    # Drop rows missing either a valid date or a valid sales amount.
    raw = raw.dropna(subset=["Transaction Date", "Total Spent"])

    # Aggregate raw transactions into daily totals.
    daily = (
        raw.assign(date=raw["Transaction Date"].dt.normalize())
        .groupby("date", as_index=False)["Total Spent"]
        .sum()
        .rename(columns={"Total Spent": "daily_total_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Create a complete daily index from the first to last date in the data.
    complete_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=daily["date"].min(),
                end=daily["date"].max(),
                freq="D",
            )
        }
    )

    # Merge with the full calendar and fill missing days with zero sales.
    daily = complete_dates.merge(daily, on="date", how="left")
    daily["daily_total_sales"] = daily["daily_total_sales"].fillna(0.0)

    return daily


def prompt_save_results() -> bool:
    """
    Prompt the user to confirm whether results should be saved to CSV.

    Parameters:
        None

    Returns:
        bool: True when the user confirms saving.
    """
    while True:
        # Ask the user to choose whether results should be written to CSV.
        save_answer = input(
            "Save results to CSV files? Enter 'y' to save or 'n' to skip: "
        ).strip().lower()
        if save_answer in {"y", "n"}:
            return save_answer == "y"
        print("Please enter 'y' or 'n'.")


def train_pipeline() -> None:
    """
    Run the training workflow: train SARIMA on the train split and evaluate on validation.

    Parameters:
        None

    Returns:
        None
    """
    print("Loading and preparing sales data...", flush=True)
    daily_input = load_daily_sales()

    # Extract the ordered time series used by the model.
    series = extract_daily_sales_series(
        daily_input,
        sales_col="daily_total_sales",
        date_col="date",
    )  # get the daily sales series for modeling

    # Ensure the dataset has a consistent daily DateTimeIndex for any downstream output.
    daily_indexed = ensure_daily_index(daily_input, date_col="date")

    print(f"Daily series length: {len(series)}", flush=True)
    print(series.head(), flush=True)

    # Split the data into train / validation / test partitions before training.
    # We leave the final test window untouched until test mode is run.
    train, validation, test = split_time_series(
        daily_input,
        validation_days=30,
        test_days=30,
        date_col="date",
    )  # split into train, validation, and test partitions

    print(
        f"\nTrain/validation/test sizes: "
        f"{len(train)}/{len(validation)}/{len(test)}",
        flush=True,
    )

    # Build simple baseline forecasts for comparison.
    naive = naive_forecast(train["daily_total_sales"], forecast_steps=30)
    seasonal = seasonal_naive_forecast(
        train["daily_total_sales"],
        forecast_steps=30,
    )

    print("\nNaive forecast sample:", naive.head().tolist(), flush=True)
    print(
        "Seasonal-naive forecast sample:",
        seasonal.head().tolist(),
        flush=True,
    )

    print("\nTraining SARIMA model. This may take a moment...", flush=True)
    fitted_model = fit_sarima_model(train["daily_total_sales"])

    # Forecast only the validation horizon to evaluate the model's generalization.
    forecast = generate_sarima_forecast(fitted_model, steps=len(validation))
    forecast = clip_negative_forecasts(forecast)

    print("\nValidation forecast sample:", flush=True)
    print(forecast.head(), flush=True)

    # Align forecast predictions with the validation dates for evaluation.
    actual = validation["daily_total_sales"].copy()
    prediction = forecast["forecast"].copy()
    prediction.index = actual.index

    metrics = calculate_forecast_metrics(actual, prediction)

    print("\nValidation metrics:", flush=True)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}", flush=True)

    # Aggregate the daily validation forecast to monthly totals for higher-level reporting.
    monthly = aggregate_forecast_to_monthly(forecast)
    final = build_forecast_output(forecast)

    if prompt_save_results():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily_output_path = OUTPUT_DIR / "validation_forecast.csv"
        monthly_output_path = OUTPUT_DIR / "validation_monthly_forecast.csv"
        metrics_output_path = OUTPUT_DIR / "validation_metrics.csv"
        a_output_path = PROJECT_ROOT / "a.csv"

        final.to_csv(daily_output_path, index_label="date")
        monthly.to_csv(monthly_output_path, index_label="month")
        pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False)
        final.to_csv(a_output_path, index_label="date")

        print("\nSaved outputs:", flush=True)
        print(f"- {daily_output_path}", flush=True)
        print(f"- {monthly_output_path}", flush=True)
        print(f"- {metrics_output_path}", flush=True)
        print(f"- {a_output_path}", flush=True)
    else:
        print("\nSkipping CSV save. No files were written.", flush=True)

    print("\nValidation monthly forecast:", flush=True)
    print(monthly, flush=True)

    # Keep the indexed series alive in case additional analysis is added later.
    _ = daily_indexed


def test_pipeline() -> None:
    """
    Run the final test workflow: retrain on train+validation and evaluate on the test split.

    Parameters:
        None

    Returns:
        None
    """
    print("Loading and preparing sales data...", flush=True)
    daily_input = load_daily_sales()

    # split into train, validation, and test partitions
    train, validation, test = split_time_series(
        daily_input,
        validation_days=30,
        test_days=30,
        date_col="date",
    ) 

    print(
        f"\nTrain/validation/test sizes: "
        f"{len(train)}/{len(validation)}/{len(test)}",
        flush=True,
    )

    print("\nRetraining SARIMA on train + validation for final test evaluation...", flush=True)
    
    # Retrain the final model on both train and validation data before test scoring.
    train_validation = pd.concat([train, validation]).sort_index()
    test_model = fit_sarima_model(train_validation["daily_total_sales"])

    test_forecast = generate_sarima_forecast(test_model, steps=len(test))
    test_forecast = clip_negative_forecasts(test_forecast)

    test_actual = test["daily_total_sales"].copy()
    test_prediction = test_forecast["forecast"].copy()
    test_prediction.index = test_actual.index

    test_metrics = calculate_forecast_metrics(test_actual, test_prediction)

    print("\nFinal test metrics:", flush=True)
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}", flush=True)

    print("\nGenerating a 6-month forecast from the final model...", flush=True)
    
    # Generate a longer horizon forecast for roughly 6 months of daily estimates.
    six_month_steps = 182
    six_month_forecast = generate_sarima_forecast(test_model, steps=six_month_steps)
    six_month_forecast = clip_negative_forecasts(six_month_forecast)

    print("6-month forecast sample:", flush=True)
    print(six_month_forecast.head(), flush=True)

    six_month_start = six_month_forecast.index.min()
    six_month_end = six_month_forecast.index.max()
    six_month_total = six_month_forecast["forecast"].sum()
    six_month_average = six_month_forecast["forecast"].mean()

    print("\n6-month forecast summary:", flush=True)
    print(f"- Horizon: {six_month_start.date()} to {six_month_end.date()}", flush=True)
    print(f"- Total forecast sales: ${six_month_total:,.2f}", flush=True)
    print(f"- Average daily forecast: ${six_month_average:,.2f}", flush=True)

    monthly = aggregate_forecast_to_monthly(test_forecast)
    final = build_forecast_output(test_forecast)

    if prompt_save_results():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily_output_path = OUTPUT_DIR / "daily_forecast.csv"
        monthly_output_path = OUTPUT_DIR / "monthly_forecast.csv"
        metrics_output_path = OUTPUT_DIR / "forecast_metrics.csv"
        six_month_output_path = OUTPUT_DIR / "six_month_forecast.csv"
        a_output_path = PROJECT_ROOT / "a.csv"

        final.to_csv(daily_output_path, index_label="date")
        monthly.to_csv(monthly_output_path, index_label="month")
        pd.DataFrame([test_metrics]).to_csv(metrics_output_path, index=False)
        six_month_forecast.to_csv(six_month_output_path, index_label="date")
        final.to_csv(a_output_path, index_label="date")

        print("\nSaved outputs:", flush=True)
        print(f"- {daily_output_path}", flush=True)
        print(f"- {monthly_output_path}", flush=True)
        print(f"- {metrics_output_path}", flush=True)
        print(f"- {six_month_output_path}", flush=True)
        print(f"- {a_output_path}", flush=True)
    else:
        print("\nSkipping CSV save. No files were written.", flush=True)

    print("\nMonthly forecast:", flush=True)
    print(monthly, flush=True)

    _ = train_validation


def main() -> None:
    """
    This function selects and runs either the training or test workflow.

    Parameters:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run the SARIMA forecasting pipeline in separate train or test mode."
    )
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Pipeline mode to run: 'train' evaluates on validation, 'test' evaluates on test data.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline()
    else:
        test_pipeline()


if __name__ == "__main__":
    main()
