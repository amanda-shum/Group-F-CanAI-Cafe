"""
Usage:
    py run_model.py train
    py run_model.py test

Commands:
    train - load data, train SARIMA, evaluate on validation, and optionally save validation forecasts.
    test  - retrain on train+validation, evaluate on test data, generate a 6-month forecast, and optionally save outputs.

This script loads daily sales from the Excel file, trains a SARIMA forecasting pipeline,
validates performance, evaluates on a test set, and optionally saves results
including a 6-month forecast.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.modeling import (
    aggregate_forecast_to_monthly,
    aggregate_daily_sales_by_group,
    build_calendar_features,
    build_forecast_output,
    calculate_forecast_metrics,
    clip_negative_forecasts,
    complete_grouped_daily_sales,
    ensure_daily_index,
    extract_daily_sales_series,
    fit_sarima_model,
    generate_sarima_forecast,
    get_model_diagnostics,
    naive_forecast,
    seasonal_naive_forecast,
    split_time_series,
    tune_sarima_parameters,
)
from src.sales_insights import calculate_sales_insights, insights_to_dataframe, print_sales_insights, prompt_save_insights


def format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if metric in {"MAE", "RMSE"}:
        return f"${value:,.2f}"
    if metric == "WAPE":
        return f"{value * 100:.2f}%"
    if metric == "MAPE":
        return f"{value:.2f}%"
    return str(value)


def print_model_comparison(metrics_by_model: dict[str, dict[str, float]]) -> None:
    print("\nVALIDATION MODEL COMPARISON", flush=True)
    print("=" * 65, flush=True)
    print(
        f"{'Model':<18}{'MAE':>12}{'RMSE':>12}{'WAPE':>12}{'MAPE':>12}",
        flush=True,
    )
    print("-" * 65, flush=True)

    for model_name, metrics in metrics_by_model.items():
        mae = format_metric_value("MAE", metrics.get("MAE", float("nan")))
        rmse = format_metric_value("RMSE", metrics.get("RMSE", float("nan")))
        wape = format_metric_value("WAPE", metrics.get("WAPE", float("nan")))
        mape = format_metric_value("MAPE", metrics.get("MAPE", float("nan")))
        print(f"{model_name:<18}{mae:>12}{rmse:>12}{wape:>12}{mape:>12}", flush=True)
    print("=" * 65, flush=True)


def select_best_validation_model(metrics_by_model: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    sorted_models = sorted(
        metrics_by_model.items(),
        key=lambda item: (
            float(item[1].get("WAPE", float("nan"))),
            float(item[1].get("MAE", float("nan"))),
        ),
    )
    return sorted_models[0]


def safe_percentage_improvement(baseline_wape: float, candidate_wape: float) -> float:
    if pd.isna(baseline_wape) or baseline_wape == 0:
        return float("nan")
    return ((baseline_wape - candidate_wape) / baseline_wape) * 100


def validation_comparison_to_dataframe(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "MAE": metrics.get("MAE", float("nan")),
                "RMSE": metrics.get("RMSE", float("nan")),
                "WAPE": metrics.get("WAPE", float("nan")),
                "MAPE": metrics.get("MAPE", float("nan")),
            }
        )
    return pd.DataFrame(rows)

#directory paths for data and outputs
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


def prompt_save_grouped_data() -> bool:
    """
    Prompt the user to confirm whether grouped Province/item-level CSVs should be saved.

    Returns:
        bool: True when the user confirms saving grouped data.
    """
    while True:
        answer = input(
            "Save Province/item-level daily sales CSVs? Enter 'y' to save or 'n' to skip: "
        ).strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"
        print("Please enter 'y' or 'n'.")


def load_raw_transactions() -> pd.DataFrame:
    """
    Load raw transaction data with province and item fields preserved.

    Returns:
        pd.DataFrame: Cleaned raw transactions ready for grouped aggregation.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = pd.read_excel(DATA_PATH, engine="openpyxl").copy()

    required_columns = {"Transaction Date", "Total Spent", "Province", "Item"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise KeyError(
            "The dataset is missing required columns for grouped exports: "
            + ", ".join(sorted(missing_columns))
        )

    raw["Transaction Date"] = pd.to_datetime(raw["Transaction Date"], errors="coerce")
    raw["Total Spent"] = pd.to_numeric(raw["Total Spent"], errors="coerce")

    raw = raw.dropna(subset=["Transaction Date", "Total Spent"])
    raw["Province"] = raw["Province"].astype(str).str.strip()
    raw["Item"] = raw["Item"].astype(str).str.strip()

    return raw


def save_grouped_csvs(raw_transactions: pd.DataFrame) -> None:
    """
    Save grouped daily total sales for Province and Item.

    Parameters:
        raw_transactions: Cleaned raw transaction DataFrame.
    """
    province_daily = aggregate_daily_sales_by_group(
        raw_transactions,
        group_cols=["Province"],
        date_col="Transaction Date",
        sales_col="Total Spent",
    )
    province_daily = complete_grouped_daily_sales(province_daily, ["Province"], date_col="date")

    item_daily = aggregate_daily_sales_by_group(
        raw_transactions,
        group_cols=["Item"],
        date_col="Transaction Date",
        sales_col="Total Spent",
    )
    item_daily = complete_grouped_daily_sales(item_daily, ["Item"], date_col="date")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    province_output = OUTPUT_DIR / "province_daily_sales.csv"
    item_output = OUTPUT_DIR / "item_daily_sales.csv"

    province_daily.to_csv(province_output, index=False)
    item_daily.to_csv(item_output, index=False)

    print("\nSaved grouped Province/item-level CSVs:", flush=True)
    print(f"- {province_output}", flush=True)
    print(f"- {item_output}", flush=True)


def display_province_sales_summary(raw_transactions: pd.DataFrame, top_n: int = 10) -> None:
    """
    Show a terminal summary of total sales by province.

    Parameters:
        raw_transactions: Cleaned raw transaction DataFrame.
        top_n: Number of top provinces to show.
    """
    province_daily = aggregate_daily_sales_by_group(
        raw_transactions,
        group_cols=["Province"],
        date_col="Transaction Date",
        sales_col="Total Spent",
    )
    totals = (
        province_daily.groupby("Province", as_index=False)["daily_total_sales"]
        .sum()
        .rename(columns={"daily_total_sales": "total_sales"})
        .sort_values("total_sales", ascending=False)
    )

    print("\nPROVINCE SALES SUMMARY", flush=True)
    if totals.empty:
        print("No province-level sales data is available.", flush=True)
        return

    print(totals.head(top_n).to_string(index=False), flush=True)
    print(
        f"\nDisplayed top {min(top_n, len(totals))} provinces by total sales.",
        flush=True,
    )


def display_item_sales_summary(raw_transactions: pd.DataFrame, top_n: int = 10) -> None:
    """
    Show a terminal summary of total sales by item.

    Parameters:
        raw_transactions: Cleaned raw transaction DataFrame.
        top_n: Number of top items to show.
    """
    item_daily = aggregate_daily_sales_by_group(
        raw_transactions,
        group_cols=["Item"],
        date_col="Transaction Date",
        sales_col="Total Spent",
    )
    totals = (
        item_daily.groupby("Item", as_index=False)["daily_total_sales"]
        .sum()
        .rename(columns={"daily_total_sales": "total_sales"})
        .sort_values("total_sales", ascending=False)
    )

    print("\nITEM SALES SUMMARY", flush=True)
    if totals.empty:
        print("No item-level sales data is available.", flush=True)
        return

    print(totals.head(top_n).to_string(index=False), flush=True)
    print(
        f"\nDisplayed top {min(top_n, len(totals))} items by total sales.",
        flush=True,
    )


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

    print("\nTuning SARIMA hyperparameters on the validation window...", flush=True)
    best_config, tuning_results = tune_sarima_parameters(
        train["daily_total_sales"],
        validation["daily_total_sales"],
        orders=[(0, 1, 1), (1, 0, 1), (1, 1, 1)],
        seasonal_orders=[(1, 0, 1, 7), (1, 1, 1, 7)],
        trends=[None, "c"],
        maxiter=200,
    )

    print("\nBest SARIMA config:", flush=True)
    print(
        f"order={best_config['order']} seasonal_order={best_config['seasonal_order']} trend={best_config['trend']}",
        flush=True,
    )

    print("\nTop tuning candidates by WAPE:", flush=True)
    top_candidates = (
        tuning_results[tuning_results["status"] == "success"]
        .sort_values(["WAPE", "MAE"], ascending=True)
        .head(3)[["order", "seasonal_order", "trend", "WAPE", "MAE"]]
    )
    print(top_candidates.to_string(index=False), flush=True)

    print("\nTraining SARIMA model with the best validation configuration...", flush=True)
    fitted_model = fit_sarima_model(
        train["daily_total_sales"],
        order=best_config["order"],
        seasonal_order=best_config["seasonal_order"],
        trend=best_config["trend"],
        maxiter=200,
    )

    diagnostics = get_model_diagnostics(
        fitted_model,
        best_config["order"],
        best_config["seasonal_order"],
        best_config["trend"],
    )
    print("\nSARIMA diagnostics:", flush=True)
    for key, value in diagnostics.items():
        print(f"- {key}: {value}", flush=True)

    # Forecast only the validation horizon to evaluate the model's generalization.
    forecast = generate_sarima_forecast(fitted_model, steps=len(validation))
    forecast = clip_negative_forecasts(forecast)

    print("\nValidation forecast sample:", flush=True)
    print(forecast.head(), flush=True)

    # Align forecast predictions with the validation dates for evaluation.
    actual = validation["daily_total_sales"].copy()
    naive_pred = naive.copy()
    naive_pred.index = actual.index
    seasonal_pred = seasonal.copy()
    seasonal_pred.index = actual.index
    sarima_pred = forecast["forecast"].copy()
    sarima_pred.index = actual.index

    naive_metrics = calculate_forecast_metrics(actual, naive_pred)
    seasonal_metrics = calculate_forecast_metrics(actual, seasonal_pred)
    sarima_metrics = calculate_forecast_metrics(actual, sarima_pred)

    metrics_by_model = {
        "Naive": naive_metrics,
        "Seasonal Naive": seasonal_metrics,
        "SARIMA": sarima_metrics,
    }
    print_model_comparison(metrics_by_model)

    best_model_name, best_model_metrics = select_best_validation_model(metrics_by_model)
    baseline_wape = seasonal_metrics["WAPE"]
    sarima_wape = sarima_metrics["WAPE"]
    improvement = safe_percentage_improvement(baseline_wape, sarima_wape)

    print(f"Best validation model: {best_model_name}", flush=True)
    if best_model_name == "SARIMA":
        print(
            f"SARIMA outperformed seasonal naive by {improvement:.2f}% WAPE improvement.",
            flush=True,
        )
    else:
        print(
            f"SARIMA did not outperform seasonal naive. Improvement would have been {improvement:.2f}%.",
            flush=True,
        )

    metrics = sarima_metrics

    # Aggregate the daily validation forecast to monthly totals for higher-level reporting.
    monthly = aggregate_forecast_to_monthly(forecast)
    final = build_forecast_output(forecast)

    print("\nValidation monthly forecast:", flush=True)
    print(monthly, flush=True)

    raw_transactions = load_raw_transactions()
    display_province_sales_summary(raw_transactions)
    display_item_sales_summary(raw_transactions)

    insights = calculate_sales_insights(
        daily_input.set_index("date")["daily_total_sales"],
        actual_values=actual,
        forecast_values=sarima_pred,
        wape=sarima_metrics.get("WAPE"),
    )
    print_sales_insights(insights)

    if prompt_save_insights():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        insights_path = OUTPUT_DIR / "sales_insights.csv"
        insights_to_dataframe(insights).to_csv(insights_path, index=False)
        print(f"\nSales insights saved to: {insights_path}", flush=True)
    else:
        print("\nSkipping sales insights save.", flush=True)

    print("\nResults are ready. You can choose whether to save the outputs.", flush=True)
    if prompt_save_results():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        daily_output_path = OUTPUT_DIR / "validation_forecast.csv"
        monthly_output_path = OUTPUT_DIR / "validation_monthly_forecast.csv"
        metrics_output_path = OUTPUT_DIR / "validation_metrics.csv"
        comparison_output_path = OUTPUT_DIR / "validation_model_comparison.csv"

        final.to_csv(daily_output_path, index_label="date")
        monthly.to_csv(monthly_output_path, index_label="month")
        pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False)
        validation_comparison_to_dataframe(metrics_by_model).to_csv(
            comparison_output_path, index=False
        )
        tuning_results.to_csv(OUTPUT_DIR / "sarima_tuning_results.csv", index=False)

        print("\nSaved outputs:", flush=True)
        print(f"- {daily_output_path}", flush=True)
        print(f"- {monthly_output_path}", flush=True)
        print(f"- {metrics_output_path}", flush=True)
        print(f"- {comparison_output_path}", flush=True)

        if prompt_save_grouped_data():
            raw_transactions = load_raw_transactions()
            save_grouped_csvs(raw_transactions)
    else:
        print("\nSkipping CSV save. No files were written.", flush=True)

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

    print("\nMonthly forecast:", flush=True)
    print(monthly, flush=True)

    raw_transactions = load_raw_transactions()
    display_province_sales_summary(raw_transactions)
    display_item_sales_summary(raw_transactions)

    insights = calculate_sales_insights(
        daily_input.set_index("date")["daily_total_sales"],
        actual_values=test_actual,
        forecast_values=test_prediction,
        wape=test_metrics.get("WAPE"),
    )
    print_sales_insights(insights)

    if prompt_save_insights():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        insights_path = OUTPUT_DIR / "sales_insights.csv"
        insights_to_dataframe(insights).to_csv(insights_path, index=False)
        print(f"\nSales insights saved to: {insights_path}", flush=True)
    else:
        print("\nSkipping sales insights save.", flush=True)

    print("\nResults are ready. You can choose whether to save the outputs.", flush=True)
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

        if prompt_save_grouped_data():
            raw_transactions = load_raw_transactions()
            save_grouped_csvs(raw_transactions)
    else:
        print("\nSkipping CSV save. No files were written.", flush=True)

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
