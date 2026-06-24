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


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "CanAI Cafe 2023 Sales Information.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def load_daily_sales() -> pd.DataFrame:
    """
    This function loads raw transactions and aggregates them into daily sales totals.

    Parameters:
        None

    Returns:
        pd.DataFrame: Daily sales data with a complete daily calendar.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load the raw Excel dataset and create a copy before mutating values.
    raw = pd.read_excel(DATA_PATH, engine="openpyxl")
    raw = raw.copy()

    required_columns = {"Transaction Date", "Total Spent"}
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

    # Ensure the sales amounts are numeric and drop invalid rows.
    raw["Total Spent"] = pd.to_numeric(
        raw["Total Spent"],
        errors="coerce",
    )

    raw = raw.dropna(subset=["Transaction Date", "Total Spent"])

    daily = (
        raw.assign(date=raw["Transaction Date"].dt.normalize())
        .groupby("date", as_index=False)["Total Spent"]
        .sum()
        .rename(columns={"Total Spent": "daily_total_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Create a complete daily index and fill gaps with zero sales.
    complete_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=daily["date"].min(),
                end=daily["date"].max(),
                freq="D",
            )
        }
    )

    daily = complete_dates.merge(daily, on="date", how="left")
    daily["daily_total_sales"] = daily["daily_total_sales"].fillna(0.0)

    return daily


def create_forecast_with_fallback(model, steps: int) -> pd.DataFrame:
    """
    This function generates a daily forecast, using a fallback when the primary helper fails.

    Parameters:
        model: A fitted SARIMA or forecast-capable model instance.
        steps: Number of future days to forecast.

    Returns:
        pd.DataFrame: Forecast output with point and interval columns.
    """
    try:
        return generate_sarima_forecast(model, steps=steps)
    except (AttributeError, TypeError):
        # Fallback path for models lacking the same forecast interface.
        prediction = model.get_forecast(steps=steps)
        intervals = prediction.conf_int(alpha=0.05)

        last_date = pd.Timestamp(model.data.dates[-1])

        forecast = pd.DataFrame(
            {
                "forecast": prediction.predicted_mean.to_numpy(),
                "lower_bound": intervals.iloc[:, 0].to_numpy(),
                "upper_bound": intervals.iloc[:, 1].to_numpy(),
            },
            index=pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq="D",
            ),
        )

        return forecast


def main() -> None:
    """
    This function runs the complete baseline and SARIMA forecasting workflow.

    Parameters:
        None

    Returns:
        None
    """
    print("Loading and preparing sales data...", flush=True)
    daily_input = load_daily_sales()

    series = extract_daily_sales_series(
        daily_input,
        sales_col="daily_total_sales",
        date_col="date",
    )

    # Confirm the daily series is properly indexed and ready for modeling.
    daily_indexed = ensure_daily_index(daily_input, date_col="date")

    print(f"Daily series length: {len(series)}", flush=True)
    print(series.head(), flush=True)

    # Build calendar features for diagnostic inspection and potential exogenous model use.
    calendar = build_calendar_features(daily_input, date_col="date")
    print("\nCalendar feature sample:", flush=True)
    print(calendar.head(), flush=True)

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

    # Generate the forecast and enforce nonnegative sales values.
    forecast = create_forecast_with_fallback(fitted_model, steps=30)
    forecast = clip_negative_forecasts(forecast)

    print("\nSARIMA forecast sample:", flush=True)
    print(forecast.head(), flush=True)

    actual = validation["daily_total_sales"].iloc[: len(forecast)]
    prediction = forecast["forecast"].iloc[: len(actual)].copy()

    # Match the prediction index to the validation index for metric calculation.
    prediction.index = actual.index

    metrics = calculate_forecast_metrics(actual, prediction)

    print("\nValidation metrics:", flush=True)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}", flush=True)

    monthly = aggregate_forecast_to_monthly(forecast)

    try:
        final = build_forecast_output(forecast)
    except ValueError:
        # Fallback for a duplicate-category issue in older versions
        # of build_forecast_output.
        final = forecast.copy()
        final["forecast_month"] = final.index.to_period("M").astype(str)
        width = final["upper_bound"] - final["lower_bound"]
        final["risk_level"] = pd.cut(
            width,
            bins=[-1, 0.1, 0.25, float("inf")],
            labels=["low", "medium", "high"],
        ).fillna("medium")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_output_path = OUTPUT_DIR / "daily_forecast.csv"
    monthly_output_path = OUTPUT_DIR / "monthly_forecast.csv"
    metrics_output_path = OUTPUT_DIR / "forecast_metrics.csv"
    a_output_path = PROJECT_ROOT / "a.csv"

    # Ask the user whether to save the results to CSV files.
    while True:
        save_answer = input("Save results to CSV files? (y/n): ").strip().lower()
        if save_answer in {"y", "n"}:
            break
        print("Please enter 'y' or 'n'.")

    if save_answer == "y":
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

    print("\nMonthly forecast:", flush=True)
    print(monthly, flush=True)

    # Keep this reference available for debugging and inspection.
    _ = daily_indexed


if __name__ == "__main__":
    main()
