
"""
Usage: 

py -m pytest tests\test_modeling.py -v

"""


from pathlib import Path

import pandas as pd
import pytest

from src.modeling import (
    calculate_forecast_metrics,
    ensure_daily_index,
    extract_daily_sales_series,
    naive_forecast,
    seasonal_naive_forecast,
    split_time_series,
    tune_sarima_parameters,
)

# Path to the raw transaction dataset used by the modeling tests.
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "CanAI Cafe 2023 Sales Information.xlsx"

# Columns expected to be present in the raw dataset
EXPECTED_COLUMNS = ["Transaction Date", "Total Spent"]


def load_raw_transaction_data() -> pd.DataFrame:
    """
    Load the raw transaction Excel sheet for model validation tests.

    Parameters:
        None

    Returns:
        pd.DataFrame: The raw transaction data.
    """
    return pd.read_excel(DATA_PATH, engine="openpyxl")


def test_raw_transaction_data_contains_expected_columns():
    """
    Verify the raw sales dataset exposes the expected columns and types.

    Parameters:
        None

    Returns:
        None
    """
    df = load_raw_transaction_data()

    for expected in EXPECTED_COLUMNS:
        assert expected in df.columns, f"Missing expected column: {expected}"

    parsed_dates = pd.to_datetime(df["Transaction Date"], errors="coerce")
    assert parsed_dates.notna().sum() > 0, "Transaction Date column should contain parseable dates"
    assert pd.api.types.is_numeric_dtype(df["Total Spent"])


def test_extract_daily_sales_series_from_raw_data():
    """
    Confirm the modeling pipeline can build a daily sales series from raw transaction data.

    Parameters:
        None

    Returns:
        None
    """
    df = load_raw_transaction_data()
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")
    df = df.dropna(subset=["Transaction Date", "Total Spent"])

    # Aggregate transactions by day into total daily sales.
    daily = (
        df.assign(date=df["Transaction Date"].dt.normalize())
        .groupby("date", as_index=False)["Total Spent"]
        .sum()
        .rename(columns={"Total Spent": "daily_total_sales"})
    )

    series = extract_daily_sales_series(daily, sales_col="daily_total_sales", date_col="date")

    assert isinstance(series, pd.Series)
    assert series.name == "daily_total_sales"
    assert series.index.freq == pd.tseries.frequencies.to_offset("D")
    assert not series.isna().all()
    assert series.index.min() == daily["date"].min()


def test_extract_daily_sales_series_raises_on_missing_sales_column():
    """
    Validate that missing sales columns raise a clear KeyError.

    Parameters:
        None

    Returns:
        None
    """
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "other_sales": [10, 20, 30, 40, 50],
        }
    )

    with pytest.raises(KeyError, match="Expected sales column"):
        extract_daily_sales_series(df, sales_col="daily_total_sales", date_col="date")


def test_naive_and_seasonal_baselines_produce_expected_length_and_index():
    train_series = pd.Series(
        [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0],
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
        name="daily_total_sales",
    )

    naive = naive_forecast(train_series, forecast_steps=5)
    seasonal = seasonal_naive_forecast(train_series, forecast_steps=5)

    assert len(naive) == 5
    assert len(seasonal) == 5
    assert naive.index.freq == pd.tseries.frequencies.to_offset("D")
    assert seasonal.index.freq == pd.tseries.frequencies.to_offset("D")
    assert naive.iloc[0] == train_series.iloc[-1]
    assert seasonal.iloc[0] == train_series.iloc[-7]


def test_calculate_forecast_metrics_handles_zero_actuals_gracefully():
    actual = pd.Series([0.0, 0.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=4, freq="D"))
    prediction = pd.Series([1.0, 2.0, 3.0, 4.0], index=actual.index)

    metrics = calculate_forecast_metrics(actual, prediction)

    assert metrics["MAE"] == pytest.approx(2.5)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(7.5))
    assert metrics["WAPE"] == pytest.approx(1.0)
    assert np.isnan(metrics["MAPE"])


def test_split_time_series_respects_validation_and_test_sizes():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="D"),
            "daily_total_sales": np.arange(100, dtype=float),
        }
    )

    train, validation, test = split_time_series(data, validation_days=10, test_days=10, date_col="date")

    assert len(train) == 80
    assert len(validation) == 10
    assert len(test) == 10
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()


def test_tune_sarima_parameters_returns_successful_configuration():
    data = pd.Series(
        np.arange(1, 41, dtype=float),
        index=pd.date_range("2024-01-01", periods=40, freq="D"),
        name="daily_total_sales",
    )
    train = data.iloc[:30]
    validation = data.iloc[30:40]

    best_config, results = tune_sarima_parameters(
        train,
        validation,
        orders=[(0, 1, 1), (1, 1, 1)],
        seasonal_orders=[(1, 0, 1, 7)],
        trends=[None],
        maxiter=100,
    )

    assert isinstance(best_config, dict)
    assert best_config["status"] == "success"
    assert "WAPE" in best_config
    assert not results.empty
    assert all(results["status"].isin({"success"}) | results["status"].str.startswith("error"))
