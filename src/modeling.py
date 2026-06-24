"""Modeling and post-cleaning preprocessing for CanAI Café forecasting."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def ensure_daily_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure the DataFrame has a daily DatetimeIndex and preserve daily rows.

    Parameters:
        df: Cleaned DataFrame containing a date column.
        date_col: Name of the column to use as the date index.

    Returns:
        DataFrame indexed by date with daily frequency enforced.
    """
    df = df.copy()
    # Convert the date column to datetime; invalid values become NaT.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows without a valid date before indexing.
    df = df.dropna(subset=[date_col])

    # Use the date column as the index and sort the data chronologically.
    df = df.set_index(date_col).sort_index()

    # Force a daily frequency so missing dates are visible as NaN.
    df = df.asfreq("D")

    return df


def extract_daily_sales_series(df: pd.DataFrame, sales_col: str = "daily_total_sales", date_col: str = "date") -> pd.Series:
    """
    Return a daily sales series from cleaned data with a daily index.

    Parameters:
        df: Cleaned daily DataFrame containing sales data.
        sales_col: Name of the sales column to extract.
        date_col: Name of the date column to enforce daily indexing.

    Returns:
        A Pandas Series of daily sales values indexed by date.
    """
    # Ensure the data has a proper daily index before extracting.
    daily = ensure_daily_index(df, date_col=date_col)

    # Validate the expected sales column exists.
    if sales_col not in daily.columns:
        raise KeyError(f"Expected sales column '{sales_col}' not found")
    
    # Convert sales values to float to support modeling.
    return daily[sales_col].astype(float)


def build_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar-based features to the daily time series DataFrame.

    Parameters:
        df: Daily DataFrame to enrich with calendar features.
        date_col: Name of the column containing dates.

    Returns:
        DataFrame enriched with day-of-week, weekend, month, quarter, and month boundary indicators.
    """
    # Make sure the DataFrame is indexed by a daily datetime index.
    df = ensure_daily_index(df, date_col=date_col)
    df = df.copy()

    # Add categorical and binary calendar features for analysis or exogenous modeling.
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df.index.day
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)

    return df


def build_lag_features(
    df: pd.DataFrame,
    sales_col: str = "daily_total_sales",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Create lag and rolling statistics features using only past values.

    Parameters:
        df: Daily DataFrame with sales values.
        sales_col: Column name of the sales series.
        lags: List of lag offsets to create.
        rolling_windows: List of rolling window sizes for mean and std statistics.

    Returns:
        DataFrame containing lag and rolling features aligned with the original dates.
    """
    df = df.copy()

    if lags is None:
        lags = [1, 7, 14, 28]
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]

    # Create lag features based on prior sales values.
    for lag in lags:
        df[f"lag_{lag}"] = df[sales_col].shift(lag)

    # Create rolling statistics from prior days only, to avoid leakage.
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df[sales_col].shift(1).rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df[sales_col].shift(1).rolling(window=window, min_periods=1).std().fillna(0.0)

    # Simple growth-rate feature comparing last day to last week.
    df["recent_growth_rate"] = df[f"lag_1"] / df[f"lag_7"] - 1
    return df


def split_time_series(
    df: pd.DataFrame,
    validation_days: int = 30,
    test_days: int = 30,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the daily series into train, validation, and test DataFrames chronologically.

    Parameters:
        df: Daily sales DataFrame.
        validation_days: Number of days to reserve for validation.
        test_days: Number of days to reserve for final testing.
        date_col: Date column to enforce daily indexing.

    Returns:
        A tuple of (train, validation, test) DataFrames.
    """
    daily = ensure_daily_index(df, date_col=date_col)
    if len(daily) < validation_days + test_days + 1:
        raise ValueError("Not enough data for the requested train/validation/test split")

    # The final window is the test set, preceding it is the validation set.
    test = daily.iloc[-test_days:]
    validation = daily.iloc[-(test_days + validation_days):-test_days]
    train = daily.iloc[: - (validation_days + test_days)]
    return train, validation, test


def naive_forecast(train_series: pd.Series, forecast_steps: int) -> pd.Series:
    """
    Produce a naïve forecast equal to the last observed value.

    Parameters:
        train_series: Historical sales series for training.
        forecast_steps: Number of future days to forecast.

    Returns:
        A Pandas Series containing repeated last observed values.
    """
    last_value = train_series.iloc[-1]

    # Forecast every future day as the same as the last known day.
    return pd.Series(
        [last_value] * forecast_steps,
        index=pd.date_range(start=train_series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D"),
    )


def seasonal_naive_forecast(train_series: pd.Series, forecast_steps: int, season_length: int = 7) -> pd.Series:
    """
    Produce a seasonal-naïve forecast using the value from one seasonal period ago.

    Parameters:
        train_series: Historical sales series for training.
        forecast_steps: Number of future days to forecast.
        season_length: Seasonal period in days used for the seasonal-naïve forecast.

    Returns:
        A Pandas Series containing repeated values from the most recent seasonal window.
    """
    if len(train_series) < season_length:
        raise ValueError("Not enough history for seasonal-naïve forecasting")
    last_season = train_series.iloc[-season_length:]
    repeats = int(np.ceil(forecast_steps / season_length))
    forecast_values = np.tile(last_season.values, repeats)[:forecast_steps]

    # Repeat last season values to fill the forecast horizon.
    return pd.Series(
        forecast_values,
        index=pd.date_range(start=train_series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D"),
    )


def fit_sarima_model(
    train_series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    exog: Optional[pd.DataFrame] = None,
    trend: str = "c",
    maxiter: int = 1000,
) -> SARIMAX:
    """
    Fit a SARIMA model to the training series and return the fitted result.

    Parameters:
        train_series: Daily sales series to fit the model on.
        order: Non-seasonal ARIMA order (p, d, q).
        seasonal_order: Seasonal ARIMA order (P, D, Q, s).
        exog: Optional exogenous regressors for SARIMAX.
        trend: Trend specification for the model.
        maxiter: Maximum number of iterations for solver convergence.

    Returns:
        A fitted SARIMAXResults object.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        exog=exog,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    # Fit the model without printing solver output.
    fitted_model = model.fit(disp=False, maxiter=maxiter)
    return fitted_model


def generate_sarima_forecast(
    model: SARIMAX,
    steps: int,
    exog_forecast: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate point forecasts and confidence intervals from a fitted SARIMA model.

    Parameters:
        model: A fitted SARIMAX model instance.
        steps: Number of future days to forecast.
        exog_forecast: Optional exogenous data for the forecast horizon.
        alpha: Significance level for prediction intervals.

    Returns:
        DataFrame containing forecast, lower_bound, and upper_bound.
    """
    prediction = model.get_forecast(steps=steps, exog=exog_forecast)
    forecast = prediction.predicted_mean
    intervals = prediction.conf_int(alpha=alpha)

    forecast_df = pd.DataFrame(
        {
            "forecast": forecast,
            "lower_bound": intervals.iloc[:, 0],
            "upper_bound": intervals.iloc[:, 1],
        }
    )

    # Assign the forecast horizon as the next consecutive daily dates.
    forecast_df.index = pd.date_range(
        start=model.data.endog.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D"
    )
    return forecast_df


def clip_negative_forecasts(forecast_df: pd.DataFrame, forecast_col: str = "forecast") -> pd.DataFrame:
    """
    Clip negative forecast values to zero for realistic sales forecasts.

    Parameters:
        forecast_df: Forecast data containing predicted and interval columns.
        forecast_col: Name of the forecast column to clip.

    Returns:
        DataFrame with negative values clipped to zero.
    """
    forecast_df = forecast_df.copy()

    # Sales cannot be negative, so clamp the forecast and interval bounds.
    forecast_df[forecast_col] = forecast_df[forecast_col].clip(lower=0.0)
    forecast_df["lower_bound"] = forecast_df["lower_bound"].clip(lower=0.0)
    forecast_df["upper_bound"] = forecast_df["upper_bound"].clip(lower=0.0)
    return forecast_df


def calculate_forecast_metrics(actual: pd.Series, prediction: pd.Series) -> Dict[str, float]:
    """
    Calculate key forecast error metrics for evaluation.

    Returns MAE, RMSE, WAPE, and MAPE. MAPE is only meaningful when actual
    values are all nonzero.

    Parameters:
        actual: Actual observed sales values.
        prediction: Forecasted sales values.

    Returns:
        Dictionary of evaluation metrics.
    """
    actual = actual.astype(float)
    prediction = prediction.astype(float)
    mae = np.mean(np.abs(actual - prediction))
    rmse = np.sqrt(np.mean((actual - prediction) ** 2))

    # WAPE is scale-aware and uses absolute actual sales as the denominator.
    wape = np.sum(np.abs(actual - prediction)) / np.sum(np.abs(actual)) if np.sum(np.abs(actual)) != 0 else np.nan
    
    # Only compute MAPE if no actual value is zero.
    mape = np.mean(np.abs((actual - prediction) / actual)) * 100 if np.all(actual != 0) else np.nan
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE": float(wape),
        "MAPE": float(mape) if not np.isnan(mape) else np.nan,
    }


def aggregate_forecast_to_monthly(forecast_df: pd.DataFrame, forecast_col: str = "forecast") -> pd.DataFrame:
    """
    Aggregate daily forecast output into monthly totals.

    Parameters:
        forecast_df: Daily forecast DataFrame that includes lower and upper bounds.
        forecast_col: Name of the point forecast column to aggregate.

    Returns:
        Monthly aggregated forecast DataFrame with summed point and interval totals.
    """
    monthly = forecast_df[[forecast_col, "lower_bound", "upper_bound"]].copy()

    # Create a monthly period label from the daily index.
    monthly["month"] = monthly.index.to_period("M")

    # Sum daily forecasts and interval bounds by month.
    monthly = monthly.groupby("month").sum()

    monthly.index = monthly.index.to_timestamp("M")

    monthly = monthly.rename(
        columns={
            forecast_col: "monthly_forecast",
            "lower_bound": "monthly_lower",
            "upper_bound": "monthly_upper",
        }
    )

    return monthly


def build_forecast_output(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a final forecast DataFrame with risk level and forecast month metadata.

    Parameters:
        forecast_df: Daily forecast DataFrame with point and interval columns.

    Returns:
        DataFrame with added forecast month label and risk level.
    """
    output = forecast_df.copy()

    # Add a month label for each forecast day.
    output["forecast_month"] = output.index.to_period("M").astype(str)

    # Use interval width as a simple risk proxy.
    width = output["upper_bound"] - output["lower_bound"]

    output["risk_level"] = pd.cut(
        width,
        bins=[-1, 0.1, 0.25, np.inf],
        labels=["low", "medium", "high"],
    )
    
    # Ensure all rows have a risk label, defaulting medium if needed.
    output["risk_level"] = output["risk_level"].cat.add_categories(["medium", "high"]).fillna("medium")
    return output
