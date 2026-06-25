"""
Modeling and post-cleaning preprocessing for CanAI Café forecasting.

This module provides functions to prepare time series data, build features, fit SARIMA models, generate forecasts, and evaluate forecast performance. 
It is designed for daily sales data and includes utilities for handling missing dates, creating lag and rolling features, and aggregating forecasts to monthly totals.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


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

    # Convert the date column to a datetime index. Any value that cannot be parsed
    # becomes NaT, which allows us to detect and remove invalid rows.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Remove rows that have no valid date, because those cannot be used in a time series.
    df = df.dropna(subset=[date_col])

    # Set the normalized datetime column as the index and make sure rows are ordered.
    df = df.set_index(date_col).sort_index()

    # Enforce a daily frequency so that all calendar dates appear explicitly in the index.
    # Missing dates become NaN and can be filled or used to create complete series later.
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
    # Ensure the data has a proper daily index before extracting, which also makes
    # the series continuous and easier to model with SARIMA.
    daily = ensure_daily_index(df, date_col=date_col)

    # Confirm we have the sales column that the caller expects. This protects against
    # upstream schema changes or malformed data frames.
    if sales_col not in daily.columns:
        raise KeyError(f"Expected sales column '{sales_col}' not found")

    # Return the sales series as floats, which is required by the forecasting routines.
    return daily[sales_col].astype(float)


def aggregate_daily_sales_by_group(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Transaction Date",
    sales_col: str = "Total Spent",
) -> pd.DataFrame:
    """
    Aggregate raw transactions to daily sales totals at the requested group level.

    Parameters:
        df: Raw transaction data.
        group_cols: Columns to group by, e.g. ["Province"] or ["Item"].
        date_col: Date column name in the raw data.
        sales_col: Sales amount column name in the raw data.

    Returns:
        DataFrame with grouped daily totals and normalized dates.
    """
    df = df.copy()
    if group_cols is None:
        group_cols = []

    # Parse raw dates and quantities so invalid rows can be dropped cleanly.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

    # Remove any transactions missing either a valid date, sales amount, or required group key.
    required_cols = [date_col, sales_col] + group_cols
    df = df.dropna(subset=required_cols)

    # Normalize the transaction timestamp to a calendar day and group on that day.
    df = df.assign(date=df[date_col].dt.normalize())

    # Aggregate daily sales at the requested group level.
    grouped = (
        df.groupby(group_cols + ["date"], as_index=False)[sales_col]
        .sum()
        .rename(columns={sales_col: "daily_total_sales"})
    )

    # Sort results by group and date so downstream code can assume ordered data.
    return grouped.sort_values(group_cols + ["date"]).reset_index(drop=True)


def complete_grouped_daily_sales(
    grouped_df: pd.DataFrame,
    group_cols: List[str],
    date_col: str = "date",
    sales_col: str = "daily_total_sales",
) -> pd.DataFrame:
    """
    Fill missing dates for each group so every group has a complete daily series.

    Parameters:
        grouped_df: Aggregated grouped sales data by date.
        group_cols: Columns that define each group.
        date_col: Name of the date column in the grouped DataFrame.
        sales_col: Name of the daily sales total column.

    Returns:
        DataFrame with a complete daily date range per group.
    """
    grouped_df = grouped_df.copy()

    # Normalize the date column for every row before building complete calendars.
    grouped_df[date_col] = pd.to_datetime(grouped_df[date_col], errors="coerce")
    grouped_df = grouped_df.dropna(subset=[date_col] + group_cols)

    frames: List[pd.DataFrame] = []
    for key_values, group in grouped_df.groupby(group_cols, sort=False):
        # Build a full daily index for the current group and preserve any missing days.
        group = group.set_index(date_col).sort_index()
        full_index = pd.date_range(start=group.index.min(), end=group.index.max(), freq="D")

        # Reindex the group to include every date in the range. Fill only the sales
        # column after reindexing so string-type group labels are not coerced.
        group = group.reindex(full_index)
        group = group.reset_index().rename(columns={"index": date_col})
        group[sales_col] = group[sales_col].fillna(0.0)

        # Restore group identity values after the reindex operation.
        if isinstance(key_values, tuple):
            for col, value in zip(group_cols, key_values):
                group[col] = value
        else:
            group[group_cols[0]] = key_values

        frames.append(group)

    if not frames:
        return pd.DataFrame(columns=group_cols + [date_col, sales_col])

    result = pd.concat(frames, ignore_index=True)

    # Return the grouped DataFrame with the same ordered columns used throughout the pipeline.
    return result[group_cols + [date_col, sales_col]]


def clip_negative_forecasts(forecast_df: pd.DataFrame, forecast_col: str = "forecast") -> pd.DataFrame:
    """
    Replace negative forecast values with zero while preserving intervals.

    Parameters:
        forecast_df: DataFrame with forecast values.
        forecast_col: Name of the forecast column to clip.

    Returns:
        DataFrame with clipped forecast values.
    """
    result = forecast_df.copy()
    if forecast_col in result.columns:
        result[forecast_col] = result[forecast_col].clip(lower=0.0)
    return result


def forecast_grouped_sales(
    grouped_daily_sales: pd.DataFrame,
    group_cols: List[str],
    forecast_steps: int,
    date_col: str = "date",
    sales_col: str = "daily_total_sales",
    min_history: int = 14,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    trend: Optional[str] = "c",
) -> pd.DataFrame:
    """
    Generate SARIMA forecasts for each group in grouped daily sales.

    Parameters:
        grouped_daily_sales: Grouped daily sales DataFrame.
        group_cols: Columns that define each group.
        forecast_steps: Forecast horizon in days.
        min_history: Minimum history required to fit a group model.
        order: Non-seasonal ARIMA order.
        seasonal_order: Seasonal ARIMA order.
        trend: Trend parameter for SARIMA.

    Returns:
        DataFrame with forecasts and group labels for each horizon day.
    """
    forecasts: List[pd.DataFrame] = []
    for key_values, group in grouped_daily_sales.groupby(group_cols, sort=False):
        if len(group) < min_history:
            # Skip groups that do not have enough historical points to fit a SARIMA model.
            continue

        # Extract the group's daily sales series, filling any NaNs created by resampling.
        series = extract_daily_sales_series(group, sales_col=sales_col, date_col=date_col).fillna(0.0)

        # Fit a SARIMA model for this group and generate the requested forecast horizon.
        fitted = fit_sarima_model(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
        )
        forecast_df = generate_sarima_forecast(fitted, steps=forecast_steps)

        # Force negative predictions to zero so forecasts remain realistic for sales data.
        forecast_df = clip_negative_forecasts(forecast_df)

        # Move the index back into a date column to keep grouped metadata aligned.
        forecast_df = forecast_df.reset_index().rename(columns={"index": date_col})

        # Attach the group identifier columns to the forecast results.
        if isinstance(key_values, tuple):
            for col, value in zip(group_cols, key_values):
                forecast_df[col] = value
        else:
            forecast_df[group_cols[0]] = key_values

        forecasts.append(forecast_df)

    if not forecasts:
        return pd.DataFrame(columns=group_cols + [date_col, "forecast", "lower_bound", "upper_bound"])

    return pd.concat(forecasts, ignore_index=True)


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

    Returns:s
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

    # Assign the forecast horizon using the model's original row labels.
    # `model.data.endog` can be a numpy array, so use row_labels or orig_endog index.
    if hasattr(model.data, "row_labels") and model.data.row_labels is not None:
        last_date = model.data.row_labels[-1]
    elif hasattr(model.data, "orig_endog") and hasattr(model.data.orig_endog, "index"):
        last_date = model.data.orig_endog.index[-1]
    else:
        raise ValueError("Unable to infer the forecast start date from the model data.")

    forecast_df.index = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )
    return forecast_df


def get_model_diagnostics(
    fitted_model: Any,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    trend: Optional[str],
) -> Dict[str, Any]:
    """Return a compact diagnostics summary for a fitted SARIMA model."""
    diagnostics: Dict[str, Any] = {
        "order": order,
        "seasonal_order": seasonal_order,
        "trend": trend,
        "aic": getattr(fitted_model, "aic", np.nan),
        "bic": getattr(fitted_model, "bic", np.nan),
        "converged": bool(getattr(getattr(fitted_model, "mle_retvals", {}), "get", lambda k, d: d)("converged", True)),
    }
    residuals = getattr(fitted_model, "resid", None)
    if residuals is not None:
        diagnostics["residual_mean"] = float(np.mean(residuals))
        diagnostics["residual_std"] = float(np.std(residuals, ddof=0))
    else:
        diagnostics["residual_mean"] = np.nan
        diagnostics["residual_std"] = np.nan
    return diagnostics


def tune_sarima_parameters(
    train_series: pd.Series,
    validation_series: pd.Series,
    orders: Optional[List[Tuple[int, int, int]]] = None,
    seasonal_orders: Optional[List[Tuple[int, int, int, int]]] = None,
    trends: Optional[List[Optional[str]]] = None,
    maxiter: int = 1000,
    suppress_warnings: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Search a small SARIMA parameter space and select the best configuration by validation WAPE."""
    if orders is None:
        orders = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
    if seasonal_orders is None:
        seasonal_orders = [
            (0, 0, 1, 7),
            (1, 0, 0, 7),
            (1, 0, 1, 7),
            (0, 1, 1, 7),
            (1, 1, 1, 7),
        ]
    if trends is None:
        trends = [None, "c"]

    rows: List[Dict[str, Any]] = []
    validation_series = validation_series.astype(float)

    for order in orders:
        for seasonal_order in seasonal_orders:
            for trend in trends:
                row: Dict[str, Any] = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "trend": trend,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "WAPE": np.nan,
                    "MAPE": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "converged": False,
                    "status": "not evaluated",
                }
                try:
                    with warnings.catch_warnings():
                        if suppress_warnings:
                            warnings.simplefilter("ignore")
                        fitted = fit_sarima_model(
                            train_series,
                            order=order,
                            seasonal_order=seasonal_order,
                            trend=trend,
                            maxiter=maxiter,
                        )

                    forecast_df = generate_sarima_forecast(fitted, steps=len(validation_series))
                    forecast_df = clip_negative_forecasts(forecast_df)
                    prediction = forecast_df["forecast"].copy()
                    prediction.index = validation_series.index
                    metrics = calculate_forecast_metrics(validation_series, prediction)

                    row.update(
                        {
                            "MAE": metrics["MAE"],
                            "RMSE": metrics["RMSE"],
                            "WAPE": metrics["WAPE"],
                            "MAPE": metrics["MAPE"],
                            "AIC": getattr(fitted, "aic", np.nan),
                            "BIC": getattr(fitted, "bic", np.nan),
                            "converged": bool(
                                getattr(getattr(fitted, "mle_retvals", {}), "get", lambda k, d: d)("converged", True)
                            ),
                            "status": "success",
                        }
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    row["status"] = f"error: {type(exc).__name__}"
                except Exception as exc:
                    row["status"] = f"error: {type(exc).__name__}"
                rows.append(row)

    results = pd.DataFrame(rows)
    success_rows = results[results["status"] == "success"].copy()
    if success_rows.empty:
        raise ValueError("SARIMA parameter tuning failed for all candidate configurations.")

    best = success_rows.sort_values(["WAPE", "MAE"], ascending=True).iloc[0]
    best_config = {
        "order": tuple(best["order"]),
        "seasonal_order": tuple(best["seasonal_order"]),
        "trend": best["trend"],
        "MAE": float(best["MAE"]),
        "RMSE": float(best["RMSE"]),
        "WAPE": float(best["WAPE"]),
        "MAPE": float(best["MAPE"]),
        "AIC": float(best["AIC"]),
        "BIC": float(best["BIC"]),
        "converged": bool(best["converged"]),
        "status": best["status"],
    }
    return best_config, results


def rolling_time_series_validation(
    series: pd.Series,
    validation_days: int = 30,
    max_folds: int = 3,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
    trend: Optional[str] = "c",
    maxiter: int = 1000,
    suppress_warnings: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run an expanding-window time-series validation using SARIMA on multiple folds."""
    series = series.copy().astype(float)
    if series.index.freq is None:
        series = series.asfreq("D")

    n = len(series)
    possible_folds = max(1, (n // validation_days) - 1)
    folds = min(max_folds, possible_folds)
    if folds < 1:
        raise ValueError("Not enough data for rolling validation with the requested fold configuration.")

    rows: List[Dict[str, Any]] = []
    for fold_number in range(1, folds + 1):
        train_end = n - validation_days * (folds - fold_number + 1)
        validation_start = train_end
        validation_end = validation_start + validation_days

        train_fold = series.iloc[:train_end]
        validation_fold = series.iloc[validation_start:validation_end]
        status = "success"

        try:
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.simplefilter("ignore")
                fitted = fit_sarima_model(
                    train_fold,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    maxiter=maxiter,
                )

            forecast_df = generate_sarima_forecast(fitted, steps=len(validation_fold))
            forecast_df = clip_negative_forecasts(forecast_df)
            prediction = forecast_df["forecast"].copy()
            prediction.index = validation_fold.index
            metrics = calculate_forecast_metrics(validation_fold, prediction)
        except (ValueError, np.linalg.LinAlgError) as exc:
            metrics = {"MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan, "MAPE": np.nan}
            status = f"error: {type(exc).__name__}"
        except Exception as exc:
            metrics = {"MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan, "MAPE": np.nan}
            status = f"error: {type(exc).__name__}"

        rows.append(
            {
                "fold": fold_number,
                "train_start": series.index[0],
                "train_end": series.index[train_end - 1],
                "validation_start": validation_fold.index[0],
                "validation_end": validation_fold.index[-1],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
                "MAPE": metrics["MAPE"],
                "status": status,
            }
        )

    results = pd.DataFrame(rows)
    valid = results[results["status"] == "success"]
    average_metrics = {
        "MAE_mean": float(valid["MAE"].mean()) if not valid.empty else np.nan,
        "RMSE_mean": float(valid["RMSE"].mean()) if not valid.empty else np.nan,
        "WAPE_mean": float(valid["WAPE"].mean()) if not valid.empty else np.nan,
        "MAPE_mean": float(valid["MAPE"].mean()) if not valid.empty else np.nan,
        "WAPE_std": float(valid["WAPE"].std(ddof=0)) if len(valid) > 1 else 0.0,
    }
    return results, average_metrics


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

    # Ensure all rows have a risk label, defaulting to medium if missing.
    risk_level = output["risk_level"]
    desired_categories = ["low", "medium", "high"]
    missing_categories = [c for c in desired_categories if c not in risk_level.cat.categories]
    if missing_categories:
        risk_level = risk_level.cat.add_categories(missing_categories)
    output["risk_level"] = risk_level.fillna("medium")
    return output
