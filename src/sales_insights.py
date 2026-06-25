"""
Sales insights module for CanAI Café forecasting.

Provides functions to compute and display a human-readable summary of
historical sales performance and forecast accuracy after a model run.

Functions
---------
calculate_sales_insights  – compute all insight metrics and return a dict
print_sales_insights      – format and print the insights to the terminal
insights_to_dataframe     – convert the insights dict to a CSV-ready DataFrame
prompt_save_insights      – interactive y/n prompt for saving the CSV
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical weekday order used when ranking busiest day (Mon → Sun).
_WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------

def calculate_sales_insights(
    daily_sales: pd.Series,
    actual_values: Optional[pd.Series] = None,
    forecast_values: Optional[pd.Series] = None,
    wape: Optional[float] = None,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
) -> dict:
    """
    Calculate sales insights from the historical daily sales series and
    optional aligned actual/forecast evaluation arrays.

    Parameters:
        daily_sales: Complete daily sales series with a DatetimeIndex.
            Used for all historical statistics.
        actual_values: Actual sales over the evaluation period (validation or test).
            Required for forecast-vs-actual comparison.
        forecast_values: Forecasted sales over the evaluation period.
            Required for forecast-vs-actual comparison.
        wape: Pre-computed WAPE value (0–1 scale). When provided the accuracy
            metric is derived directly from it. When omitted it is re-computed
            from actual_values and forecast_values.
        period_start: Optional start date (inclusive) to filter historical stats.
            If None, uses the full range of daily_sales.
        period_end: Optional end date (inclusive) to filter historical stats.
            If None, uses the full range of daily_sales.

    Returns:
        Dictionary with all insight fields. Unavailable fields are None.

    Raises:
        TypeError: When daily_sales does not have a DatetimeIndex.
        ValueError: When actual_values and forecast_values have different lengths,
            or when the period range is invalid.
    """
    # Guard: the series must carry a DatetimeIndex so weekday grouping works.
    if not isinstance(daily_sales.index, pd.DatetimeIndex):
        raise TypeError("daily_sales must have a DatetimeIndex.")

    # Drop NaN rows so statistics reflect only days with valid recorded sales.
    sales = daily_sales.dropna()

    if sales.empty:
        raise ValueError("daily_sales contains no valid (non-NaN) values.")

    # Filter to the specified period if provided (for monthly or custom insights).
    if period_start is not None or period_end is not None:
        if period_start is not None and period_end is not None:
            if period_start > period_end:
                raise ValueError(
                    f"period_start ({period_start}) cannot be after period_end ({period_end})."
                )
        sales = sales.loc[period_start:period_end]
        if sales.empty:
            raise ValueError(
                f"No data found in the specified period "
                f"({period_start} to {period_end})."
            )

    # -----------------------------------------------------------------------
    # Historical statistics
    # Computed from the FULL cleaned series, not just the evaluation period.
    # -----------------------------------------------------------------------

    total_sales = float(sales.sum())            # sum of all recorded daily sales
    average_daily_sales = float(sales.mean())   # mean across all trading days

    best_date = sales.idxmax()      # date with the single highest daily revenue
    best_amount = float(sales.max())
    worst_date = sales.idxmin()     # date with the single lowest daily revenue
    worst_amount = float(sales.min())

    # Group by weekday name and take the average rather than the total so that
    # weekdays with more occurrences do not unfairly dominate the ranking.
    weekday_avg = sales.groupby(sales.index.day_name()).mean()

    # Reorder to canonical Mon–Sun so the display reads naturally.
    ordered_days = [d for d in _WEEKDAY_ORDER if d in weekday_avg.index]
    weekday_avg = weekday_avg.reindex(ordered_days)

    busiest_weekday = weekday_avg.idxmax()          # name of the busiest day
    busiest_weekday_avg = float(weekday_avg.max())  # its average daily revenue

    # Record the covered date range for the period label in the display.
    date_range_start = sales.index.min()
    date_range_end = sales.index.max()

    # -----------------------------------------------------------------------
    # Forecast-vs-actual comparison (optional)
    # Only populated when actual and forecast arrays are supplied.
    # -----------------------------------------------------------------------

    # Initialise optional fields to None; they are filled in below if data is available.
    forecast_percentage_diff: Optional[float] = None
    forecast_interpretation: Optional[str] = None
    forecast_accuracy: Optional[float] = None
    actual_total: Optional[float] = None
    forecast_total: Optional[float] = None

    if actual_values is not None and forecast_values is not None:
        # Convert to plain NumPy float arrays for consistent arithmetic.
        actual_arr = pd.Series(actual_values).dropna().to_numpy(dtype=float)
        forecast_arr = pd.Series(forecast_values).dropna().to_numpy(dtype=float)

        # Both arrays must cover the same number of evaluation periods.
        if len(actual_arr) != len(forecast_arr):
            raise ValueError(
                f"actual_values length ({len(actual_arr)}) does not match "
                f"forecast_values length ({len(forecast_arr)})."
            )

        actual_total = float(actual_arr.sum())
        forecast_total = float(forecast_arr.sum())

        # Percentage difference: positive → over-forecast, negative → under-forecast.
        if actual_total != 0:
            forecast_percentage_diff = ((forecast_total - actual_total) / actual_total) * 100

            if forecast_percentage_diff > 0:
                forecast_interpretation = (
                    f"The model over-forecasted total sales by "
                    f"{abs(forecast_percentage_diff):.2f}%."
                )
            elif forecast_percentage_diff < 0:
                forecast_interpretation = (
                    f"The model under-forecasted total sales by "
                    f"{abs(forecast_percentage_diff):.2f}%."
                )
            else:
                forecast_interpretation = "The model forecast exactly matched actual sales."
        else:
            # Division by zero guard: actual total is zero so the ratio is undefined.
            forecast_percentage_diff = None
            forecast_interpretation = "Cannot compute percentage difference: actual total is zero."

        # Derive WAPE from aligned arrays when a pre-computed value was not passed in.
        if wape is None:
            abs_actual_sum = np.sum(np.abs(actual_arr))
            wape = (
                float(np.sum(np.abs(actual_arr - forecast_arr)) / abs_actual_sum)
                if abs_actual_sum != 0
                else None
            )

    # Forecast accuracy: WAPE-based, clamped to [0, 100] so it never goes negative.
    if wape is not None:
        forecast_accuracy = max(0.0, 100.0 * (1.0 - wape))

    # Return all metrics as a flat dict so callers can use or export any subset.
    return {
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "total_sales": total_sales,
        "average_daily_sales": average_daily_sales,
        "best_date": best_date,
        "best_amount": best_amount,
        "worst_date": worst_date,
        "worst_amount": worst_amount,
        "busiest_weekday": busiest_weekday,
        "busiest_weekday_avg": busiest_weekday_avg,
        "actual_total": actual_total,
        "forecast_total": forecast_total,
        "forecast_percentage_diff": forecast_percentage_diff,
        "forecast_interpretation": forecast_interpretation,
        "wape": wape,
        "forecast_accuracy": forecast_accuracy,
    }


# ---------------------------------------------------------------------------
# Period filtering and validation
# ---------------------------------------------------------------------------

def get_available_date_range(daily_sales: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the minimum and maximum dates available in the sales data.

    Parameters:
        daily_sales: Complete daily sales series with a DatetimeIndex.

    Returns:
        Tuple of (min_date, max_date) after dropping NaN values.

    Raises:
        TypeError: When daily_sales does not have a DatetimeIndex.
        ValueError: When daily_sales contains no valid (non-NaN) values.
    """
    if not isinstance(daily_sales.index, pd.DatetimeIndex):
        raise TypeError("daily_sales must have a DatetimeIndex.")

    sales = daily_sales.dropna()
    if sales.empty:
        raise ValueError("daily_sales contains no valid (non-NaN) values.")

    return sales.index.min(), sales.index.max()


def prompt_for_period(daily_sales: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Prompt the user to enter a month/year (YYYY-MM format) and return the start
    and end dates (first and last day of that month). Validates that the month
    exists in the available dataset.

    Parameters:
        daily_sales: Complete daily sales series with a DatetimeIndex.
            Used to validate the entered period exists in the data.

    Returns:
        Tuple of (period_start, period_end) representing the first and last day
        of the requested month.

    Raises:
        Errors are caught and reprompted; only successful validation returns.
    """
    min_date, max_date = get_available_date_range(daily_sales)

    while True:
        print(
            f"\nAvailable data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            flush=True,
        )
        month_input = input(
            "Enter a month for historical insights (YYYY-MM format, e.g., 2023-03): "
        ).strip()

        try:
            # Parse the input as YYYY-MM and create the first day of that month.
            period_start = pd.Timestamp(month_input + "-01")

            # Compute the last day of the month (first day of next month minus 1 day).
            if period_start.month == 12:
                period_end = (
                    pd.Timestamp(year=period_start.year + 1, month=1, day=1)
                    - pd.Timedelta(days=1)
                )
            else:
                period_end = (
                    pd.Timestamp(
                        year=period_start.year,
                        month=period_start.month + 1,
                        day=1,
                    )
                    - pd.Timedelta(days=1)
                )

            # Validate that the month falls within the available data range.
            if period_start > max_date or period_end < min_date:
                print(
                    f"Error: Month {month_input} is outside the available data range. "
                    f"Available: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}",
                    flush=True,
                )
                continue

            # Clamp the period to the available data range (in case user enters
            # a partial month at the boundaries).
            period_start = max(period_start, min_date)
            period_end = min(period_end, max_date)

            print(
                f"Selected: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                flush=True,
            )
            return period_start, period_end

        except (ValueError, pd.errors.ParserError):
            print(
                "Error: Invalid format. Please enter the month as YYYY-MM (e.g., 2023-03).",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def print_sales_insights(insights: dict) -> None:
    """
    Print a formatted sales insights summary to the terminal.

    Parameters:
        insights: Dictionary returned by calculate_sales_insights().
    """
    start = insights["date_range_start"]
    end = insights["date_range_end"]

    # Build a human-readable period label from the actual data range.
    # Single-year datasets show just the year; multi-year datasets show the full range.
    if start.year == end.year:
        period_label = f"{start.year}"
    else:
        period_label = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

    # ---- Section header ----
    print("\nSALES INSIGHTS", flush=True)
    print("==============", flush=True)

    # ---- Historical summary ----
    print(f"\nHistorical Sales ({period_label})", flush=True)
    print("-" * 40, flush=True)
    print(f"Total Sales:             ${insights['total_sales']:>12,.2f}", flush=True)
    print(f"Average Daily Sales:     ${insights['average_daily_sales']:>12,.2f}", flush=True)
    print(f"Busiest Day of the Week: {insights['busiest_weekday']}", flush=True)
    print(
        f"Average {insights['busiest_weekday']} Sales: "
        f"${insights['busiest_weekday_avg']:,.2f}",
        flush=True,
    )

    # ---- Best and worst trading days ----
    # strftime %B %d gives "January 05"; replace " 0" removes the leading zero on Windows.
    print("\nBest Sales Day", flush=True)
    print("-" * 40, flush=True)
    best_date_str = insights["best_date"].strftime("%B %d, %Y").replace(" 0", " ")
    print(f"Date:  {best_date_str}", flush=True)
    print(f"Sales: ${insights['best_amount']:,.2f}", flush=True)

    print("\nWorst Sales Day", flush=True)
    print("-" * 40, flush=True)
    worst_date_str = insights["worst_date"].strftime("%B %d, %Y").replace(" 0", " ")
    print(f"Date:  {worst_date_str}", flush=True)
    print(f"Sales: ${insights['worst_amount']:,.2f}", flush=True)

    # ---- Forecast performance (only shown when evaluation data was provided) ----
    if insights.get("actual_total") is not None:
        print("\nForecast Performance", flush=True)
        print("-" * 40, flush=True)
        print(f"Actual Sales:   ${insights['actual_total']:>12,.2f}", flush=True)
        print(f"Forecast Sales: ${insights['forecast_total']:>12,.2f}", flush=True)

        if insights["forecast_percentage_diff"] is not None:
            # Prefix a "+" on positive differences so over-forecasting is visually clear.
            sign = "+" if insights["forecast_percentage_diff"] > 0 else ""
            print(
                f"Forecast vs. Actual Difference: {sign}{insights['forecast_percentage_diff']:.2f}%",
                flush=True,
            )
            print(f"Interpretation: {insights['forecast_interpretation']}", flush=True)
        else:
            # Actual total was zero; display the explanation string instead.
            print(insights["forecast_interpretation"], flush=True)

    # ---- WAPE-based accuracy (shown whenever WAPE is available) ----
    if insights.get("forecast_accuracy") is not None:
        print(
            f"Forecast Accuracy Based on WAPE: {insights['forecast_accuracy']:.2f}%",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def insights_to_dataframe(insights: dict) -> pd.DataFrame:
    """
    Convert a sales insights dictionary into a two-column CSV-ready DataFrame.

    Each row represents one metric so the file is easy to read in any
    spreadsheet application.

    Parameters:
        insights: Dictionary returned by calculate_sales_insights().

    Returns:
        DataFrame with 'metric' and 'value' columns.
    """
    # Format dates as readable strings; strip the leading zero on single-digit days.
    best_date_str = insights["best_date"].strftime("%B %d, %Y").replace(" 0", " ")
    worst_date_str = insights["worst_date"].strftime("%B %d, %Y").replace(" 0", " ")

    start = insights["date_range_start"]
    end = insights["date_range_end"]
    # Match the period label used in the terminal display.
    period_label = (
        str(start.year) if start.year == end.year
        else f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    )

    # Build rows in display order so the CSV reads top-to-bottom like the terminal output.
    rows = [
        ("Period", period_label),
        ("Total Sales ($)", f"{insights['total_sales']:,.2f}"),
        ("Average Daily Sales ($)", f"{insights['average_daily_sales']:,.2f}"),
        ("Busiest Day of the Week", insights["busiest_weekday"]),
        (f"Average {insights['busiest_weekday']} Sales ($)", f"{insights['busiest_weekday_avg']:,.2f}"),
        ("Best Sales Date", best_date_str),
        ("Best Sales Amount ($)", f"{insights['best_amount']:,.2f}"),
        ("Worst Sales Date", worst_date_str),
        ("Worst Sales Amount ($)", f"{insights['worst_amount']:,.2f}"),
    ]

    # Append forecast comparison rows only when evaluation data was available.
    if insights.get("actual_total") is not None:
        rows += [
            ("Actual Period Sales ($)", f"{insights['actual_total']:,.2f}"),
            ("Forecast Period Sales ($)", f"{insights['forecast_total']:,.2f}"),
        ]
        if insights["forecast_percentage_diff"] is not None:
            sign = "+" if insights["forecast_percentage_diff"] > 0 else ""
            rows.append(
                ("Forecast vs. Actual Difference (%)", f"{sign}{insights['forecast_percentage_diff']:.2f}%")
            )
            rows.append(("Interpretation", insights["forecast_interpretation"]))

    # WAPE-based accuracy is the final row when present.
    if insights.get("forecast_accuracy") is not None:
        rows.append(("Forecast Accuracy Based on WAPE (%)", f"{insights['forecast_accuracy']:.2f}%"))

    return pd.DataFrame(rows, columns=["metric", "value"])


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

def prompt_save_insights() -> bool:
    """
    Ask the user whether to save the sales insights to a CSV file.

    Returns:
        True when the user confirms, False otherwise.
    """
    while True:
        answer = input(
            "Save sales insights to CSV? Enter 'y' to save or 'n' to skip: "
        ).strip().lower()
        if answer in {"y", "n"}:
            return answer == "y"
        print("Please enter 'y' or 'n'.")  # reprompt on unexpected input

