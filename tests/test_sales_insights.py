"""
Tests for src/sales_insights.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sales_insights import calculate_sales_insights, print_sales_insights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daily_sales() -> pd.Series:
    """One full year of deterministic daily sales for predictable assertions."""
    index = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    rng = np.random.default_rng(seed=0)
    values = rng.uniform(50, 500, size=len(index))
    # Force a known best and worst day
    values[14] = 900.0   # 2023-01-15 → best
    values[30] = 10.0    # 2023-01-31 → worst
    return pd.Series(values, index=index, name="daily_total_sales")


@pytest.fixture()
def eval_actuals() -> pd.Series:
    index = pd.date_range("2023-12-01", periods=30, freq="D")
    return pd.Series(np.full(30, 200.0), index=index)


@pytest.fixture()
def eval_forecasts(eval_actuals) -> pd.Series:
    # 5% under-forecast
    return eval_actuals * 0.95


# ---------------------------------------------------------------------------
# Historical statistics
# ---------------------------------------------------------------------------

def test_total_sales_matches_series_sum(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["total_sales"] == pytest.approx(daily_sales.sum())


def test_average_daily_sales(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["average_daily_sales"] == pytest.approx(daily_sales.mean())


def test_best_date_and_amount(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["best_date"] == pd.Timestamp("2023-01-15")
    assert insights["best_amount"] == pytest.approx(900.0)


def test_worst_date_and_amount(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["worst_date"] == pd.Timestamp("2023-01-31")
    assert insights["worst_amount"] == pytest.approx(10.0)


def test_busiest_weekday_is_string(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["busiest_weekday"] in [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]


def test_busiest_weekday_uses_average_not_total(daily_sales):
    """Average weekday sales should determine busiest day, not total."""
    insights = calculate_sales_insights(daily_sales)
    weekday_avg = daily_sales.groupby(daily_sales.index.day_name()).mean()
    assert insights["busiest_weekday"] == weekday_avg.idxmax()
    assert insights["busiest_weekday_avg"] == pytest.approx(weekday_avg.max())


def test_date_range_matches_series_bounds(daily_sales):
    insights = calculate_sales_insights(daily_sales)
    assert insights["date_range_start"] == daily_sales.index.min()
    assert insights["date_range_end"] == daily_sales.index.max()


def test_historical_insights_use_full_series_not_eval_subset(daily_sales, eval_actuals, eval_forecasts):
    """Total sales must come from the full series, not just the eval period."""
    full_insights = calculate_sales_insights(daily_sales)
    partial_insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=eval_forecasts
    )
    assert full_insights["total_sales"] == pytest.approx(partial_insights["total_sales"])


# ---------------------------------------------------------------------------
# Forecast comparison
# ---------------------------------------------------------------------------

def test_percentage_diff_correct(daily_sales, eval_actuals, eval_forecasts):
    insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=eval_forecasts
    )
    expected = ((eval_forecasts.sum() - eval_actuals.sum()) / eval_actuals.sum()) * 100
    assert insights["forecast_percentage_diff"] == pytest.approx(expected)


def test_negative_diff_labelled_under_forecast(daily_sales, eval_actuals, eval_forecasts):
    insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=eval_forecasts
    )
    assert insights["forecast_percentage_diff"] < 0
    assert "under-forecasted" in insights["forecast_interpretation"]


def test_positive_diff_labelled_over_forecast(daily_sales, eval_actuals):
    over_forecasts = eval_actuals * 1.10
    insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=over_forecasts
    )
    assert insights["forecast_percentage_diff"] > 0
    assert "over-forecasted" in insights["forecast_interpretation"]


def test_zero_actual_total_handled_gracefully(daily_sales):
    zero_actual = pd.Series(np.zeros(10))
    zero_forecast = pd.Series(np.ones(10))
    insights = calculate_sales_insights(
        daily_sales, actual_values=zero_actual, forecast_values=zero_forecast
    )
    assert insights["forecast_percentage_diff"] is None
    assert insights["forecast_interpretation"] is not None


def test_length_mismatch_raises(daily_sales):
    actual = pd.Series(np.ones(10))
    forecast = pd.Series(np.ones(12))
    with pytest.raises(ValueError, match="length"):
        calculate_sales_insights(daily_sales, actual_values=actual, forecast_values=forecast)


# ---------------------------------------------------------------------------
# WAPE-based accuracy
# ---------------------------------------------------------------------------

def test_accuracy_from_provided_wape(daily_sales):
    insights = calculate_sales_insights(daily_sales, wape=0.2144)
    assert insights["forecast_accuracy"] == pytest.approx(100.0 * (1.0 - 0.2144))


def test_accuracy_never_below_zero(daily_sales):
    insights = calculate_sales_insights(daily_sales, wape=1.5)
    assert insights["forecast_accuracy"] == 0.0


def test_accuracy_computed_from_actuals_and_forecasts(daily_sales, eval_actuals, eval_forecasts):
    insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=eval_forecasts
    )
    assert insights["forecast_accuracy"] is not None
    assert 0.0 <= insights["forecast_accuracy"] <= 100.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_raises_on_non_datetime_index():
    bad_series = pd.Series([100.0, 200.0], index=[0, 1])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        calculate_sales_insights(bad_series)


def test_nan_values_are_dropped_before_calculation():
    index = pd.date_range("2023-01-01", periods=5, freq="D")
    series = pd.Series([100.0, np.nan, 200.0, np.nan, 300.0], index=index)
    insights = calculate_sales_insights(series)
    assert insights["total_sales"] == pytest.approx(600.0)
    assert insights["average_daily_sales"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# print_sales_insights smoke test
# ---------------------------------------------------------------------------

def test_print_sales_insights_runs_without_error(daily_sales, capsys):
    insights = calculate_sales_insights(daily_sales, wape=0.25)
    print_sales_insights(insights)
    captured = capsys.readouterr()
    assert "SALES INSIGHTS" in captured.out
    assert "Total Sales" in captured.out
    assert "Forecast Accuracy Based on WAPE" in captured.out


def test_print_sales_insights_with_forecast_comparison(daily_sales, eval_actuals, eval_forecasts, capsys):
    insights = calculate_sales_insights(
        daily_sales, actual_values=eval_actuals, forecast_values=eval_forecasts
    )
    print_sales_insights(insights)
    captured = capsys.readouterr()
    assert "Forecast Performance" in captured.out
    assert "under-forecasted" in captured.out
