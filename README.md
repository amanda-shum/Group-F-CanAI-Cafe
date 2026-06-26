# CanAI Café Sales Forecasting System

A comprehensive time-series forecasting system for CanAI Café sales built with **SARIMA models**, featuring monthly historical insights, data quality monitoring, and multi-model evaluation.

## Project Overview

This project forecasts daily sales for CanAI Café using transaction data from 2023. It implements a complete machine learning pipeline that includes:

- **Data Cleaning & Validation**: Remove duplicates, handle missing values, validate data quality
- **Exploratory Analysis**: Assess data distribution, completeness, and trends
- **SARIMA Forecasting**: Time-series modeling with seasonal ARIMA
- **Multi-Model Comparison**: Naive, Seasonal Naive, and SARIMA baselines
- **Monthly Historical Insights**: Analyze sales by month with interactive user prompts
- **6-Month Forecast Horizon**: Generate predictions with confidence intervals
- **Automated Reporting**: Export metrics, forecasts, and insights to CSV

## Project Structure

```
Group-F-CanAI-Cafe/
├── data/
│   ├── CanAI Cafe 2023 Sales Information.xlsx     # Raw sales data
│   ├── processed/
│   │   ├── cleaned_transactions.csv               # Main dataset (used by model)
│   │   ├── cleaned_generated_data.csv
│   │   └── feature_engineered_transactions.csv
│   ├── raw/
│   └── interim/
├── src/
│   ├── modeling.py                                # SARIMA model, forecasting, metrics
│   ├── data_quality.py                            # Data validation & quality checks
│   ├── clean_data.py                              # Data cleaning pipeline
│   ├── helpers.py                                 # Utility functions
│   └── sales_insights.py                          # Monthly sales analysis & insights
├── tests/
│   └── test_modeling.py                           # Unit tests for modeling
├── reports/                                       # Generated outputs
│   ├── validation_forecast.csv
│   ├── validation_metrics.csv
│   ├── daily_forecast.csv
│   ├── monthly_forecast.csv
│   ├── six_month_forecast.csv
│   ├── sales_insights.csv
│   └── forecast_metrics.csv
├── notebooks/                                     # Jupyter notebooks for exploration
├── run_model.py                                   # Main entry point
├── requirements.txt                               # Python dependencies
└── README.md                                      # This file
```

## Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Group-F-CanAI-Cafe
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell):**
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Pipeline

The main entry point is `run_model.py`. Use it with one of two commands:

#### **Training Pipeline**
```bash
python run_model.py train
```

This will:
- Load and prepare sales data
- Split into train/validation sets (305 days / 30 days)
- Tune SARIMA parameters and fit the model
- Compare with naive and seasonal naive baselines
- Evaluate on validation set
- Prompt for optional CSV exports

#### **Test Pipeline**
```bash
python run_model.py test
```

This will:
- Load and prepare all data
- Split into train/validation/test (305/30/30 days)
- Retrain on combined train + validation
- Evaluate on test set
- Generate a 6-month forecast (Dec 2023 - May 2024)
- Prompt for optional CSV exports

**Note:** Both pipelines will prompt you to:
1. Enter a month (YYYY-MM format) for historical sales insights
2. Choose whether to save results to CSV files

### Interactive Features

#### Monthly Historical Insights
When running either pipeline, you'll be prompted:
```
Available data range: 2023-01-01 to 2023-12-31
Enter a month for historical insights (YYYY-MM format, e.g., 2023-03):
```

Enter a month (e.g., `2023-10`) to get:
- **Total & Average Daily Sales** for that month
- **Busiest Day of Week** and average sales
- **Best/Worst Sales Days** with amounts
- **Forecast Performance** metrics (for test pipeline)
- **Forecast Accuracy** based on WAPE

After viewing insights, choose to save as CSV:
```
Save sales insights to CSV? Enter 'y' to save or 'n' to skip:
```

## Key Features

### 1. **SARIMA Forecasting**
- Automatic parameter tuning using grid search
- Seasonal decomposition for 7-day weekly patterns
- Confidence intervals for uncertainty quantification
- Handles daily sales aggregation with missing dates

### 2. **Data Quality Monitoring**
- Validates transaction dates, quantities, and totals
- Detects and handles missing/duplicate records
- Generates before/after quality reports
- Tracks data completeness by province and item

### 3. **Multi-Model Comparison**
- **Naive Forecast**: Simple last-value baseline
- **Seasonal Naive**: Day-of-week patterns
- **SARIMA**: Seasonal ARIMA with automatic tuning
- **Ridge Regression**: Menu items, provinces, calendar features, and recent sales history
- Compares using MAE, RMSE, WAPE, and MAPE metrics

### 4. **Ridge Regression Forecasting**
- Lag features: Previous sales (1, 7, 14 days)
- Rolling window features: Moving averages and standard deviations (7, 14 days)
- Calendar features: Day of week, month, weekend indicators
- Categorical features: Top 5 menu items and top 3 provinces
- Feature normalization and L2 regularization for stable predictions

### 5. **Monthly Historical Insights**
- User-selected time periods (by month, YYYY-MM)
- Daily aggregation with proper date validation
- Weekday-level analysis (busiest days)
- Best/worst performing days with amounts
- CSV export for easy sharing

### 6. **Extended Forecasting**
- 6-month forward forecast from end of training data
- Monthly aggregation from daily predictions
- Confidence intervals and bounds
- Prophet-style trend visualization ready

### 7. **Automated Reporting**
- Validation metrics comparison table
- Daily and monthly forecast exports
- Province and item-level sales breakdowns
- Sales insights metrics table

## Data Pipeline

### 1. Data Loading
- Reads from `data/processed/cleaned_transactions.csv` (main source)
- Falls back to Excel if CSV not available
- Expects columns: Transaction ID, Total Spent, Location, Transaction Date, Province, Item

### 2. Data Preparation
- Aggregate transactions to daily sales totals
- Ensure complete daily date range (no gaps)
- Handle missing dates with NaN values
- Validate date ranges and duplicates

### 3. Train/Validation/Test Split
- **Train**: Days 1-305 (Jan 1 - Oct 31)
- **Validation**: Days 306-335 (Nov 1-30)
- **Test**: Days 336-365 (Dec 1-31)
- 6-month forecast: Dec 2 - May 31, 2024

### 4. Model Training
- SARIMA tuning on train set
- Parameter search: `(p,d,q) × (P,D,Q,s)`
- Seasonal period: 7 days (weekly)
- Select best by WAPE, then MAE

### 5. Evaluation Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **WAPE**: Weighted Average Percentage Error
- **MAPE**: Mean Absolute Percentage Error

### 6. Output Generation
- CSV forecasts (daily and monthly)
- Metrics comparison tables
- Sales insights summaries
- Province/item-level breakdowns

## Output Files

Generated in the `reports/` directory:

| File | Description |
|------|-------------|
| `validation_forecast.csv` | Daily validation predictions with bounds |
| `validation_metrics.csv` | Model performance on validation set |
| `daily_forecast.csv` | 6-month daily forecast |
| `monthly_forecast.csv` | Monthly aggregated forecast |
| `six_month_forecast.csv` | Forecast summary statistics |
| `sales_insights.csv` | Historical and forecast metrics |
| `province_daily_sales.csv` | Daily sales by province |
| `item_daily_sales.csv` | Daily sales by item |

## Model Configuration

### SARIMA Parameters
The model uses seasonal ARIMA (SARIMA) with:
- **Non-seasonal differencing (d)**: Typically 0-1 to remove trends
- **Seasonal differencing (D)**: Typically 0-1 for weekly patterns
- **Autoregressive order (p)**: Past values influence future
- **Moving average order (q)**: Past residuals influence future
- **Seasonal order (P,Q)**: Seasonal components
- **Seasonal period (s)**: 7 days (weekly)

Best model is selected by:
1. Lowest WAPE (primary metric)
2. Lowest MAE (secondary metric)

### Ridge Regression Parameters
The model uses L2 regularized linear regression with:
- **Lag Features**: Sales from 1, 7, and 14 days prior
- **Rolling Features**: 7-day and 14-day moving averages and standard deviations
- **Calendar Features**: Day of week, day of month, month, weekend indicator
- **Categorical Features**: One-hot encoded top 5 items and top 3 provinces
- **Regularization (alpha)**: Default 1.0, higher values increase regularization
- **Feature Scaling**: StandardScaler normalization applied to all features

Ridge Regression complements SARIMA by capturing:
- Weekly item sales patterns (menu-specific seasonality)
- Regional sales differences (province-level variations)
- Multi-week historical dependencies

## Testing

Run unit tests with pytest:

```bash
pytest tests/ -v
```

Tests cover:
- SARIMA model fitting and prediction
- Forecast aggregation (daily → monthly)
- Metrics calculation accuracy
- Edge cases (empty data, NaN handling)

## Troubleshooting

### CSV File Locked
**Error:** `PermissionError: Cannot save to sales_insights.csv`

**Solution:** Close any open Excel/spreadsheet applications and try again.

### Invalid Month Format
**Error:** `Error: Invalid format. Please enter the month as YYYY-MM`

**Solution:** Enter month in YYYY-MM format (e.g., `2023-10` for October 2023).

### Month Outside Data Range
**Error:** `Error: Month 2024-01 is outside the available data range`

**Solution:** Use months within 2023-01 to 2023-12 (available dataset).

## Notes

- All monetary amounts are in **CAD**
- Daily sales are aggregated from individual transactions
- Missing dates are preserved as NaN (not interpolated by default)
- Forecast confidence intervals are 95% by default
- Dates follow ISO format: YYYY-MM-DD

## Contributing

To contribute:

1. Create a feature branch
2. Make your changes
3. Add/update tests as needed
4. Run tests to ensure compatibility
5. Submit a pull request

## License

[Add your license information here]

## Team

**CanAI Hackathon - Group F**

---

**Last Updated:** June 2026  
**Python Version:** 3.10+  
**Key Dependencies:** pandas, numpy, statsmodels, scikit-learn