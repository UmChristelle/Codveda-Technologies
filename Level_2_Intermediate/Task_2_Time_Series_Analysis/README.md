# Level 2 Task 2: Time Series Analysis

## Objective

Analyze a time-series dataset to identify trends, seasonality, and short-term movement patterns.

## Dataset

- Source dataset: stock prices dataset
- Focused series: `AAPL`
- Local file used: `data/aapl_stock_prices.csv`
- Date range: `2014-01-02` to `2017-12-29`

## Workflow

- Loaded and sorted the daily stock price data
- Converted the date column to datetime format
- Checked for missing values
- Plotted the closing price over time
- Calculated 30-day and 90-day moving averages
- Built an additive-style decomposition using:
  - rolling trend
  - month-based seasonal pattern
  - residual component
- Calculated daily percentage returns
- Exported charts and analysis outputs

## Key Findings

- The AAPL closing price shows a clear long-term upward trend across the period.
- The 30-day moving average reacts faster to short-term price changes than the 90-day moving average.
- Average monthly prices are highest toward the end of the year, especially in November and December.
- The decomposition residuals capture short-term price volatility not explained by the rolling trend or seasonal pattern.

## Environment Note

- `statsmodels` was not available in this environment, so the decomposition was implemented manually in a transparent additive-style way using pandas operations.
- This still satisfies the internship objective by separating the series into trend, seasonality, and residual behavior.

## Files

- `task2_time_series_analysis.py`
- `task2_time_series_analysis.ipynb`
- `outputs/time_series_summary.txt`
- `outputs/aapl_enriched_timeseries.csv`
- `outputs/01_closing_price_trend.png`
- `outputs/02_moving_averages.png`
- `outputs/03_monthly_average_close.png`
- `outputs/04_decomposition.png`
- `outputs/05_daily_return_distribution.png`
