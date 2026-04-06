from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "aapl_stock_prices.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def run_time_series_analysis() -> pd.DataFrame:
    """Analyze AAPL stock prices as a time series."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["close_30_ma"] = df["close"].rolling(window=30, min_periods=1).mean()
    df["close_90_ma"] = df["close"].rolling(window=90, min_periods=1).mean()
    df["trend"] = df["close"].rolling(window=30, center=True, min_periods=15).mean()
    df["detrended"] = df["close"] - df["trend"]
    seasonal_by_month = df.groupby(df["date"].dt.month)["detrended"].mean()
    seasonal_by_month = seasonal_by_month.fillna(0)
    seasonal_by_month = seasonal_by_month - seasonal_by_month.mean()
    df["seasonal"] = df["date"].dt.month.map(seasonal_by_month)
    df["residual"] = df["close"] - df["trend"] - df["seasonal"]
    df["daily_return_pct"] = df["close"].pct_change() * 100
    df["month_name"] = df["date"].dt.month_name()

    monthly_avg_close = (
        df.groupby(["date"])
        .agg(close=("close", "mean"))
        .resample("ME")
        .mean()
        .reset_index()
    )
    monthly_seasonality = (
        df.groupby("month_name")["close"]
        .mean()
        .reindex(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        )
    )

    summary_text = (
        f"Symbol: {df['symbol'].iloc[0]}\n"
        f"Rows: {len(df)}\n"
        f"Start date: {df['date'].min().date()}\n"
        f"End date: {df['date'].max().date()}\n"
        f"Average close price: {df['close'].mean():.3f}\n"
        f"Minimum close price: {df['close'].min():.3f}\n"
        f"Maximum close price: {df['close'].max():.3f}\n"
        f"Average daily return (%): {df['daily_return_pct'].mean():.3f}\n"
        f"Daily return std (%): {df['daily_return_pct'].std():.3f}\n"
    )
    (OUTPUTS_DIR / "time_series_summary.txt").write_text(summary_text, encoding="utf-8")

    export_columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_30_ma",
        "close_90_ma",
        "trend",
        "seasonal",
        "residual",
        "daily_return_pct",
    ]
    df[export_columns].to_csv(OUTPUTS_DIR / "aapl_enriched_timeseries.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["close"], color="#1f77b4", linewidth=1.8)
    plt.title("AAPL Closing Price Over Time", fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_closing_price_trend.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["close"], color="#c7c7c7", linewidth=1.2, label="Close")
    plt.plot(df["date"], df["close_30_ma"], color="#2ca02c", linewidth=2.0, label="30-day MA")
    plt.plot(df["date"], df["close_90_ma"], color="#d62728", linewidth=2.0, label="90-day MA")
    plt.title("AAPL Closing Price with Moving Averages", fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_moving_averages.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_seasonality.index, y=monthly_seasonality.values, color="#3498db")
    plt.title("Average Monthly Closing Price", fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Average Close Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "03_monthly_average_close.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(df["date"], df["close"], color="#1f77b4")
    axes[0].set_title("Observed Close Price")
    axes[1].plot(df["date"], df["trend"], color="#2ca02c")
    axes[1].set_title("Trend (30-day centered rolling mean)")
    axes[2].plot(df["date"], df["seasonal"], color="#ff7f0e")
    axes[2].set_title("Seasonal Component (monthly pattern)")
    axes[3].plot(df["date"], df["residual"], color="#7f7f7f")
    axes[3].set_title("Residual Component")
    axes[3].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "04_decomposition.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["daily_return_pct"].dropna(), bins=40, kde=True, color="#8e44ad")
    plt.title("Distribution of Daily Percentage Returns", fontweight="bold")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "05_daily_return_distribution.png", bbox_inches="tight")
    plt.close()

    print("Symbol analyzed:", df["symbol"].iloc[0])
    print("Rows:", len(df))
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Average close price:", round(df["close"].mean(), 3))
    print("Average daily return (%):", round(df["daily_return_pct"].mean(), 3))
    print("Daily return std (%):", round(df["daily_return_pct"].std(), 3))

    return df


if __name__ == "__main__":
    run_time_series_analysis()
