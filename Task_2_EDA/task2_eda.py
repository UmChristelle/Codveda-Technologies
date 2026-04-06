from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

NUMERIC_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
PALETTE = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}


def run_eda() -> pd.DataFrame:
    """Perform exploratory data analysis on the Iris dataset."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    summary_stats = df[NUMERIC_COLUMNS].describe().round(3).T
    summary_stats["median"] = df[NUMERIC_COLUMNS].median().round(3)
    summary_stats["mode"] = df[NUMERIC_COLUMNS].mode().iloc[0].round(3)
    summary_stats["std"] = df[NUMERIC_COLUMNS].std().round(3)

    print("\nSummary statistics:")
    print(summary_stats)

    species_means = df.groupby("species")[NUMERIC_COLUMNS].mean().round(3)
    print("\nMean values by species:")
    print(species_means)

    summary_stats.to_csv(OUTPUTS_DIR / "01_summary_statistics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Iris Feature Distributions", fontsize=14, fontweight="bold")
    for ax, column in zip(axes.flatten(), NUMERIC_COLUMNS):
        sns.histplot(df[column], bins=20, kde=True, color="#3498db", ax=ax)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_histograms.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Iris Boxplots by Species", fontsize=14, fontweight="bold")
    for ax, column in zip(axes.flatten(), NUMERIC_COLUMNS):
        sns.boxplot(data=df, x="species", y=column, hue="species", palette=PALETTE, legend=False, ax=ax)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xlabel("Species")
        ax.set_ylabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "03_boxplots.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Iris Scatter Plots", fontsize=14, fontweight="bold")
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species", palette=PALETTE, s=70, ax=axes[0])
    axes[0].set_title("Sepal Length vs Sepal Width")
    sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", palette=PALETTE, s=70, ax=axes[1])
    axes[1].set_title("Petal Length vs Petal Width")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "04_scatter_plots.png", bbox_inches="tight")
    plt.close()

    correlation_matrix = df[NUMERIC_COLUMNS].corr().round(3)
    print("\nCorrelation matrix:")
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "05_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    return df


if __name__ == "__main__":
    run_eda()
