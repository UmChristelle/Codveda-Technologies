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

PALETTE = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}
NUMERIC_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def create_visualizations() -> pd.DataFrame:
    """Create and export basic data visualizations for the Iris dataset."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    df = pd.read_csv(DATA_PATH)
    species_means = df.groupby("species")[NUMERIC_COLUMNS].mean().round(3)

    print("Dataset shape:", df.shape)
    print("\nSpecies-level averages:")
    print(species_means)

    avg_petal_length = species_means["petal_length"].reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_petal_length, x="species", y="petal_length", hue="species", palette=PALETTE, legend=False)
    plt.title("Average Petal Length by Species", fontweight="bold")
    plt.xlabel("Species")
    plt.ylabel("Average Petal Length (cm)")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_avg_petal_length_bar.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    for species in species_means.index:
        plt.plot(
            NUMERIC_COLUMNS,
            species_means.loc[species, NUMERIC_COLUMNS],
            marker="o",
            linewidth=2.5,
            label=species.title(),
            color=PALETTE[species],
        )
    plt.title("Average Feature Values by Species", fontweight="bold")
    plt.xlabel("Feature")
    plt.ylabel("Average Value (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_species_feature_line_chart.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="petal_length",
        y="petal_width",
        hue="species",
        palette=PALETTE,
        s=80,
        alpha=0.85,
    )
    plt.title("Petal Length vs Petal Width", fontweight="bold")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "03_petal_scatter.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        palette=PALETTE,
        s=80,
        alpha=0.85,
    )
    plt.title("Sepal Length vs Sepal Width", fontweight="bold")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "04_sepal_scatter.png", bbox_inches="tight")
    plt.close()

    print("\nSaved visualizations:")
    for output_file in sorted(OUTPUTS_DIR.iterdir()):
        if output_file.is_file():
            print(output_file.name)

    return df


if __name__ == "__main__":
    create_visualizations()
