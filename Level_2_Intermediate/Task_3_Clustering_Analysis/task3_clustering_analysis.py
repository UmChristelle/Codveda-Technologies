from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "house_prediction_dataset.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

COLUMN_NAMES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

FEATURES = ["RM", "LSTAT", "MEDV", "CRIM", "NOX", "DIS"]


def standardize_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Standardize selected numeric columns with z-score scaling."""
    means = df[feature_columns].mean()
    stds = df[feature_columns].std(ddof=0)
    scaled = (df[feature_columns] - means) / stds
    return scaled.to_numpy(), means, stds


def kmeans(data: np.ndarray, k: int, random_state: int = 42, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a simple K-Means clustering algorithm."""
    rng = np.random.default_rng(random_state)
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()

    for _ in range(max_iter):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        new_centers = np.array(
            [
                data[labels == cluster_id].mean(axis=0) if np.any(labels == cluster_id) else centers[cluster_id]
                for cluster_id in range(k)
            ]
        )

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    inertia = float(((data - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def run_clustering_analysis() -> pd.DataFrame:
    """Perform clustering analysis on the housing dataset."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
    scaled_data, means, stds = standardize_features(df, FEATURES)

    elbow_records = []
    for k in range(2, 7):
        _, _, inertia = kmeans(scaled_data, k=k, random_state=42)
        elbow_records.append({"k": k, "inertia": round(inertia, 3)})

    elbow_df = pd.DataFrame(elbow_records)
    elbow_df.to_csv(OUTPUTS_DIR / "elbow_method.csv", index=False)

    chosen_k = 3
    labels, centers, inertia = kmeans(scaled_data, k=chosen_k, random_state=42)
    df["cluster"] = labels

    cluster_profiles = df.groupby("cluster")[FEATURES].mean().round(3)
    cluster_profiles.to_csv(OUTPUTS_DIR / "cluster_profiles.csv")

    assignment_columns = COLUMN_NAMES + ["cluster"]
    df[assignment_columns].to_csv(OUTPUTS_DIR / "cluster_assignments.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(elbow_df["k"], elbow_df["inertia"], marker="o", linewidth=2.2, color="#1f77b4")
    plt.title("Elbow Method for K-Means", fontweight="bold")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(elbow_df["k"])
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_elbow_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="RM", y="LSTAT", hue="cluster", palette="Set2", s=70, alpha=0.9)
    plt.title("K-Means Clusters in RM vs LSTAT Space", fontweight="bold")
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("Lower Status Population Percentage (LSTAT)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_cluster_scatter_rm_lstat.png", bbox_inches="tight")
    plt.close()

    print("Dataset shape:", df.shape)
    print("\nSelected features:", FEATURES)
    print("\nFeature means used for scaling:")
    print(means.round(3))
    print("\nFeature standard deviations used for scaling:")
    print(stds.round(3))
    print("\nElbow table:")
    print(elbow_df.to_string(index=False))
    print(f"\nChosen k: {chosen_k}")
    print(f"Inertia at k={chosen_k}: {inertia:.3f}")
    print("\nCluster profile means:")
    print(cluster_profiles)

    return df


if __name__ == "__main__":
    run_clustering_analysis()
