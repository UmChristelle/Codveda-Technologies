from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def train_test_split(feature: np.ndarray, target: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split a feature array and target array into train and test sets."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(feature))
    split_index = int(len(feature) * (1 - test_size))
    train_idx, test_idx = indices[:split_index], indices[split_index:]
    return feature[train_idx], feature[test_idx], target[train_idx], target[test_idx]


def fit_simple_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> tuple[float, float]:
    """Fit a simple linear regression model using least squares."""
    design_matrix = np.column_stack([np.ones(len(x_train)), x_train])
    intercept, slope = np.linalg.lstsq(design_matrix, y_train, rcond=None)[0]
    return float(intercept), float(slope)


def predict(intercept: float, slope: float, x_values: np.ndarray) -> np.ndarray:
    """Generate predictions from the fitted regression model."""
    return intercept + slope * x_values


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1 - ss_res / ss_tot)


def run_regression_analysis() -> pd.DataFrame:
    """Run simple regression analysis on the housing dataset."""
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)

    feature_name = "RM"
    target_name = "MEDV"

    x = df[feature_name].to_numpy()
    y = df[target_name].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    intercept, slope = fit_simple_linear_regression(x_train, y_train)
    y_pred = predict(intercept, slope, x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r_squared(y_test, y_pred)

    print("Dataset shape:", df.shape)
    print(f"\nFeature used: {feature_name}")
    print(f"Target variable: {target_name}")
    print(f"Training rows: {len(x_train)}")
    print(f"Testing rows: {len(x_test)}")
    print(f"\nIntercept: {intercept:.4f}")
    print(f"Slope: {slope:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    coefficients_df = pd.DataFrame(
        {
            "term": ["intercept", feature_name],
            "value": [round(intercept, 4), round(slope, 4)],
        }
    )
    coefficients_df.to_csv(OUTPUTS_DIR / "model_coefficients.csv", index=False)

    metrics_text = (
        f"Feature: {feature_name}\n"
        f"Target: {target_name}\n"
        f"Training rows: {len(x_train)}\n"
        f"Testing rows: {len(x_test)}\n"
        f"Intercept: {intercept:.4f}\n"
        f"Slope: {slope:.4f}\n"
        f"Mean Squared Error: {mse:.4f}\n"
        f"R-squared: {r2:.4f}\n"
    )
    (OUTPUTS_DIR / "model_metrics.txt").write_text(metrics_text, encoding="utf-8")

    predictions_df = pd.DataFrame(
        {
            "RM_test": x_test,
            "actual_MEDV": y_test,
            "predicted_MEDV": np.round(y_pred, 4),
        }
    ).sort_values(by="RM_test")
    predictions_df.to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)

    sorted_indices = np.argsort(x_test)
    x_sorted = x_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure(figsize=(9, 6))
    plt.scatter(x_train, y_train, alpha=0.7, color="#3498db", label="Training data")
    plt.scatter(x_test, y_test, alpha=0.8, color="#2ecc71", label="Testing data")
    plt.plot(x_sorted, y_pred_sorted, color="#e74c3c", linewidth=2.5, label="Regression line")
    plt.title("Simple Linear Regression: RM vs MEDV", fontweight="bold")
    plt.xlabel("Average Number of Rooms (RM)")
    plt.ylabel("Median Home Value (MEDV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "regression_line.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="#8e44ad", alpha=0.8)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1.5)
    plt.title("Actual vs Predicted Home Prices", fontweight="bold")
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", bbox_inches="tight")
    plt.close()

    return df


if __name__ == "__main__":
    run_regression_analysis()
