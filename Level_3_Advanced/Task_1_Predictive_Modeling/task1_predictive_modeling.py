from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "churn_train.csv"
TEST_PATH = BASE_DIR / "data" / "churn_test.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "Churn"


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Encode categorical columns and align train/test feature spaces."""
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

    x_train_df = train_df.drop(columns=[TARGET_COLUMN]).copy()
    x_test_df = test_df.drop(columns=[TARGET_COLUMN]).copy()

    x_train_df["Area code"] = x_train_df["Area code"].astype(str)
    x_test_df["Area code"] = x_test_df["Area code"].astype(str)

    combined = pd.concat([x_train_df, x_test_df], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=["State", "Area code", "International plan", "Voice mail plan"], drop_first=False)

    x_train_encoded = combined_encoded.iloc[: len(train_df)].copy()
    x_test_encoded = combined_encoded.iloc[len(train_df) :].copy()

    return x_train_encoded, x_test_encoded, y_train, y_test


def standardize(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize feature matrices using training-set statistics."""
    means = train_df.mean()
    stds = train_df.std(ddof=0).replace(0, 1)
    x_train = ((train_df - means) / stds).to_numpy(dtype=float)
    x_val = ((val_df - means) / stds).to_numpy(dtype=float)
    x_test = ((test_df - means) / stds).to_numpy(dtype=float)
    return x_train, x_val, x_test


def train_validation_split(x_df: pd.DataFrame, y: np.ndarray, validation_size: float = 0.2, random_state: int = 42):
    """Split training data into train and validation sets."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(x_df))
    split_index = int(len(x_df) * (1 - validation_size))
    train_idx, val_idx = indices[:split_index], indices[split_index:]
    return x_df.iloc[train_idx], x_df.iloc[val_idx], y[train_idx], y[val_idx]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int, l2_penalty: float) -> np.ndarray:
    """Train logistic regression with gradient descent."""
    weights = np.zeros(x.shape[1] + 1)
    x_bias = np.column_stack([np.ones(len(x)), x])

    for _ in range(epochs):
        logits = x_bias @ weights
        predictions = sigmoid(logits)
        errors = predictions - y
        gradient = (x_bias.T @ errors) / len(x)
        gradient[1:] += l2_penalty * weights[1:] / len(x)
        weights -= learning_rate * gradient

    return weights


def predict_logistic(weights: np.ndarray, x: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and labels for logistic regression."""
    x_bias = np.column_stack([np.ones(len(x)), x])
    probabilities = sigmoid(x_bias @ weights)
    labels = (probabilities >= threshold).astype(int)
    return probabilities, labels


def fit_gaussian_nb(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a Gaussian Naive Bayes model."""
    classes = np.unique(y)
    means = {}
    variances = {}
    priors = {}
    for class_value in classes:
        x_class = x[y == class_value]
        means[class_value] = x_class.mean(axis=0)
        variances[class_value] = x_class.var(axis=0) + 1e-6
        priors[class_value] = len(x_class) / len(x)
    return {"classes": classes, "means": means, "variances": variances, "priors": priors}


def predict_gaussian_nb(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and labels for Gaussian Naive Bayes."""
    log_posteriors = []
    for class_value in model["classes"]:
        mean = model["means"][class_value]
        variance = model["variances"][class_value]
        prior = np.log(model["priors"][class_value])
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + ((x - mean) ** 2) / variance, axis=1)
        log_posteriors.append(prior + log_likelihood)
    log_posteriors = np.column_stack(log_posteriors)
    max_log = log_posteriors.max(axis=1, keepdims=True)
    exp_scores = np.exp(log_posteriors - max_log)
    probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    labels = model["classes"][probabilities.argmax(axis=1)].astype(int)
    positive_index = list(model["classes"]).index(1)
    return probabilities[:, positive_index], labels


def predict_knn(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and labels for K-Nearest Neighbors."""
    distances = np.sqrt(((x_eval[:, None, :] - x_train[None, :, :]) ** 2).sum(axis=2))
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_labels = y_train[nearest_indices]
    probabilities = nearest_labels.mean(axis=1)
    labels = (probabilities >= 0.5).astype(int)
    return probabilities, labels


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy, precision, recall, F1-score, and confusion counts."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_predictive_modeling() -> pd.DataFrame:
    """Train, tune, compare, and evaluate churn classification models."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    x_train_full_df, x_test_full_df, y_train_full, y_test = prepare_features(train_df, test_df)
    x_train_df, x_val_df, y_train, y_val = train_validation_split(x_train_full_df, y_train_full, validation_size=0.2, random_state=42)
    x_train, x_val, x_test = standardize(x_train_df, x_val_df, x_test_full_df)

    model_results = []
    trained_models = {}

    logistic_grid = [
        {"learning_rate": 0.05, "epochs": 1500, "l2_penalty": 0.0},
        {"learning_rate": 0.05, "epochs": 2000, "l2_penalty": 0.01},
        {"learning_rate": 0.1, "epochs": 1500, "l2_penalty": 0.01},
        {"learning_rate": 0.1, "epochs": 2000, "l2_penalty": 0.1},
    ]

    best_logistic = None
    best_logistic_f1 = -1.0
    for params in logistic_grid:
        weights = fit_logistic_regression(x_train, y_train, **params)
        _, val_pred = predict_logistic(weights, x_val)
        metrics = classification_metrics(y_val, val_pred)
        if metrics["f1_score"] > best_logistic_f1:
            best_logistic_f1 = metrics["f1_score"]
            best_logistic = {"weights": weights, "params": params, "validation_metrics": metrics}

    trained_models["Logistic Regression"] = best_logistic
    model_results.append(
        {
            "model": "Logistic Regression",
            "selection_basis": "best validation F1",
            "hyperparameters": str(best_logistic["params"]),
            **best_logistic["validation_metrics"],
        }
    )

    knn_grid = [3, 5, 7, 9, 11]
    best_knn = None
    best_knn_f1 = -1.0
    for k in knn_grid:
        _, val_pred = predict_knn(x_train, y_train, x_val, k=k)
        metrics = classification_metrics(y_val, val_pred)
        if metrics["f1_score"] > best_knn_f1:
            best_knn_f1 = metrics["f1_score"]
            best_knn = {"k": k, "validation_metrics": metrics}

    trained_models["KNN"] = best_knn
    model_results.append(
        {
            "model": "KNN",
            "selection_basis": "best validation F1",
            "hyperparameters": f"{{'k': {best_knn['k']}}}",
            **best_knn["validation_metrics"],
        }
    )

    nb_model = fit_gaussian_nb(x_train, y_train)
    _, nb_val_pred = predict_gaussian_nb(nb_model, x_val)
    nb_metrics = classification_metrics(y_val, nb_val_pred)
    trained_models["Gaussian Naive Bayes"] = {"model": nb_model, "validation_metrics": nb_metrics}
    model_results.append(
        {
            "model": "Gaussian Naive Bayes",
            "selection_basis": "single fitted model",
            "hyperparameters": "{}",
            **nb_metrics,
        }
    )

    validation_df = pd.DataFrame(model_results).sort_values(by=["f1_score", "recall", "precision"], ascending=False).reset_index(drop=True)

    x_train_all, _, x_test_scaled = standardize(x_train_full_df, x_train_full_df.iloc[:0], x_test_full_df)
    y_train_all = y_train_full

    final_results = []
    best_model_name = None
    best_test_f1 = -1.0
    best_test_pred = None
    best_test_prob = None
    best_confusion = None
    best_params = None

    for _, row in validation_df.iterrows():
        model_name = row["model"]

        if model_name == "Logistic Regression":
            params = trained_models[model_name]["params"]
            final_weights = fit_logistic_regression(x_train_all, y_train_all, **params)
            probabilities, predictions = predict_logistic(final_weights, x_test_scaled)
            hyperparameters = str(params)
        elif model_name == "KNN":
            k = trained_models[model_name]["k"]
            probabilities, predictions = predict_knn(x_train_all, y_train_all, x_test_scaled, k=k)
            hyperparameters = f"{{'k': {k}}}"
        else:
            final_nb_model = fit_gaussian_nb(x_train_all, y_train_all)
            probabilities, predictions = predict_gaussian_nb(final_nb_model, x_test_scaled)
            hyperparameters = "{}"

        metrics = classification_metrics(y_test, predictions)
        final_results.append(
            {
                "model": model_name,
                "selection_basis": row["selection_basis"],
                "hyperparameters": hyperparameters,
                **metrics,
            }
        )

        if metrics["f1_score"] > best_test_f1:
            best_test_f1 = metrics["f1_score"]
            best_model_name = model_name
            best_test_pred = predictions
            best_test_prob = probabilities
            best_confusion = metrics
            best_params = hyperparameters

    results_df = pd.DataFrame(final_results).sort_values(by=["f1_score", "recall", "precision"], ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_predictions = test_df.copy()
    best_predictions["predicted_churn"] = best_test_pred.astype(bool)
    best_predictions["predicted_probability"] = np.round(best_test_prob, 4)
    best_predictions.to_csv(OUTPUTS_DIR / "best_model_predictions.csv", index=False)

    summary_text = (
        f"Best model: {best_model_name}\n"
        f"Hyperparameters: {best_params}\n"
        f"Accuracy: {best_confusion['accuracy']:.4f}\n"
        f"Precision: {best_confusion['precision']:.4f}\n"
        f"Recall: {best_confusion['recall']:.4f}\n"
        f"F1-score: {best_confusion['f1_score']:.4f}\n"
        f"TP: {best_confusion['tp']}\n"
        f"TN: {best_confusion['tn']}\n"
        f"FP: {best_confusion['fp']}\n"
        f"FN: {best_confusion['fn']}\n"
    )
    (OUTPUTS_DIR / "best_model_summary.txt").write_text(summary_text, encoding="utf-8")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="f1_score", hue="model", palette="Set2", legend=False)
    plt.title("Model Comparison by F1-score", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "01_model_f1_comparison.png", bbox_inches="tight")
    plt.close()

    confusion_matrix = np.array(
        [
            [best_confusion["tn"], best_confusion["fp"]],
            [best_confusion["fn"], best_confusion["tp"]],
        ]
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Pred False", "Pred True"], yticklabels=["Actual False", "Actual True"])
    plt.title(f"Confusion Matrix: {best_model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "02_best_model_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Validation comparison:")
    print(validation_df[["model", "accuracy", "precision", "recall", "f1_score", "hyperparameters"]].to_string(index=False))
    print("\nTest comparison:")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "hyperparameters"]].to_string(index=False))
    print(f"\nBest model on test set: {best_model_name}")

    return results_df


if __name__ == "__main__":
    run_predictive_modeling()
