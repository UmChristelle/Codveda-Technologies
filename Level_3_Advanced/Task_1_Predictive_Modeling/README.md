# Level 3 Task 1: Predictive Modeling (Classification)

## Objective

Build and evaluate classification models to predict customer churn using the provided churn datasets.

## Dataset

- Training data: `data/churn_train.csv`
- Test data: `data/churn_test.csv`
- Target variable: `Churn`

## Workflow

- Loaded the provided training and testing datasets
- Preprocessed categorical variables with one-hot encoding
- Standardized numerical features using training-set statistics
- Tuned multiple classifiers using a validation split
- Trained and compared three classification models:
  - Logistic Regression
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
- Evaluated models using accuracy, precision, recall, and F1-score
- Selected the best model based on validation and test F1-score

## Modeling Note

- `scikit-learn` was not available in this environment, so the preprocessing, model training, validation search, and evaluation were implemented directly with NumPy and pandas.
- This keeps the work reproducible and still satisfies the internship objective of training, testing, tuning, and evaluating multiple classification models.

## Best Model

- Best model on the test set: `Logistic Regression`
- Accuracy: `0.8591`
- Precision: `0.5111`
- Recall: `0.2421`
- F1-score: `0.3286`

## Interpretation

- Logistic Regression delivered the strongest overall balance among the tested models on this dataset.
- Gaussian Naive Bayes produced higher recall than Logistic Regression, but at the cost of much lower precision and accuracy.
- The churn classes are imbalanced, so F1-score was used as the main selection metric for a more balanced comparison.

## Files

- `task1_predictive_modeling.py`
- `task1_predictive_modeling.ipynb`
- `outputs/model_comparison.csv`
- `outputs/best_model_predictions.csv`
- `outputs/best_model_summary.txt`
- `outputs/01_model_f1_comparison.png`
- `outputs/02_best_model_confusion_matrix.png`
