# Level 2 Task 1: Regression Analysis

## Objective

Perform a simple linear regression analysis to predict house prices using one numerical feature from the housing dataset.

## Dataset

- File: `data/house_prediction_dataset.csv`
- Source format: whitespace-separated values
- Target variable: `MEDV` (median value of owner-occupied homes)
- Predictor used: `RM` (average number of rooms per dwelling)

## Workflow

- Loaded and labeled the dataset columns
- Split the dataset into training and testing sets
- Built a simple linear regression model
- Evaluated the model using Mean Squared Error (MSE) and R-squared
- Exported model metrics, coefficients, predictions, and charts

## Results

- Intercept: `-33.1646`
- Slope for `RM`: `8.8715`
- Test MSE: `40.7512`
- Test R-squared: `0.4927`

## Interpretation

- The positive coefficient for `RM` means that homes with more rooms tend to have higher prices.
- The model explains about 49% of the variation in house prices using `RM` alone.
- This is a useful baseline model, but a multivariable model would likely perform better.

## Environment Note

- The internship brief mentions `scikit-learn`, but it could not be installed successfully in this environment because the package installation timed out.
- To keep the task moving professionally, the regression model was implemented directly with NumPy least squares while preserving the required train-test split, coefficient interpretation, and evaluation workflow.

## Files

- `task1_regression_analysis.py`
- `task1_regression_analysis.ipynb`
- `outputs/model_coefficients.csv`
- `outputs/model_metrics.txt`
- `outputs/test_predictions.csv`
- `outputs/regression_line.png`
- `outputs/actual_vs_predicted.png`
