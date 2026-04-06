# Level 1 Task 1: Data Cleaning and Preprocessing

## Objective

Load a raw dataset with pandas, inspect it for missing values and duplicates, clean the data, and save a cleaned version for later analysis.

## Dataset

- Iris dataset
- Source file: `data/iris.csv`

## Work Completed

- Loaded the dataset with pandas
- Checked the shape, data types, missing values, and duplicate rows
- Removed duplicate records
- Applied missing-value handling logic for numeric and categorical columns
- Standardized text values in categorical columns
- Exported the cleaned dataset to `outputs/iris_cleaned.csv`

## Result Summary

- Original shape: 150 rows, 5 columns
- Missing values found: 0
- Duplicate rows found: 3
- Cleaned shape: 147 rows, 5 columns

## Files

- `task1_data_cleaning.ipynb`
- `task1_data_cleaning.py`
- `outputs/iris_cleaned.csv`
