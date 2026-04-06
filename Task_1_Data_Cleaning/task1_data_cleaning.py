
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def clean_iris_data() -> pd.DataFrame:
    """Load, inspect, clean, and save the Iris dataset."""
    df = pd.read_csv(DATA_PATH)

    print("Original data shape:", df.shape)
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print("\nDuplicate rows:", duplicate_count)

    cleaned_df = df.drop_duplicates().copy()

    numeric_columns = cleaned_df.select_dtypes(include="number").columns
    cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(
        cleaned_df[numeric_columns].mean()
    )

    categorical_columns = cleaned_df.select_dtypes(exclude="number").columns
    for column in categorical_columns:
        cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.lower()
        mode = cleaned_df[column].mode()
        if not mode.empty:
            cleaned_df[column] = cleaned_df[column].fillna(mode.iloc[0])

    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nMissing values after cleaning:")
    print(cleaned_df.isnull().sum())
    print("\nUnique values in categorical columns after standardization:")
    for column in categorical_columns:
        print(f"{column}: {sorted(cleaned_df[column].unique().tolist())}")

    cleaned_file = OUTPUTS_DIR / "iris_cleaned.csv"
    cleaned_df.to_csv(cleaned_file, index=False)
    print(f"\nCleaned dataset saved to: {cleaned_file}")

    return cleaned_df


if __name__ == "__main__":
    clean_iris_data()
