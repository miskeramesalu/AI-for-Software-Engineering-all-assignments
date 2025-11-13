"""
Data Processing Script for Hospital Readmission Project
--------------------------------------------------------
Handles data loading, cleaning, feature engineering, and splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """Load and preview raw hospital data."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess data."""
    # Example transformations
    df = df.dropna(subset=["readmission_30d"])
    df = pd.get_dummies(df, drop_first=True)
    return df

def split_data(df: pd.DataFrame):
    """Split dataset into train, validation, and test sets."""
    X = df.drop("readmission_30d", axis=1)
    y = df["readmission_30d"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = load_data("../data/raw/hospital_data.csv")
    df_clean = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
