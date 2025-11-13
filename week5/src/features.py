"""
Feature Engineering Helpers
---------------------------
Defines reusable feature creation functions for hospital data.
"""

import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df["length_of_stay_ratio"] = df["length_of_stay"] / (df["previous_admissions"] + 1)
    df["medication_burden"] = df["num_medications"] / (df["age"] + 1)
    return df
