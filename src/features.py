"""
Feature engineering utilities.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def scale_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize numeric features."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encode categorical features."""
    df_encoded = df.copy()
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
    return df_encoded
