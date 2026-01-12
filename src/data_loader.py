"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path


def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw data from the data/raw directory."""
    data_path = Path(__file__).parent.parent / "data" / "raw" / filename
    return pd.read_csv(data_path)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to the data/processed directory."""
    data_path = Path(__file__).parent.parent / "data" / "processed" / filename
    df.to_csv(data_path, index=False)
