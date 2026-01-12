"""
Model training and evaluation utilities.
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from pathlib import Path


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_model(model, filename: str) -> None:
    """Save trained model to the models directory."""
    model_path = Path(__file__).parent.parent / "models" / filename
    joblib.dump(model, model_path)


def load_model(filename: str):
    """Load model from the models directory."""
    model_path = Path(__file__).parent.parent / "models" / filename
    return joblib.load(model_path)


def evaluate_classifier(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate classification model performance."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
