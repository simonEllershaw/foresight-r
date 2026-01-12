"""
Visualization utilities for data exploration and model results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_distribution(df: pd.DataFrame, column: str, save: bool = False) -> None:
    """Plot the distribution of a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    if save:
        save_path = Path(__file__).parent.parent / "reports" / f"dist_{column}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save: bool = False) -> None:
    """Plot correlation matrix for numeric columns."""
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=["number"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    
    if save:
        save_path = Path(__file__).parent.parent / "reports" / "correlation_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_feature_importance(
    feature_names: list, importances: list, save: bool = False
) -> None:
    """Plot feature importance from a trained model."""
    plt.figure(figsize=(10, 8))
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    
    if save:
        save_path = Path(__file__).parent.parent / "reports" / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
