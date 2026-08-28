"""Load the Breast Cancer Wisconsin (Diagnostic) dataset."""

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    """
    Returns
    -------
    X : pd.DataFrame, shape (569, 30)  -- feature matrix
    y : pd.Series,    shape (569,)     -- target, 0 = malignant, 1 = benign
    feature_names : list[str]
    target_names  : list[str]
    """
    dataset = load_breast_cancer()

    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="diagnosis")

    return X, y, list(dataset.feature_names), list(dataset.target_names)


def describe_data(X, y, target_names):
    """Print a quick summary of the dataset for the report."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples        : {X.shape[0]}")
    print(f"Features       : {X.shape[1]}")
    print(f"Missing values : {X.isnull().sum().sum()}")
    print("\nClass distribution:")
    for label, name in enumerate(target_names):
        count = (y == label).sum()
        print(f"  {name:<12} (label {label}): {count:>3}  ({count / len(y):.1%})")
    print("\nFirst 5 features, first 5 rows:")
    print(X.iloc[:5, :5].to_string())
    print()


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    describe_data(X, y, target_names)