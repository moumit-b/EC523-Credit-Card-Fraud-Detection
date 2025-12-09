"""
Data utilities for credit card fraud detection project.

This module provides functions for loading, preprocessing, and splitting
the credit card fraud dataset in a time-ordered manner.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


def load_creditcard_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset from the given path.

    Args:
        path: Path to the creditcard.csv file.

    Returns:
        DataFrame containing the credit card transaction data.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ directory."
        )


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.

    Args:
        df: DataFrame containing the credit card data with a 'Class' column.
    """
    total_samples = len(df)
    num_fraud = (df['Class'] == 1).sum()
    num_normal = (df['Class'] == 0).sum()
    fraud_rate = (num_fraud / total_samples) * 100

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total samples: {total_samples:,}")
    print(f"Normal transactions: {num_normal:,}")
    print(f"Fraudulent transactions: {num_fraud:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")
    print("=" * 60)


def time_ordered_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets in time order.

    The dataset is first sorted by the 'Time' column to ensure temporal ordering,
    then split according to the provided fractions.

    Args:
        df: DataFrame containing the credit card data with a 'Time' column.
        train_frac: Fraction of data for training (default 0.6).
        val_frac: Fraction of data for validation (default 0.2).
        test_frac: Fraction of data for testing (default 0.2).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    # Sort by time to maintain temporal ordering
    df_sorted = df.sort_values('Time').reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"\nTime-ordered split:")
    print(f"  Train: {len(train_df):,} samples ({train_frac*100:.0f}%)")
    print(f"  Val:   {len(val_df):,} samples ({val_frac*100:.0f}%)")
    print(f"  Test:  {len(test_df):,} samples ({test_frac*100:.0f}%)")

    return train_df, val_df, test_df


def standardize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           StandardScaler]:
    """
    Standardize features using training set statistics only.

    Fits a StandardScaler on the training data and applies it to all splits.
    This prevents data leakage from validation/test sets.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        feature_cols: List of feature column names to standardize.
                     If None, uses V1-V28 and Amount.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler).
    """
    if feature_cols is None:
        # Default: PCA features V1-V28 and Amount
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']

    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['Class'].values

    X_val = val_df[feature_cols].values
    y_val = val_df['Class'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['Class'].values

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply same transformation to val and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeatures standardized: {len(feature_cols)} features")
    print(f"Feature columns: {', '.join(feature_cols[:5])}... (and {len(feature_cols)-5} more)")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler


def get_normal_transactions_only(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Extract only normal (non-fraud) transactions for unsupervised training.

    This is critical for autoencoder training: we train only on legitimate
    transactions so the model learns normal behavior patterns.

    Args:
        X: Feature matrix.
        y: Labels (0 = normal, 1 = fraud).

    Returns:
        Feature matrix containing only normal transactions.
    """
    normal_mask = (y == 0)
    X_normal = X[normal_mask]

    print(f"\nFiltered to normal transactions only:")
    print(f"  Original samples: {len(X):,}")
    print(f"  Normal samples: {len(X_normal):,}")
    print(f"  Removed frauds: {(~normal_mask).sum():,}")

    return X_normal


def get_dataset_statistics(path: str = "data/creditcard.csv") -> dict:
    """
    Get detailed statistics about the dataset for reporting.

    Args:
        path: Path to the dataset file.

    Returns:
        Dictionary with dataset statistics.
    """
    # File size
    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Load data
    df = pd.read_csv(path)

    stats = {
        'file_size_mb': file_size_mb,
        'total_samples': len(df),
        'num_features': len(df.columns) - 1,  # Exclude 'Class' column
        'normal_count': (df['Class'] == 0).sum(),
        'fraud_count': (df['Class'] == 1).sum(),
        'fraud_rate': (df['Class'] == 1).sum() / len(df) * 100,
        'feature_names': [col for col in df.columns if col != 'Class']
    }

    return stats


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Print detailed statistics for each split.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
    """
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        total = len(df)
        normal = (df['Class'] == 0).sum()
        fraud = (df['Class'] == 1).sum()
        fraud_rate = fraud / total * 100

        print(f"\n{name} Set:")
        print(f"  Total: {total:,}")
        print(f"  Normal: {normal:,} ({normal/total*100:.2f}%)")
        print(f"  Fraud: {fraud:,} ({fraud_rate:.4f}%)")

    print("=" * 60)
