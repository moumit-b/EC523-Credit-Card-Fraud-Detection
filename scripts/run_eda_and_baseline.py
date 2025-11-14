"""
Main script for EDA and baseline model evaluation.

This script loads the credit card fraud dataset, performs exploratory
data analysis, trains baseline models, and evaluates their performance.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import (
    load_creditcard_data,
    print_dataset_info,
    time_ordered_split,
    standardize_features
)
from src.baselines import (
    train_logistic_regression,
    train_isolation_forest,
    print_baseline_comparison
)
from src.plot_utils import plot_class_distribution


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("CREDIT CARD FRAUD DETECTION - EDA & BASELINE MODELS")
    print("=" * 60)

    # Load dataset
    try:
        df = load_creditcard_data("data/creditcard.csv")
        print("\nDataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease follow these steps:")
        print("1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Download the creditcard.csv file")
        print("3. Place it in the data/ directory of this project")
        return

    print_dataset_info(df)

    # Create time-ordered splits
    train_df, val_df, test_df = time_ordered_split(df)

    # Standardize features
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_features(
        train_df, val_df, test_df
    )

    # Plot class distribution
    print("\nGenerating visualizations...")
    plot_class_distribution(
        df['Class'].values,
        save_path='figures/class_distribution.png',
        show=False
    )

    # Train baseline models
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODELS")
    print("=" * 60)

    results = {}

    # Logistic Regression
    lr_results = train_logistic_regression(X_train, y_train, X_val, y_val)
    results['Logistic Regression'] = lr_results

    # Isolation Forest
    if_results = train_isolation_forest(X_train, y_train, X_val, y_val)
    results['Isolation Forest'] = if_results

    print_baseline_comparison(results)
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    print("Completed:")
    print("  - Data loading and preprocessing")
    print("  - Time-ordered train/val/test split")
    print("  - Feature standardization")
    print("  - Baseline model training and evaluation")
    print("  - Class distribution visualization")
    print("\nNext steps:")
    print("  - Implement deep autoencoder architecture")
    print("  - Train autoencoder on normal transactions")
    print("  - Develop anomaly scoring mechanism")
    print("  - Comprehensive evaluation and comparison")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
