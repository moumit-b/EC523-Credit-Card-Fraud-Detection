"""
Baseline models for credit card fraud detection.

This module provides baseline classifiers to compare against the deep
autoencoder approach, including Logistic Regression and Isolation Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a class-weighted logistic regression as a baseline model.

    Uses balanced class weights to handle the highly imbalanced dataset.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        max_iter: Maximum number of iterations for solver.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'model': Fitted LogisticRegression model
            - 'roc_auc': ROC-AUC score on validation set
            - 'pr_auc': PR-AUC (average precision) score on validation set
            - 'y_pred_proba': Predicted probabilities for positive class
    """
    print("\nTraining Logistic Regression (class-weighted)...")

    # Train model with balanced class weights
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)

    # Get predicted probabilities for the positive class (fraud)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Compute metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)

    print(f"  Training completed.")
    print(f"  Validation ROC-AUC: {roc_auc:.4f}")
    print(f"  Validation PR-AUC:  {pr_auc:.4f}")

    return {
        'model': model,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred_proba': y_pred_proba
    }


def train_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    contamination: float = 'auto',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train an Isolation Forest as an unsupervised baseline model.

    Isolation Forest is an unsupervised anomaly detection algorithm that
    works well for fraud detection.

    Args:
        X_train: Training features.
        y_train: Training labels (not used for training, only for reference).
        X_val: Validation features.
        y_val: Validation labels.
        contamination: Expected proportion of outliers. 'auto' or float.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'model': Fitted IsolationForest model
            - 'roc_auc': ROC-AUC score on validation set
            - 'pr_auc': PR-AUC (average precision) score on validation set
            - 'anomaly_scores': Anomaly scores (negated for compatibility)
    """
    print("\nTraining Isolation Forest (unsupervised)...")

    # Train model (unsupervised, doesn't use y_train)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    model.fit(X_train)

    # Get anomaly scores (more negative = more anomalous)
    # We negate them so higher scores indicate more likely fraud
    anomaly_scores_val = -model.score_samples(X_val)

    # Compute metrics
    roc_auc = roc_auc_score(y_val, anomaly_scores_val)
    pr_auc = average_precision_score(y_val, anomaly_scores_val)

    print(f"  Training completed.")
    print(f"  Validation ROC-AUC: {roc_auc:.4f}")
    print(f"  Validation PR-AUC:  {pr_auc:.4f}")

    return {
        'model': model,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'anomaly_scores': anomaly_scores_val
    }


def evaluate_baseline_on_test(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'logistic'
) -> Dict[str, Any]:
    """
    Evaluate a trained baseline model on test set.

    Args:
        model: Trained model (LogisticRegression or IsolationForest)
        X_test: Test features
        y_test: Test labels
        model_type: 'logistic' or 'isolation_forest'

    Returns:
        Dictionary with test metrics
    """
    if model_type == 'logistic':
        y_scores = model.predict_proba(X_test)[:, 1]
    elif model_type == 'isolation_forest':
        y_scores = -model.score_samples(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    from sklearn.metrics import roc_auc_score, average_precision_score

    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_scores': y_scores
    }


def print_baseline_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a comparison table of baseline model results.

    Args:
        results: Dictionary mapping model names to their result dictionaries.
    """
    print("\n" + "=" * 60)
    print("BASELINE MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'PR-AUC':<12}")
    print("-" * 60)

    for model_name, result in results.items():
        roc_auc = result['roc_auc']
        pr_auc = result['pr_auc']
        print(f"{model_name:<25} {roc_auc:<12.4f} {pr_auc:<12.4f}")

    print("=" * 60)
