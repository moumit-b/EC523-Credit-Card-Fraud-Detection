"""
Comprehensive evaluation framework for fraud detection models.

This module provides functions for:
- Computing all required metrics (ROC-AUC, PR-AUC, Recall@90%Precision)
- Cost-sensitive evaluation (λ·FN + FP)
- Threshold selection based on precision targets
- Confusion matrices and classification reports
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List, Tuple, Optional


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a model.

    Args:
        y_true: True labels (0 = normal, 1 = fraud)
        y_scores: Anomaly scores (higher = more likely fraud)
        model_name: Name of the model for reporting

    Returns:
        Dictionary with all metrics
    """
    # Basic metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Recall@90% Precision
    recall_at_90p, threshold_at_90p = find_recall_at_precision(
        y_true, y_scores, target_precision=0.90
    )

    # Precision-Recall curve data
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'recall_at_90p': recall_at_90p,
        'threshold_at_90p': threshold_at_90p,
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'y_scores': y_scores
    }


def find_recall_at_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_precision: float = 0.90
) -> Tuple[float, float]:
    """
    Find recall at a target precision level.

    This is a key metric for fraud detection: we want to know how many
    frauds we can catch while maintaining high precision (few false alarms).

    Args:
        y_true: True labels
        y_scores: Anomaly scores
        target_precision: Target precision level (default 0.90)

    Returns:
        Tuple of (recall_at_target, threshold_at_target)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Find the recall where precision is >= target
    # Note: precision and recall are in descending order of threshold
    valid_indices = np.where(precision >= target_precision)[0]

    if len(valid_indices) == 0:
        # Target precision not achievable
        return 0.0, np.inf

    # Get the maximum recall at target precision
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    recall_at_target = recall[best_idx]

    # Get corresponding threshold
    # (thresholds array is one shorter than precision/recall)
    if best_idx < len(thresholds):
        threshold_at_target = thresholds[best_idx]
    else:
        threshold_at_target = thresholds[-1] if len(thresholds) > 0 else 0.0

    return recall_at_target, threshold_at_target


def compute_cost_sensitive_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    lambda_values: List[float] = [10, 50, 100]
) -> Dict[float, Dict[str, Any]]:
    """
    Compute cost-sensitive evaluation metrics.

    Cost function: Cost = λ·FN + FP
    Where:
    - FN = False Negatives (missed frauds) - very expensive
    - FP = False Positives (false alarms) - less expensive
    - λ = cost ratio (how much more expensive is missing a fraud)

    Args:
        y_true: True labels
        y_scores: Anomaly scores
        lambda_values: List of cost ratios to evaluate

    Returns:
        Dictionary mapping lambda values to optimal results
    """
    results = {}

    for lambda_val in lambda_values:
        # Find threshold that minimizes cost for this lambda
        best_cost = np.inf
        best_threshold = 0.0
        best_metrics = None

        # Try many thresholds
        thresholds = np.percentile(y_scores, np.linspace(0, 100, 1000))

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Compute cost
            cost = lambda_val * fn + fp

            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'cost': cost,
                    'tp': int(tp),
                    'fp': int(fp),
                    'tn': int(tn),
                    'fn': int(fn),
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                }

        results[lambda_val] = best_metrics

    return results


def get_confusion_matrix_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Get confusion matrix and metrics at a specific threshold.

    Args:
        y_true: True labels
        y_scores: Anomaly scores
        threshold: Classification threshold

    Returns:
        Tuple of (confusion_matrix, metrics_dict)
    """
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    return cm, metrics


def print_evaluation_summary(
    results: Dict[str, Any],
    split_name: str = "Validation"
) -> None:
    """
    Print a formatted summary of evaluation results.

    Args:
        results: Results dictionary from compute_all_metrics()
        split_name: Name of the split (e.g., "Validation", "Test")
    """
    print("\n" + "=" * 60)
    print(f"{results['model_name']} - {split_name} Set Evaluation")
    print("=" * 60)
    print(f"ROC-AUC:              {results['roc_auc']:.4f}")
    print(f"PR-AUC:               {results['pr_auc']:.4f}")
    print(f"Recall@90% Precision: {results['recall_at_90p']:.4f}")
    print(f"  (Threshold:         {results['threshold_at_90p']:.6f})")
    print("=" * 60)


def compare_models(
    results_list: List[Dict[str, Any]],
    split_name: str = "Validation"
) -> None:
    """
    Print comparison table of multiple models.

    Args:
        results_list: List of results dictionaries from compute_all_metrics()
        split_name: Name of the split
    """
    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON - {split_name} Set")
    print("=" * 80)
    print(f"{'Model':<25} {'ROC-AUC':<12} {'PR-AUC':<12} {'Recall@90%P':<12}")
    print("-" * 80)

    for results in results_list:
        print(f"{results['model_name']:<25} "
              f"{results['roc_auc']:<12.4f} "
              f"{results['pr_auc']:<12.4f} "
              f"{results['recall_at_90p']:<12.4f}")

    print("=" * 80)


def print_cost_sensitive_analysis(
    cost_results: Dict[float, Dict[str, Any]],
    model_name: str = "Model"
) -> None:
    """
    Print cost-sensitive evaluation results.

    Args:
        cost_results: Results from compute_cost_sensitive_metrics()
        model_name: Name of the model
    """
    print("\n" + "=" * 60)
    print(f"{model_name} - Cost-Sensitive Analysis")
    print("=" * 60)
    print(f"{'λ':<10} {'Cost':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for lambda_val, metrics in sorted(cost_results.items()):
        print(f"{lambda_val:<10} "
              f"{metrics['cost']:<12.2f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f}")

    print("=" * 60)


def save_metrics_to_dict(
    results: Dict[str, Any],
    split_name: str = "val"
) -> Dict[str, float]:
    """
    Extract key metrics for saving to file.

    Args:
        results: Results dictionary
        split_name: Split name prefix

    Returns:
        Flattened metrics dictionary
    """
    return {
        f'{split_name}_roc_auc': results['roc_auc'],
        f'{split_name}_pr_auc': results['pr_auc'],
        f'{split_name}_recall_at_90p': results['recall_at_90p'],
        f'{split_name}_threshold_at_90p': results['threshold_at_90p']
    }


if __name__ == "__main__":
    # Test the evaluation functions
    print("Testing evaluation framework...")

    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_fraud = 50

    # Simulate labels
    y_true = np.zeros(n_samples)
    y_true[:n_fraud] = 1
    np.random.shuffle(y_true)

    # Simulate scores (fraud should have higher scores on average)
    y_scores = np.random.randn(n_samples)
    y_scores[y_true == 1] += 1.5  # Boost fraud scores

    # Test basic metrics
    print("\n1. Testing basic metrics...")
    results = compute_all_metrics(y_true, y_scores, model_name="Test Model")
    print_evaluation_summary(results, split_name="Test")

    # Test cost-sensitive
    print("\n2. Testing cost-sensitive evaluation...")
    cost_results = compute_cost_sensitive_metrics(y_true, y_scores)
    print_cost_sensitive_analysis(cost_results, model_name="Test Model")

    # Test confusion matrix
    print("\n3. Testing confusion matrix...")
    cm, metrics = get_confusion_matrix_at_threshold(
        y_true, y_scores, threshold=results['threshold_at_90p']
    )
    print(f"Confusion matrix at threshold {results['threshold_at_90p']:.4f}:")
    print(cm)
    print(f"\nMetrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
