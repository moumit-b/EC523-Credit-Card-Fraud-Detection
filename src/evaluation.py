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
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from typing import Dict, Any, List, Tuple, Optional


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a model.

    Primary metrics:
    - PR-AUC: Area under precision-recall curve (threshold-free)
    - F1-optimal: F1, precision, recall, accuracy at best F1 threshold
    - Confusion matrix: TN, FP, FN, TP at F1-optimal threshold

    Secondary metrics (for reference):
    - ROC-AUC: Area under ROC curve
    - Recall@90%P: Recall at 90% precision (optional future work)

    Args:
        y_true: True labels (0 = normal, 1 = fraud)
        y_scores: Anomaly scores (higher = more likely fraud)
        model_name: Name of the model for reporting

    Returns:
        Dictionary with all metrics
    """
    # Primary metric 1: PR-AUC (threshold-free)
    pr_auc = average_precision_score(y_true, y_scores)

    # Primary metric 2: F1-optimal operating point
    best_f1_threshold, f1_metrics = find_best_f1_threshold(y_true, y_scores)

    # Secondary metrics
    roc_auc = roc_auc_score(y_true, y_scores)

    # Recall@90% Precision (de-emphasized, kept for reference)
    recall_at_90p, threshold_at_90p = find_recall_at_precision(
        y_true, y_scores, target_precision=0.90
    )

    # Precision-Recall curve data (for plotting)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

    # ROC curve data (for plotting)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    return {
        'model_name': model_name,
        # Primary metrics
        'pr_auc': pr_auc,
        'f1_optimal_threshold': best_f1_threshold,
        'f1_optimal_f1': f1_metrics['f1'],
        'f1_optimal_precision': f1_metrics['precision'],
        'f1_optimal_recall': f1_metrics['recall'],
        'f1_optimal_accuracy': f1_metrics['accuracy'],
        'f1_optimal_tp': f1_metrics['tp'],
        'f1_optimal_fp': f1_metrics['fp'],
        'f1_optimal_tn': f1_metrics['tn'],
        'f1_optimal_fn': f1_metrics['fn'],
        # Secondary metrics
        'roc_auc': roc_auc,
        'recall_at_90p': recall_at_90p,
        'threshold_at_90p': threshold_at_90p,
        # Curve data for plotting
        'precision': precision,
        'recall': recall,
        'pr_thresholds': pr_thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'y_scores': y_scores
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, Dict[str, Any]]:
    """
    Find the threshold that maximizes F1 score.

    This sweeps through thresholds from the precision-recall curve to find
    the operating point with the best F1 score.

    Args:
        y_true: True labels (0 = normal, 1 = fraud)
        y_scores: Anomaly scores (higher = more likely fraud)

    Returns:
        Tuple of (best_threshold, metrics_dict)
        where metrics_dict contains: f1, precision, recall, accuracy,
        tp, fp, tn, fn at the best threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1 scores for each threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]

    # Get corresponding threshold
    # Note: thresholds array is one element shorter than precision/recall
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        # If best is at the end, use last threshold or default
        best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.5

    # Compute confusion matrix at best threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if cm.size == 4:  # Standard 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where all predictions are one class
        if cm.shape == (1, 1):
            if y_true[0] == 0:  # All negatives
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:  # All positives
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

    # Compute final metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        'threshold': float(best_threshold),
        'f1': float(best_f1),
        'precision': float(prec),
        'recall': float(rec),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    return best_threshold, metrics


def find_recall_at_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_precision: float = 0.90
) -> Tuple[float, float]:
    """
    Find recall at a target precision level.

    This is a key metric for fraud detection: we want to know how many
    frauds we can catch while maintaining high precision (few false alarms).

    NOTE: This metric is kept for reference but is de-emphasized in favor
    of F1-optimal operating points.

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
    print("\n" + "=" * 70)
    print(f"{results['model_name']} - {split_name} Set Evaluation")
    print("=" * 70)
    print("PRIMARY METRICS:")
    print(f"  PR-AUC (threshold-free):    {results['pr_auc']:.4f}")
    print(f"\nF1-OPTIMAL OPERATING POINT:")
    print(f"  Threshold:                  {results['f1_optimal_threshold']:.6f}")
    print(f"  F1 Score:                   {results['f1_optimal_f1']:.4f}")
    print(f"  Precision:                  {results['f1_optimal_precision']:.4f}")
    print(f"  Recall:                     {results['f1_optimal_recall']:.4f}")
    print(f"  Accuracy:                   {results['f1_optimal_accuracy']:.4f}")
    print(f"\nCONFUSION MATRIX (at F1-optimal threshold):")
    print(f"  True Negatives:  {results['f1_optimal_tn']:6d}")
    print(f"  False Positives: {results['f1_optimal_fp']:6d}")
    print(f"  False Negatives: {results['f1_optimal_fn']:6d}")
    print(f"  True Positives:  {results['f1_optimal_tp']:6d}")
    print(f"\nSECONDARY METRICS (for reference):")
    print(f"  ROC-AUC:                    {results['roc_auc']:.4f}")
    print(f"  Recall@90% Precision:       {results['recall_at_90p']:.4f}")
    print("=" * 70)


def compare_models(
    results_list: List[Dict[str, Any]],
    split_name: str = "Validation"
) -> None:
    """
    Print comparison table of multiple models.

    Focuses on primary metrics: PR-AUC and F1-optimal operating point.

    Args:
        results_list: List of results dictionaries from compute_all_metrics()
        split_name: Name of the split
    """
    print("\n" + "=" * 100)
    print(f"MODEL COMPARISON - {split_name} Set")
    print("=" * 100)
    print(f"{'Model':<25} {'PR-AUC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12} {'Accuracy':<12}")
    print("-" * 100)

    for results in results_list:
        print(f"{results['model_name']:<25} "
              f"{results['pr_auc']:<12.4f} "
              f"{results['f1_optimal_f1']:<12.4f} "
              f"{results['f1_optimal_precision']:<12.4f} "
              f"{results['f1_optimal_recall']:<12.4f} "
              f"{results['f1_optimal_accuracy']:<12.4f}")

    print("=" * 100)


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
