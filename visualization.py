"""
Visualization utilities for model results and report generation.

This module provides functions for creating publication-quality figures:
- Training curves (train/val loss vs epochs)
- PR curves (Precision-Recall curves for model comparison)
- ROC curves
- Reconstruction error distributions
- Comparison tables
- Cost-sensitive analysis plots
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any, Optional
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "Autoencoder",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot training and validation loss curves.

    MANDATORY for final report per professor's requirements.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Name of the model
        save_path: Path to save figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
               label=f'Best Epoch ({best_epoch})')
    ax.plot(best_epoch, best_val_loss, 'g*', markersize=15)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Training Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_pr_curves(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Precision-Recall Curves"
) -> None:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        results_list: List of result dictionaries from evaluation.compute_all_metrics()
        save_path: Path to save figure
        show: Whether to display the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, results in enumerate(results_list):
        color = colors[idx % len(colors)]
        label = f"{results['model_name']} (AUC={results['pr_auc']:.4f})"

        ax.plot(
            results['recall'],
            results['precision'],
            color=color,
            linewidth=2.5,
            label=label,
            alpha=0.8
        )

        # Mark the 90% precision point if available
        if results['recall_at_90p'] > 0:
            ax.plot(
                results['recall_at_90p'],
                0.90,
                marker='o',
                markersize=8,
                color=color,
                markeredgecolor='black',
                markeredgewidth=1
            )

    # Add 90% precision line
    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5,
               label='90% Precision Target')

    # No-skill baseline (fraud rate)
    # This would be the PR-AUC of a random classifier
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3,
               label='Random Classifier')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"PR curves saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_roc_curves(
    results_list: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "ROC Curves"
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        results_list: List of result dictionaries
        save_path: Path to save figure
        show: Whether to display the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, results in enumerate(results_list):
        color = colors[idx % len(colors)]
        label = f"{results['model_name']} (AUC={results['roc_auc']:.4f})"

        ax.plot(
            results['fpr'],
            results['tpr'],
            color=color,
            linewidth=2.5,
            label=label,
            alpha=0.8
        )

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_reconstruction_error_distribution(
    errors_normal: np.ndarray,
    errors_fraud: np.ndarray,
    model_name: str = "Autoencoder",
    save_path: Optional[str] = None,
    show: bool = False,
    use_log_scale: bool = True
) -> None:
    """
    Plot distribution of reconstruction errors for normal vs fraud transactions.

    Args:
        errors_normal: Reconstruction errors for normal transactions
        errors_fraud: Reconstruction errors for fraudulent transactions
        model_name: Name of the model
        save_path: Path to save figure
        show: Whether to display the figure
        use_log_scale: Whether to use log scale for y-axis
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histograms
    bins = 100
    ax.hist(errors_normal, bins=bins, density=True, alpha=0.6,
            color='green', label=f'Normal (n={len(errors_normal):,})', edgecolor='black')
    ax.hist(errors_fraud, bins=bins, density=True, alpha=0.6,
            color='red', label=f'Fraud (n={len(errors_fraud):,})', edgecolor='black')

    # Add vertical lines for means
    ax.axvline(errors_normal.mean(), color='darkgreen', linestyle='--',
               linewidth=2, label=f'Normal Mean: {errors_normal.mean():.4f}')
    ax.axvline(errors_fraud.mean(), color='darkred', linestyle='--',
               linewidth=2, label=f'Fraud Mean: {errors_fraud.mean():.4f}')

    if use_log_scale:
        ax.set_yscale('log')
        ylabel = 'Density (log scale)'
    else:
        ylabel = 'Density'

    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Reconstruction Error Distribution',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_comparison_table(
    results_list: List[Dict[str, Any]],
    split_name: str = "Test",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create a visual comparison table of model results.

    Args:
        results_list: List of result dictionaries
        split_name: Name of the split
        save_path: Path to save figure
        show: Whether to display the figure
    """
    # Extract data
    models = [r['model_name'] for r in results_list]
    roc_aucs = [r['roc_auc'] for r in results_list]
    pr_aucs = [r['pr_auc'] for r in results_list]
    recalls = [r['recall_at_90p'] for r in results_list]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x - width, roc_aucs, width, label='ROC-AUC', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, pr_aucs, width, label='PR-AUC', color='lightcoral', edgecolor='black')
    bars3 = ax.bar(x + width, recalls, width, label='Recall@90%P', color='lightgreen', edgecolor='black')

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison - {split_name} Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison table saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_cost_vs_lambda(
    cost_results_dict: Dict[str, Dict[float, Dict[str, Any]]],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot total cost vs lambda for different models.

    Args:
        cost_results_dict: Dictionary mapping model names to cost results
        save_path: Path to save figure
        show: Whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (model_name, cost_results) in enumerate(cost_results_dict.items()):
        lambdas = sorted(cost_results.keys())
        costs = [cost_results[l]['cost'] for l in lambdas]

        color = colors[idx % len(colors)]
        ax.plot(lambdas, costs, marker='o', linewidth=2, markersize=8,
                label=model_name, color=color, alpha=0.8)

    ax.set_xlabel('λ (Cost Ratio: FN vs FP)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Cost (λ·FN + FP)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Sensitive Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Cost analysis plot saved to: {save_path}")

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization module...")

    # 1. Test training curves
    print("\n1. Testing training curves...")
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.22, 0.21, 0.205, 0.203, 0.202, 0.202]
    val_losses = [0.55, 0.45, 0.35, 0.30, 0.28, 0.27, 0.275, 0.28, 0.285, 0.29]
    plot_training_curves(train_losses, val_losses, model_name="Test AE",
                        save_path="test_training_curves.png")

    # 2. Test PR curves
    print("\n2. Testing PR curves...")
    n = 1000
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, n)

    results_list = []
    for model_name in ["Model A", "Model B"]:
        from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

        y_scores = np.random.randn(n) + y_true * 1.5
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)

        results_list.append({
            'model_name': model_name,
            'roc_auc': roc_auc_score(y_true, y_scores),
            'pr_auc': average_precision_score(y_true, y_scores),
            'recall_at_90p': 0.5,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr
        })

    plot_pr_curves(results_list, save_path="test_pr_curves.png")
    plot_roc_curves(results_list, save_path="test_roc_curves.png")

    print("\nVisualization tests complete!")
