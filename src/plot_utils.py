"""
Plotting utilities for credit card fraud detection project.

This module provides functions for creating visualizations of the dataset
and model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional


def plot_class_distribution(
    y: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot and optionally save a bar chart of class counts (non-fraud vs fraud).

    Args:
        y: Array of class labels (0 = normal, 1 = fraud).
        save_path: Path to save the figure (e.g., 'figures/class_distribution.png').
                  If None, figure is not saved.
        show: Whether to display the figure.
    """
    # Count classes
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Normal', 'Fraud']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create bar chart
    bars = ax.bar(class_names, counts, color=['#2ecc71', '#e74c3c'], alpha=0.8)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{count:,}\n({count/sum(counts)*100:.2f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Labels and title
    ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Credit Card Fraud Dataset',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.15)

    # Grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_amount_distribution(
    df,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot histogram of transaction amounts by class.

    Args:
        df: DataFrame with 'Amount' and 'Class' columns.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot for normal transactions
    normal_amounts = df[df['Class'] == 0]['Amount']
    axes[0].hist(normal_amounts, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Transaction Amount ($)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Normal Transactions', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Plot for fraudulent transactions
    fraud_amounts = df[df['Class'] == 1]['Amount']
    axes[1].hist(fraud_amounts, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Transaction Amount ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Fraudulent Transactions', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Transaction Amount Distribution by Class',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_time_distribution(
    df,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot fraud occurrence over time.

    Args:
        df: DataFrame with 'Time' and 'Class' columns.
        save_path: Path to save the figure.
        show: Whether to display the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Convert time to hours
    time_hours = df['Time'] / 3600

    # Plot normal and fraud transactions
    normal_mask = df['Class'] == 0
    fraud_mask = df['Class'] == 1

    ax.scatter(time_hours[normal_mask], df[normal_mask]['Amount'],
               c='#2ecc71', alpha=0.1, s=1, label='Normal')
    ax.scatter(time_hours[fraud_mask], df[fraud_mask]['Amount'],
               c='#e74c3c', alpha=0.6, s=10, label='Fraud')

    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Transaction Amount ($)', fontsize=11)
    ax.set_title('Transactions Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    plt.close()
