"""
Training script for Denoising Autoencoder (DAE).

The DAE adds Gaussian noise to inputs during training, forcing the model
to learn more robust representations. This can improve anomaly detection.
"""

import sys
import os
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import *
from src.data_utils import (
    load_creditcard_data,
    time_ordered_split,
    standardize_features,
    get_normal_transactions_only,
    print_split_statistics
)
from src.models import DenoisingAutoencoder
from src.training import train_autoencoder, evaluate_reconstruction_error
from src.evaluation import compute_all_metrics, print_evaluation_summary
from src.visualization import plot_training_curves, plot_reconstruction_error_distribution


def main(args):
    """Main training function for Denoising Autoencoder."""

    # Set all random seeds
    set_all_seeds(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("DENOISING AUTOENCODER TRAINING")
    print("=" * 80)
    print(f"Course: {COURSE}")
    print(f"Project: {PROJECT_TITLE}")
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Noise Level: {args.noise_level}")
    print("=" * 80)

    # Load data
    print("\n1. Loading dataset...")
    df = load_creditcard_data(DATA_PATH)

    # Create time-ordered splits
    print("\n2. Creating time-ordered train/val/test splits...")
    train_df, val_df, test_df = time_ordered_split(
        df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    )
    print_split_statistics(train_df, val_df, test_df)

    # Standardize features
    print("\n3. Standardizing features...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_features(
        train_df, val_df, test_df, FEATURE_COLUMNS
    )

    # Extract ONLY normal transactions for training
    print("\n4. Extracting normal transactions for unsupervised training...")
    X_train_normal = get_normal_transactions_only(X_train, y_train)

    print(f"\n   WARNING: Training ONLY on {len(X_train_normal):,} normal transactions")
    print(f"   Noise level: {args.noise_level} (Gaussian std)")

    # Create Denoising Autoencoder
    print("\n5. Creating Denoising Autoencoder...")
    model = DenoisingAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=args.latent_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=args.dropout,
        noise_level=args.noise_level
    )
    print(model)

    # Train model
    print("\n6. Training denoising autoencoder...")
    model_save_path = get_model_path(
        'dae',
        latent=args.latent_dim,
        noise=int(args.noise_level*100),  # e.g., 0.1 -> 10
        seed=RANDOM_SEED
    )

    training_results = train_autoencoder(
        model=model,
        X_train=X_train_normal,  # ONLY normal transactions!
        X_val=X_val,  # Full validation set
        device=DEVICE,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        save_path=model_save_path,
        verbose=True
    )

    # Save training curves
    print("\n7. Generating training curve visualization...")
    curve_path = os.path.join(
        FIGURES_DIR,
        f'dae_training_curves_latent{args.latent_dim}_noise{int(args.noise_level*100)}.png'
    )
    plot_training_curves(
        train_losses=training_results['train_losses'],
        val_losses=training_results['val_losses'],
        model_name=f"Denoising AE (latent={args.latent_dim}, noise={args.noise_level})",
        save_path=curve_path,
        show=False
    )

    # Evaluate on validation set
    print("\n8. Evaluating on validation set...")
    trained_model = training_results['model']

    val_errors = evaluate_reconstruction_error(
        model=trained_model,
        X=X_val,
        device=DEVICE,
        batch_size=args.batch_size
    )

    # Compute metrics
    val_results = compute_all_metrics(
        y_true=y_val,
        y_scores=val_errors,
        model_name="Denoising Autoencoder"
    )
    print_evaluation_summary(val_results, split_name="Validation")

    # Plot reconstruction error distribution
    print("\n9. Plotting reconstruction error distribution...")
    val_normal_mask = (y_val == 0)
    val_fraud_mask = (y_val == 1)

    dist_path = os.path.join(
        FIGURES_DIR,
        f'dae_error_dist_latent{args.latent_dim}_noise{int(args.noise_level*100)}.png'
    )
    plot_reconstruction_error_distribution(
        errors_normal=val_errors[val_normal_mask],
        errors_fraud=val_errors[val_fraud_mask],
        model_name=f"Denoising AE (latent={args.latent_dim}, noise={args.noise_level})",
        save_path=dist_path,
        show=False,
        use_log_scale=True
    )

    # Save metadata
    print("\n10. Saving training metadata...")
    metadata = {
        'model_type': 'DenoisingAutoencoder',
        'latent_dim': args.latent_dim,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': args.dropout,
        'noise_level': args.noise_level,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'max_epochs': args.epochs,
        'actual_epochs': training_results['total_epochs'],
        'best_epoch': training_results['best_epoch'],
        'training_time_seconds': training_results['training_time'],
        'param_count': training_results['param_count'],
        'device': str(DEVICE),
        'random_seed': RANDOM_SEED,
        'train_samples': len(X_train_normal),
        'val_samples': len(X_val),
        'val_roc_auc': val_results['roc_auc'],
        'val_pr_auc': val_results['pr_auc'],
        'val_recall_at_90p': val_results['recall_at_90p'],
        'model_path': model_save_path
    }

    metadata_path = os.path.join(
        TABLES_DIR,
        f'dae_metadata_latent{args.latent_dim}_noise{int(args.noise_level*100)}.json'
    )
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to:          {model_save_path}")
    print(f"Training curves saved to: {curve_path}")
    print(f"Error distribution saved to: {dist_path}")
    print(f"Metadata saved to:       {metadata_path}")
    print("\nValidation Metrics:")
    print(f"  ROC-AUC:               {val_results['roc_auc']:.4f}")
    print(f"  PR-AUC:                {val_results['pr_auc']:.4f}")
    print(f"  Recall@90% Precision:  {val_results['recall_at_90p']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Denoising Autoencoder for fraud detection')

    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM,
                       help=f'Latent dimension (default: {LATENT_DIM})')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (default: 0.2)')
    parser.add_argument('--noise_level', type=float, default=NOISE_LEVEL,
                       help=f'Noise level (Gaussian std) (default: {NOISE_LEVEL})')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                       help=f'Maximum epochs (default: {MAX_EPOCHS})')
    parser.add_argument('--patience', type=int, default=EARLY_STOPPING_PATIENCE,
                       help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})')

    args = parser.parse_args()

    main(args)
