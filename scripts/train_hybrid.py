import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

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

# Import feature columns from config
from config.config import FEATURE_COLUMNS, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
from src.models import DeepAutoencoder
from src.training import train_autoencoder
from src.evaluation import (
    compute_all_metrics,
    print_evaluation_summary
)
from src.visualization import plot_comparison_table


def extract_latent_features(model, X, device, batch_size=256):
    """
    Extract latent representations from trained autoencoder.

    Args:
        model: Trained autoencoder model
        X: Input features (n_samples, input_dim)
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        latent_features: numpy array of shape (n_samples, latent_dim)
    """
    model.eval()
    model = model.to(device)

    latent_features = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            # Forward through encoder only
            latent = model.encoder(batch_tensor)
            latent_features.append(latent.cpu().numpy())

    return np.vstack(latent_features)


def main():
    """Main execution function."""

    print("=" * 80)
    print("HYBRID APPROACH: AUTOENCODER FEATURE LEARNING + SUPERVISED CLASSIFICATION")
    print("=" * 80)

    # Print device info
    print(f"\n🖥️  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif DEVICE.type == 'cpu':
        print(f"   ⚠️  Running on CPU (slower). Consider using GPU for faster training.")
    print()

    # ==================================================================
    # STEP 1: Load and prepare data (same as main pipeline)
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Load data
    df = load_creditcard_data(DATA_PATH)

    # Time-ordered splits
    train_df, val_df, test_df = time_ordered_split(
        df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
    )
    print_split_statistics(train_df, val_df, test_df)

    # Standardize features
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = standardize_features(
        train_df, val_df, test_df, FEATURE_COLUMNS
    )

    # Extract normal transactions for AE training
    X_train_normal = get_normal_transactions_only(X_train, y_train)
    print(f"\nNormal transactions for AE training: {len(X_train_normal):,}")
    print(f"Frauds excluded from AE training: {np.sum(y_train == 1):,}")

    # ==================================================================
    # STEP 2: Train or load Deep Autoencoder
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING DEEP AUTOENCODER (FOR FEATURE LEARNING)")
    print("=" * 80)

    ae_model = DeepAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS
    )

    print(f"Architecture: {INPUT_DIM} → {HIDDEN_DIMS[0]} → {HIDDEN_DIMS[1]} → {LATENT_DIM} (latent)")
    print(f"Training on {len(X_train_normal):,} normal transactions only")

    ae_training = train_autoencoder(
        model=ae_model,
        X_train=X_train_normal,
        X_val=X_val,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )

    print(f"\nAutoencoder training complete:")
    print(f"  Best epoch: {ae_training['best_epoch']}")
    print(f"  Best val loss: {ae_training['best_val_loss']:.4f}")
    print(f"  Training time: {ae_training['training_time']:.1f}s")
    print(f"  Parameters: {ae_training['param_count']:,}")

    # ==================================================================
    # STEP 3: Extract latent features for all sets
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 3: EXTRACTING LATENT FEATURES FROM AUTOENCODER")
    print("=" * 80)

    print(f"\nExtracting {LATENT_DIM}D latent features...")

    X_train_latent = extract_latent_features(
        ae_training['model'], X_train, DEVICE, batch_size=BATCH_SIZE
    )
    X_val_latent = extract_latent_features(
        ae_training['model'], X_val, DEVICE, batch_size=BATCH_SIZE
    )
    X_test_latent = extract_latent_features(
        ae_training['model'], X_test, DEVICE, batch_size=BATCH_SIZE
    )

    print(f"Train latent features: {X_train_latent.shape}")
    print(f"Val latent features: {X_val_latent.shape}")
    print(f"Test latent features: {X_test_latent.shape}")

    # ==================================================================
    # STEP 4: Train Logistic Regression on latent features
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING LOGISTIC REGRESSION ON LATENT FEATURES")
    print("=" * 80)

    print(f"\nTraining LR on {LATENT_DIM}D latent features (instead of {INPUT_DIM}D raw features)")
    print("Using class_weight='balanced' to handle class imbalance")

    lr_hybrid = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver='lbfgs'
    )

    lr_hybrid.fit(X_train_latent, y_train)

    print(f"LR training complete")
    print(f"Feature weights shape: {lr_hybrid.coef_.shape}")
    print(f"Most important latent dimension: {np.argmax(np.abs(lr_hybrid.coef_[0]))}")

    # ==================================================================
    # STEP 5: Evaluate on validation set
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 5: VALIDATION SET EVALUATION")
    print("=" * 80)

    y_val_proba = lr_hybrid.predict_proba(X_val_latent)[:, 1]

    val_metrics = compute_all_metrics(
        y_val, y_val_proba, "Hybrid (AE→LR)"
    )

    print_evaluation_summary(val_metrics, "Validation")

    # ==================================================================
    # STEP 6: Evaluate on test set (FINAL RESULTS)
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 6: TEST SET EVALUATION (FINAL RESULTS)")
    print("=" * 80)

    y_test_proba = lr_hybrid.predict_proba(X_test_latent)[:, 1]

    test_metrics = compute_all_metrics(
        y_test, y_test_proba, "Hybrid (AE→LR)"
    )

    print_evaluation_summary(test_metrics, "Test")

    # ==================================================================
    # STEP 7: Compare to baselines
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 7: COMPARISON TO BASELINES")
    print("=" * 80)

    print("\nPerformance Comparison (Test Set):")
    print("-" * 60)
    print(f"{'Model':<30} {'ROC-AUC':>12} {'PR-AUC':>12} {'Recall@90%P':>12}")
    print("-" * 60)
    print(f"{'Pure LR (29D raw features)':<30} {0.982:>12.3f} {0.744:>12.3f} {0.68:>12.2f}")
    print(f"{'Pure AE (reconstruction)':<30} {0.950:>12.3f} {0.152:>12.3f} {0.00:>12.2f}")
    print(f"{'Hybrid (AE→LR, 8D latent)':<30} {test_metrics['roc_auc']:>12.3f} {test_metrics['pr_auc']:>12.3f} {test_metrics['recall_at_90p']:>12.2f}")
    print("-" * 60)

    # Calculate improvements
    pr_auc_improvement_vs_ae = (test_metrics['pr_auc'] - 0.152) / 0.152 * 100
    pr_auc_improvement_vs_lr = (test_metrics['pr_auc'] - 0.744) / 0.744 * 100

    print(f"\nImprovement vs Pure AE: {pr_auc_improvement_vs_ae:+.1f}% (PR-AUC)")
    print(f"Improvement vs Pure LR: {pr_auc_improvement_vs_lr:+.1f}% (PR-AUC)")

    if test_metrics['pr_auc'] > 0.744:
        print("\n🎉 HYBRID APPROACH OUTPERFORMS BOTH PURE METHODS!")
    elif test_metrics['pr_auc'] > 0.152:
        print("\n✅ Hybrid approach successfully bridges supervised and unsupervised methods")

    # ==================================================================
    # STEP 8: Save models for deployment
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVING MODELS FOR DEPLOYMENT")
    print("=" * 80)

    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save autoencoder (encoder only needed for inference)
    ae_path = os.path.join(models_dir, 'hybrid_ae_encoder.pth')
    torch.save({
        'model_state_dict': ae_training['model'].state_dict(),
        'architecture': {
            'input_dim': INPUT_DIM,
            'latent_dim': LATENT_DIM,
            'hidden_dims': HIDDEN_DIMS
        },
        'training_info': {
            'best_epoch': ae_training['best_epoch'],
            'best_val_loss': ae_training['best_val_loss'],
            'training_time': ae_training['training_time']
        }
    }, ae_path)
    print(f"Autoencoder saved to: {ae_path}")

    # Save logistic regression
    lr_path = os.path.join(models_dir, 'hybrid_lr_classifier.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_hybrid, f)
    print(f"LR classifier saved to: {lr_path}")

    # Save scaler
    scaler_path = os.path.join(models_dir, 'hybrid_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save inference example
    inference_example_path = os.path.join(models_dir, 'hybrid_inference_example.py')
    with open(inference_example_path, 'w') as f:
        f.write('''"""
Example: How to use the hybrid model for inference on new transactions.
"""

import numpy as np
import torch
import pickle
from src.models import DeepAutoencoder

# Load models
checkpoint = torch.load('hybrid_ae_encoder.pth')
ae_model = DeepAutoencoder(
    input_dim=checkpoint['architecture']['input_dim'],
    latent_dim=checkpoint['architecture']['latent_dim'],
    hidden_dims=checkpoint['architecture']['hidden_dims']
)
ae_model.load_state_dict(checkpoint['model_state_dict'])
ae_model.eval()

with open('hybrid_lr_classifier.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('hybrid_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Classify new transaction
def classify_transaction(transaction):
    """
    Args:
        transaction: numpy array of shape (29,) with raw features

    Returns:
        fraud_probability: float in [0, 1]
        prediction: 0 (normal) or 1 (fraud)
    """
    # Standardize
    transaction_scaled = scaler.transform(transaction.reshape(1, -1))

    # Extract latent features
    with torch.no_grad():
        transaction_tensor = torch.FloatTensor(transaction_scaled)
        latent = ae_model.encoder(transaction_tensor).numpy()

    # Classify
    fraud_probability = lr_model.predict_proba(latent)[0, 1]
    prediction = 1 if fraud_probability > 0.5 else 0

    return fraud_probability, prediction

# Example usage
new_transaction = np.random.randn(29)  # Replace with real transaction
prob, pred = classify_transaction(new_transaction)
print(f"Fraud Probability: {prob:.3f}")
print(f"Prediction: {'FRAUD' if pred == 1 else 'NORMAL'}")
''')
    print(f"Inference example saved to: {inference_example_path}")

    # ==================================================================
    # STEP 9: Save results to CSV/JSON
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 9: SAVING RESULTS")
    print("=" * 80)

    # Save to existing results table
    results_row = {
        'Model': 'Hybrid (AE→LR)',
        'ROC-AUC': float(test_metrics['roc_auc']),
        'PR-AUC': float(test_metrics['pr_auc']),
        'Recall@90%P': float(test_metrics['recall_at_90p']),
        'Threshold@90%P': float(test_metrics['threshold_at_90p'])
    }

    # Append to existing CSV
    csv_path = os.path.join(TABLES_DIR, 'final_test_results.csv')
    df_existing = pd.read_csv(csv_path)
    df_new = pd.concat([df_existing, pd.DataFrame([results_row])], ignore_index=True)
    df_new.to_csv(csv_path, index=False)
    print(f"Results appended to: {csv_path}")

    # Save detailed results
    hybrid_results_path = os.path.join(TABLES_DIR, 'hybrid_detailed_results.json')
    hybrid_results = {
        'model_name': 'Hybrid (AE→LR)',
        'approach': 'Autoencoder feature learning + Logistic Regression',
        'latent_dim': LATENT_DIM,
        'ae_training': {
            'best_epoch': int(ae_training['best_epoch']),
            'best_val_loss': float(ae_training['best_val_loss']),
            'training_time_seconds': float(ae_training['training_time']),
            'parameters': int(ae_training['param_count'])
        },
        'lr_training': {
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'max_iter': 1000
        },
        'validation_metrics': {
            'roc_auc': float(val_metrics['roc_auc']),
            'pr_auc': float(val_metrics['pr_auc']),
            'recall_at_90p': float(val_metrics['recall_at_90p'])
        },
        'test_metrics': {
            'roc_auc': float(test_metrics['roc_auc']),
            'pr_auc': float(test_metrics['pr_auc']),
            'recall_at_90p': float(test_metrics['recall_at_90p']),
            'threshold_at_90p': float(test_metrics['threshold_at_90p'])
        },
        'comparison': {
            'pr_auc_vs_pure_ae': {
                'pure_ae': 0.152,
                'hybrid': float(test_metrics['pr_auc']),
                'improvement_percent': float(pr_auc_improvement_vs_ae)
            },
            'pr_auc_vs_pure_lr': {
                'pure_lr': 0.744,
                'hybrid': float(test_metrics['pr_auc']),
                'improvement_percent': float(pr_auc_improvement_vs_lr)
            }
        }
    }

    with open(hybrid_results_path, 'w') as f:
        json.dump(hybrid_results, f, indent=2)
    print(f"Detailed results saved to: {hybrid_results_path}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("HYBRID APPROACH COMPLETE!")
    print("=" * 80)

    print(f"\n✅ Autoencoder feature learning: {LATENT_DIM}D latent representation")
    print(f"✅ Supervised classification: Logistic Regression on learned features")
    print(f"✅ Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"✅ Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"✅ Recall@90%P: {test_metrics['recall_at_90p']:.2f}")

    print(f"\n📦 Models saved for deployment:")
    print(f"   - {ae_path}")
    print(f"   - {lr_path}")
    print(f"   - {scaler_path}")

    print(f"\n📊 Results saved to:")
    print(f"   - {csv_path}")
    print(f"   - {hybrid_results_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
