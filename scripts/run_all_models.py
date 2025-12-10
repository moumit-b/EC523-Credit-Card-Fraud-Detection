"""
Comprehensive training and evaluation script for ALL models.

This script:
1. Trains/loads baseline models (Logistic Regression, Isolation Forest)
2. Trains/loads Deep Autoencoder
3. Trains/loads Denoising Autoencoder
4. Trains Hybrid model (AE→LR) using latent features
5. Evaluates ALL models on validation AND test sets
6. Generates comprehensive comparison tables and figures (with F1-optimal metrics)
7. Saves all results for report generation

Models evaluated (5 total):
- Logistic Regression (supervised baseline)
- Isolation Forest (unsupervised baseline)
- Deep Autoencoder (unsupervised)
- Denoising Autoencoder (unsupervised with noise regularization)
- Hybrid AE→LR (semi-supervised: AE feature learning + LR classification)

This is the main script for generating final report results.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import torch

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
from src.models import DeepAutoencoder, DenoisingAutoencoder
from src.baselines import train_logistic_regression, train_isolation_forest
from src.training import train_autoencoder, evaluate_reconstruction_error
from sklearn.linear_model import LogisticRegression
from src.evaluation import (
    compute_all_metrics,
    print_evaluation_summary,
    compare_models,
    compute_cost_sensitive_metrics,
    print_cost_sensitive_analysis
)
from src.visualization import (
    plot_pr_curves,
    plot_roc_curves,
    plot_comparison_table,
    plot_cost_vs_lambda,
    plot_training_curves,
    plot_reconstruction_error_distribution,
    plot_f1_comparison,
    plot_confusion_matrices,
    plot_single_confusion_matrix
)


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

    # Set all random seeds
    set_all_seeds(RANDOM_SEED)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    print(f"Course: {COURSE}")
    print(f"Project: {PROJECT_TITLE}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    # ==================================================================
    # STEP 1: Load and preprocess data
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("=" * 80)

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

    # Extract normal transactions for autoencoder training
    X_train_normal = get_normal_transactions_only(X_train, y_train)

    # ==================================================================
    # STEP 2: Train Baseline Models
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING BASELINE MODELS")
    print("=" * 80)

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_results = train_logistic_regression(
        X_train, y_train, X_val, y_val,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED
    )

    # Isolation Forest
    print("\n--- Isolation Forest ---")
    if_results = train_isolation_forest(
        X_train, y_train, X_val, y_val,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_SEED
    )

    # ==================================================================
    # STEP 3: Train Autoencoder Models
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING AUTOENCODER MODELS")
    print("=" * 80)

    # Deep Autoencoder
    print("\n--- Deep Autoencoder ---")
    ae_model = DeepAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS
    )

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

    # Plot AE training curves (MANDATORY for report)
    ae_train_curve_path = os.path.join(FIGURES_DIR, f'ae_training_curves_latent{LATENT_DIM}.png')
    plot_training_curves(
        train_losses=ae_training['train_losses'],
        val_losses=ae_training['val_losses'],
        model_name=f"Deep Autoencoder (latent={LATENT_DIM})",
        save_path=ae_train_curve_path
    )
    print(f"Training curves saved to: {ae_train_curve_path}")

    # Denoising Autoencoder
    print("\n--- Denoising Autoencoder ---")
    dae_model = DenoisingAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        noise_level=NOISE_LEVEL
    )

    dae_training = train_autoencoder(
        model=dae_model,
        X_train=X_train_normal,
        X_val=X_val,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )

    # Plot DAE training curves (MANDATORY for report)
    dae_train_curve_path = os.path.join(FIGURES_DIR, f'dae_training_curves_latent{LATENT_DIM}_noise{int(NOISE_LEVEL*100)}.png')
    plot_training_curves(
        train_losses=dae_training['train_losses'],
        val_losses=dae_training['val_losses'],
        model_name=f"Denoising Autoencoder (latent={LATENT_DIM}, noise={NOISE_LEVEL})",
        save_path=dae_train_curve_path
    )
    print(f"Training curves saved to: {dae_train_curve_path}")

    # ==================================================================
    # STEP 3.5: Train Hybrid Model (AE → LR)
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 3.5: TRAINING HYBRID MODEL (AE → LR)")
    print("=" * 80)

    print(f"\nExtracting {LATENT_DIM}D latent features from Deep Autoencoder...")

    # Extract latent features for all sets
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

    print(f"\nTraining Logistic Regression on {LATENT_DIM}D latent features...")
    print("Using class_weight='balanced' to handle class imbalance")

    lr_hybrid = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver='lbfgs'
    )

    lr_hybrid.fit(X_train_latent, y_train)

    print(f"Hybrid model training complete")
    print(f"LR feature weights shape: {lr_hybrid.coef_.shape}")

    # ==================================================================
    # STEP 4: Evaluate ALL models on VALIDATION set
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 4: VALIDATION SET EVALUATION")
    print("=" * 80)

    # Get reconstruction errors for autoencoders
    ae_val_errors = evaluate_reconstruction_error(
        ae_training['model'], X_val, DEVICE, BATCH_SIZE
    )
    dae_val_errors = evaluate_reconstruction_error(
        dae_training['model'], X_val, DEVICE, BATCH_SIZE
    )

    # Compute metrics for all models
    val_results = []

    # Baselines
    lr_val_metrics = compute_all_metrics(
        y_val, lr_results['y_pred_proba'], "Logistic Regression"
    )
    val_results.append(lr_val_metrics)

    if_val_metrics = compute_all_metrics(
        y_val, if_results['anomaly_scores'], "Isolation Forest"
    )
    val_results.append(if_val_metrics)

    # Autoencoders
    ae_val_metrics = compute_all_metrics(
        y_val, ae_val_errors, "Deep Autoencoder"
    )
    val_results.append(ae_val_metrics)

    dae_val_metrics = compute_all_metrics(
        y_val, dae_val_errors, "Denoising Autoencoder"
    )
    val_results.append(dae_val_metrics)

    # Hybrid model
    hybrid_val_scores = lr_hybrid.predict_proba(X_val_latent)[:, 1]
    hybrid_val_metrics = compute_all_metrics(
        y_val, hybrid_val_scores, "Hybrid (AE→LR)"
    )
    val_results.append(hybrid_val_metrics)

    # Print comparison
    compare_models(val_results, split_name="Validation")

    # ==================================================================
    # STEP 5: Evaluate ALL models on TEST set (FINAL RESULTS)
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 5: TEST SET EVALUATION (FINAL RESULTS)")
    print("=" * 80)

    # Get scores/errors for all models on test set
    lr_test_scores = lr_results['model'].predict_proba(X_test)[:, 1]
    if_test_scores = -if_results['model'].score_samples(X_test)
    ae_test_errors = evaluate_reconstruction_error(
        ae_training['model'], X_test, DEVICE, BATCH_SIZE
    )
    dae_test_errors = evaluate_reconstruction_error(
        dae_training['model'], X_test, DEVICE, BATCH_SIZE
    )

    # Plot reconstruction error distributions
    ae_error_dist_path = os.path.join(FIGURES_DIR, f'ae_error_dist_latent{LATENT_DIM}.png')
    plot_reconstruction_error_distribution(
        errors_normal=ae_test_errors[y_test == 0],
        errors_fraud=ae_test_errors[y_test == 1],
        model_name=f"Deep Autoencoder (latent={LATENT_DIM})",
        save_path=ae_error_dist_path
    )
    print(f"AE error distribution saved to: {ae_error_dist_path}")

    dae_error_dist_path = os.path.join(FIGURES_DIR, f'dae_error_dist_latent{LATENT_DIM}_noise{int(NOISE_LEVEL*100)}.png')
    plot_reconstruction_error_distribution(
        errors_normal=dae_test_errors[y_test == 0],
        errors_fraud=dae_test_errors[y_test == 1],
        model_name=f"Denoising Autoencoder (latent={LATENT_DIM}, noise={NOISE_LEVEL})",
        save_path=dae_error_dist_path
    )
    print(f"DAE error distribution saved to: {dae_error_dist_path}")

    # Compute test metrics
    test_results = []

    lr_test_metrics = compute_all_metrics(
        y_test, lr_test_scores, "Logistic Regression"
    )
    test_results.append(lr_test_metrics)
    print_evaluation_summary(lr_test_metrics, "Test")

    if_test_metrics = compute_all_metrics(
        y_test, if_test_scores, "Isolation Forest"
    )
    test_results.append(if_test_metrics)
    print_evaluation_summary(if_test_metrics, "Test")

    ae_test_metrics = compute_all_metrics(
        y_test, ae_test_errors, "Deep Autoencoder"
    )
    test_results.append(ae_test_metrics)
    print_evaluation_summary(ae_test_metrics, "Test")

    dae_test_metrics = compute_all_metrics(
        y_test, dae_test_errors, "Denoising Autoencoder"
    )
    test_results.append(dae_test_metrics)
    print_evaluation_summary(dae_test_metrics, "Test")

    # Hybrid model
    hybrid_test_scores = lr_hybrid.predict_proba(X_test_latent)[:, 1]
    hybrid_test_metrics = compute_all_metrics(
        y_test, hybrid_test_scores, "Hybrid (AE→LR)"
    )
    test_results.append(hybrid_test_metrics)
    print_evaluation_summary(hybrid_test_metrics, "Test")

    # Print final comparison
    compare_models(test_results, split_name="Test")

    # ==================================================================
    # STEP 6: Cost-Sensitive Analysis
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 6: COST-SENSITIVE ANALYSIS")
    print("=" * 80)

    cost_results_dict = {}

    for result in test_results:
        model_name = result['model_name']
        cost_res = compute_cost_sensitive_metrics(
            y_test, result['y_scores'], LAMBDA_VALUES
        )
        cost_results_dict[model_name] = cost_res
        print_cost_sensitive_analysis(cost_res, model_name)

    # ==================================================================
    # STEP 7: Generate Visualizations
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)

    # PRIMARY VISUALIZATIONS: F1-based evaluation
    # F1 comparison across models
    f1_comp_path = os.path.join(FIGURES_DIR, 'final_f1_comparison.png')
    plot_f1_comparison(test_results, save_path=f1_comp_path,
                      title="Test Set: F1 Score Comparison at Optimal Thresholds")

    # Confusion matrices grid
    conf_mat_path = os.path.join(FIGURES_DIR, 'final_confusion_matrices.png')
    plot_confusion_matrices(test_results, save_path=conf_mat_path,
                           title="Test Set: Confusion Matrices at F1-Optimal Thresholds")

    # Individual confusion matrices for each model (for slides)
    for result in test_results:
        model_name_clean = result['model_name'].replace(' ', '_').lower()
        single_cm_path = os.path.join(FIGURES_DIR, f'confusion_matrix_{model_name_clean}.png')
        plot_single_confusion_matrix(result, save_path=single_cm_path)

    # SECONDARY VISUALIZATIONS: PR/ROC curves
    # PR curves
    pr_path = os.path.join(FIGURES_DIR, 'final_pr_curves_comparison.png')
    plot_pr_curves(test_results, save_path=pr_path, title="Test Set: Precision-Recall Curves")

    # ROC curves
    roc_path = os.path.join(FIGURES_DIR, 'final_roc_curves_comparison.png')
    plot_roc_curves(test_results, save_path=roc_path, title="Test Set: ROC Curves")

    # Comparison table
    comp_path = os.path.join(FIGURES_DIR, 'final_model_comparison.png')
    plot_comparison_table(test_results, split_name="Test", save_path=comp_path)

    # Cost vs lambda
    cost_path = os.path.join(FIGURES_DIR, 'final_cost_vs_lambda.png')
    plot_cost_vs_lambda(cost_results_dict, save_path=cost_path)

    # ==================================================================
    # STEP 8: Save Results to JSON/CSV
    # ==================================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVING RESULTS")
    print("=" * 80)

    # Create results table (convert numpy types to Python native types for JSON serialization)
    # Focus on primary metrics: PR-AUC and F1-optimal operating point
    results_table = []
    for result in test_results:
        results_table.append({
            'Model': result['model_name'],
            # Primary metrics
            'PR-AUC': float(result['pr_auc']),
            'F1': float(result['f1_optimal_f1']),
            'Precision': float(result['f1_optimal_precision']),
            'Recall': float(result['f1_optimal_recall']),
            'Accuracy': float(result['f1_optimal_accuracy']),
            'F1_Threshold': float(result['f1_optimal_threshold']),
            # Confusion matrix counts
            'TN': int(result['f1_optimal_tn']),
            'FP': int(result['f1_optimal_fp']),
            'FN': int(result['f1_optimal_fn']),
            'TP': int(result['f1_optimal_tp']),
            # Secondary metrics (for reference)
            'ROC-AUC': float(result['roc_auc']),
            'Recall@90%P': float(result['recall_at_90p']),
            'Threshold@90%P': float(result['threshold_at_90p'])
        })

    results_df = pd.DataFrame(results_table)

    # Save as CSV
    csv_path = os.path.join(TABLES_DIR, 'final_test_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

    # Save as JSON
    json_path = os.path.join(TABLES_DIR, 'final_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_table, f, indent=2)
    print(f"Results JSON saved to: {json_path}")

    # Save training metadata (ensure all values are JSON-serializable)
    metadata = {
        'random_seed': int(RANDOM_SEED),
        'device': str(DEVICE),
        'dataset_path': str(DATA_PATH),
        'train_samples': int(len(X_train)),
        'train_normal_samples': int(len(X_train_normal)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'ae_params': int(ae_training['param_count']),
        'ae_training_time': float(ae_training['training_time']),
        'dae_params': int(dae_training['param_count']),
        'dae_training_time': float(dae_training['training_time'])
    }

    meta_path = os.path.join(TABLES_DIR, 'experiment_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Experiment metadata saved to: {meta_path}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nFinal Test Results:")
    print(results_df[['Model', 'PR-AUC', 'F1', 'Precision', 'Recall', 'Accuracy']].to_string(index=False))
    print("\nGenerated Artifacts:")
    print("\nPRIMARY VISUALIZATIONS (F1-based):")
    print(f"  - {f1_comp_path}")
    print(f"  - {conf_mat_path}")
    print(f"  - Individual confusion matrices for each model in {FIGURES_DIR}/")
    print("\nSECONDARY VISUALIZATIONS:")
    print(f"  - {pr_path}")
    print(f"  - {roc_path}")
    print(f"  - {comp_path}")
    print(f"  - {cost_path}")
    print("\nRESULTS DATA:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"  - {meta_path}")
    print("\nAll results saved to artifacts/ directory")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
