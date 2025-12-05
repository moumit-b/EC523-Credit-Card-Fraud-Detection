#Template for evaluation of model, needs to be edited 

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import load_creditcard_data, time_ordered_split, standardize_features
from src.models import DeepAutoencoder


MODEL_PATH = "models/autoencoder.pth"
FIGURES_DIR = "figures"

BASELINE_SCORES = {
    'Logistic Regression': 0.7725,
    'Isolation Forest': 0.0210
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def main():
    print(f"Running evaluation on device: {DEVICE}")

    # 1. Load Data (Must use exact same split/scaling as training)
    print("\nLoading and processing Test data...")
    df = load_creditcard_data("data/creditcard.csv")
    train_df, val_df, test_df = time_ordered_split(df)
    
    # We only need X_test and y_test here
    _, _, _, _, X_test, y_test, _ = standardize_features(train_df, val_df, test_df)
    
    # 2. Load Model
    input_dim = X_test.shape[1]
    model = DeepAutoencoder(input_dim=input_dim).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run scripts/train_autoencoder.py first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # 3. Inference: Calculate Reconstruction Error
    print("\nCalculating Anomaly Scores...")
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    with torch.no_grad():
        # Get reconstructions
        reconstructed = model(X_test_tensor)
        
        # Calculate MSE per transaction (this is our Anomaly Score)
        # Shape: [batch_size]
        loss_per_sample = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1)
        
        # Move to CPU for metrics
        y_scores = loss_per_sample.cpu().numpy()

    # 4. Compute Metrics
    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    # Calculate Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    # Find the recall value where precision is closest to 0.9
    try:
        idx = np.where(precision >= 0.90)[0][0]
        recall_at_90 = recall[idx]
    except IndexError:
        recall_at_90 = 0.0  # Precision never reached 90%

    # 5. Print Results & Comparison
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"{'Metric':<20} {'Autoencoder':<15}")
    print("-" * 50)
    print(f"{'ROC-AUC':<20} {roc_auc:.4f}")
    print(f"{'PR-AUC':<20} {pr_auc:.4f}")
    print(f"{'Recall @ 90% Prec':<20} {recall_at_90:.4f}")
    
    print("\n" + "="*50)
    print("HEAD-TO-HEAD COMPARISON (PR-AUC)")
    print("="*50)
    print(f"{'Model':<20} {'Score':<10} {'Gap'}")
    print("-" * 50)
    
    # Compare with Baselines
    for name, score in BASELINE_SCORES.items():
        diff = pr_auc - score
        print(f"{name:<20} {score:.4f}     {diff:+.4f}")
    
    print(f"{'Autoencoder (You)':<20} {pr_auc:.4f}")

    # 6. Generate Visualizations
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Plot A: Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Autoencoder (AUC={pr_auc:.4f})', linewidth=2)
    plt.axhline(y=BASELINE_SCORES['Logistic Regression'], color='gray', linestyle='--', label='Logistic Reg Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{FIGURES_DIR}/pr_curve_comparison.png")
    
    # Plot B: Reconstruction Error Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_scores[y_test==0], bins=100, density=True, alpha=0.6, color='green', label='Normal')
    plt.hist(y_scores[y_test==1], bins=100, density=True, alpha=0.6, color='red', label='Fraud')
    
    # Log scale *if needed*
    plt.yscale('log') 
    plt.title('Reconstruction Error Distribution (Log Scale)')
    plt.xlabel('Anomaly Score (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/anomaly_score_dist.png")
    
    print(f"\nVisualizations saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
