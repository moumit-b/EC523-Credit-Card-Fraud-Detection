"""
Configuration file for Credit Card Fraud Detection project.

Centralizes all hyperparameters, paths, and random seeds for reproducibility.
"""

import os
import torch
import numpy as np
import random

# ============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

def set_all_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "artifacts", "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "artifacts", "tables")

# Data file
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# Ensure artifact directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Train/val/test split ratios
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

# Features to use (V1-V28 + Amount = 29 features)
# Time is NOT used as a feature, only for ordering
FEATURE_COLUMNS = [f'V{i}' for i in range(1, 29)] + ['Amount']
INPUT_DIM = len(FEATURE_COLUMNS)  # 29


# ============================================================================
# AUTOENCODER HYPERPARAMETERS
# ============================================================================
# Architecture
LATENT_DIM = 8  # Bottleneck dimension (will test: 2, 4, 8, 16)
HIDDEN_DIMS = [64, 32]  # Encoder: 29 -> 64 -> 32 -> latent

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10  # Stop if val loss doesn't improve for N epochs

# Denoising Autoencoder
NOISE_LEVEL = 0.1  # Gaussian noise std (will test: 0.05, 0.1)


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Operating point
TARGET_PRECISION = 0.90  # For Recall@90%Precision metric

# Cost-sensitive evaluation
LAMBDA_VALUES = [10, 50, 100]  # Cost ratios for λ·FN + FP


# ============================================================================
# BASELINE MODEL CONFIGURATION
# ============================================================================
# Logistic Regression
LR_MAX_ITER = 1000
LR_SOLVER = 'lbfgs'

# Isolation Forest
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 'auto'


# ============================================================================
# LOGGING & VISUALIZATION
# ============================================================================
# Figure settings
FIG_DPI = 300
FIG_FORMAT = 'png'

# Progress bars
SHOW_PROGRESS = True

# Verbose logging
VERBOSE = True


# ============================================================================
# MODEL SAVE PATHS
# ============================================================================
def get_model_path(model_name, **kwargs):
    """
    Generate model save path with hyperparameters in filename.

    Args:
        model_name: 'ae' or 'dae'
        **kwargs: Hyperparameters to include in filename

    Returns:
        Path to model file
    """
    parts = [model_name]
    for key, val in kwargs.items():
        parts.append(f"{key}{val}")
    filename = "_".join(parts) + ".pth"
    return os.path.join(MODELS_DIR, filename)


# ============================================================================
# REPORT METADATA
# ============================================================================
TEAM_MEMBERS = [
    "Rohan Hegde (rohanh@bu.edu)",
    "Moumit Bhattacharjee (moumitb@bu.edu)",
    "Emmanuel Herold (ebherold@bu.edu)"
]

PROJECT_TITLE = "Credit Card Fraud Detection with Deep Autoencoders"
COURSE = "EC523 Deep Learning - Spring 2025"


if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Input Dimension: {INPUT_DIM}")
    print(f"Latent Dimension: {LATENT_DIM}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"Figures Dir: {FIGURES_DIR}")
    print("=" * 60)
