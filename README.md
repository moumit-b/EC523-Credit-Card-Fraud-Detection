# Credit Card Fraud Detection with Deep Autoencoders

**EC523 Deep Learning - Spring 2025**

## Group Members

- Moumit Bhattacharjee - moumitb@bu.edu
- Rohan Hegde - rohanh@bu.edu
- Emmanuel Herold - ebherold@bu.edu

## Project Overview

This project explores anomaly detection for credit card fraud using deep autoencoders. The approach trains an autoencoder on legitimate (non-fraud) transactions and uses reconstruction error as an anomaly score to identify fraudulent transactions. We compare the deep learning approach against classical baseline methods including Logistic Regression and Isolation Forest.

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. The dataset contains transactions made by European cardholders in September 2013.

**Dataset characteristics:**
- 284,807 transactions over 2 days
- 492 frauds (0.172% of all transactions) - highly imbalanced
- Features V1-V28: PCA-transformed features (confidentiality)
- Feature `Time`: seconds elapsed since first transaction
- Feature `Amount`: transaction amount
- Feature `Class`: 0 for legitimate, 1 for fraud

**Important:** The dataset file (`creditcard.csv`) must be downloaded manually from Kaggle and placed in the `data/` directory.

## Current Status

### Implemented So Far

**Data Pipeline:**
- Data loading with proper error handling
- Time-ordered train/validation/test split (60%/20%/20%)
- Feature standardization using training set statistics only (prevents data leakage)
- Basic exploratory data analysis (sample counts, fraud rates)

**Baseline Models:**
- Class-weighted Logistic Regression
- Isolation Forest (unsupervised anomaly detection)
- Evaluation metrics: ROC-AUC and PR-AUC on validation set

**Visualizations:**
- Class distribution bar chart
- Additional plotting utilities for amount and time distributions

### Not Yet Implemented

The following components will be developed in subsequent phases:

- **Deep autoencoder architecture** (PyTorch implementation)
- **Denoising autoencoder** variant
- **Anomaly scoring mechanism** using reconstruction error
- **Cost-sensitive evaluation** framework (λ·FN + FP)
- **Advanced metrics**: Recall@90% Precision
- **Comprehensive model comparison** and ablation studies
- **Final report and presentation**

## Project Structure

```
.
├── data/
│   └── creditcard.csv          # Download from Kaggle (not included)
├── docs/
│   ├── CCF1.pdf                # Project description
│   ├── CCF2.pdf                # Project proposal
│   └── roadmap.md              # Project roadmap and timeline
├── src/
│   ├── data_utils.py           # Data loading and preprocessing
│   ├── baselines.py            # Baseline models
│   └── plot_utils.py           # Visualization utilities
├── scripts/
│   └── run_eda_and_baseline.py # Main EDA and baseline script
├── figures/                    # Generated visualizations
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EC523-Credit-Card-Fraud-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   - Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in the `data/` directory

### Running the Code

**Run EDA and baseline models:**
```bash
python scripts/run_eda_and_baseline.py
```

This script will:
- Load and analyze the dataset
- Create time-ordered train/val/test splits
- Standardize features
- Train baseline models (Logistic Regression, Isolation Forest)
- Evaluate models on validation set
- Generate and save visualizations to `figures/`

## Evaluation Metrics

Given the highly imbalanced nature of fraud detection, we use the following metrics:

- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Precision-Recall AUC (more informative for imbalanced datasets)
- **Recall@90% Precision**: Recall achieved at 90% precision threshold (planned)

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for detailed project timeline and remaining work.

**Key milestones:**
1. ✅ Data pipeline and baseline models
2. ⬜ Deep autoencoder implementation
3. ⬜ Anomaly detection and evaluation framework
4. ⬜ Experiments and hyperparameter tuning
5. ⬜ Final report and presentation

## References

- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Original paper: Andrea Dal Pozzolo et al. "Calibrating Probability with Undersampling for Unbalanced Classification." (2015)


