# Credit Card Fraud Detection with Deep Autoencoders

**EC523 Deep Learning - Spring 2025**

## Group Members

- Moumit Bhattacharjee - moumitb@bu.edu
- Rohan Hegde - rohanh@bu.edu
- Emmanuel Herold - ebherold@bu.edu

## Project Overview

This project explores anomaly detection for credit card fraud using deep autoencoders. The approach trains an autoencoder on legitimate (non-fraud) transactions and uses reconstruction error as an anomaly score to identify fraudulent transactions. We compare the deep learning approach against classical baseline methods including Logistic Regression and Isolation Forest.

## Dataset and Metrics

The Kaggle Credit Card Fraud Detection dataset is a .csv file that contains 284,807 transactions, of which only 492 are fraudulent (0.17%). This extreme imbalance makes the classification task challenging and motivates our choice of modeling strategy and evaluation metrics. Each transaction includes a timestamp, the transaction amount, and 28 PCA-transformed features (V1–V28), with PCA applied to protect confidentiality.

**Important:** The dataset file (`creditcard.csv`) must be downloaded manually from Kaggle and placed in the `data/` directory. Download from: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

For evaluation, we emphasize metrics appropriate for highly imbalanced anomaly detection. **PR-AUC** is our primary metric because it focuses on model performance on the minority class and avoids the misleading optimism of accuracy or ROC-AUC. We use **F1 score** on the validation set to select operating thresholds, enabling a balanced trade-off between precision and recall. At first, we aimed to also use **Recall @ 90% Precision**, reflecting real-world deployment requirements where banks must maintain high precision to avoid customer disruption while still detecting as much fraud as possible. However, the nature of the dataset and our unsupervised models made this metric irrelevant. Finally, we include **cost-sensitive evaluation** under the cost function Cost = λ·FN + FP, computing total cost under different values of λ to reflect the asymmetric real-world impact of false negatives versus false positives. These choices ensure that our evaluation aligns with realistic constraints of financial fraud detection systems.

## Approach

Autoencoders are neural networks designed to learn compressed representations of data by encoding an input into a lower-dimensional latent space and then reconstructing it. During training, the model learns to reproduce only the patterns that are common and consistent in the training distribution. In our setting, we train the autoencoder exclusively on normal (non-fraud) transactions, allowing it to learn the underlying structure of legitimate behavior. At inference time, transactions that do not align with this learned structure are expected to reconstruct poorly. We quantify this reconstruction quality using Mean Squared Error (MSE) between the input and its reconstruction. High reconstruction error indicates that a transaction lies outside the normal manifold and is therefore flagged as a potential anomaly:

s(x) = ||x - x̂||²₂

Our experiment compares the performance of five main models on the trained data: Logistic Regression, Isolation Forest (best performing classical model for anomaly detection), a base Deep Autoencoder, a Denoising Autoencoder, and a Hybrid autoencoder that also introduces Logistic Regression to the latent layer.

### Training Architecture

To address concept drift and avoid temporal leakage, we split the dataset in a time-ordered 60/20/20 manner, sorting by timestamp and training on the earliest transactions and validating/testing on more recent ones. All features are standardized using StandardScaler to ensure stable optimization for both logistic regression and the autoencoders. Because of the severe class imbalance, our Logistic Regression baseline uses `class_weight="balanced"`, which increases the relative importance of rare fraud samples so the model does not default to predicting only the majority class.

**Hardware:**
- NVIDIA RTX 3070 Ti (8GB VRAM)
- ~10× speedup vs CPU

**Training Configuration:**
- Batch Size: 256
- Learning Rate: 0.001
- Optimizer: Adam
- Early Stopping: Patience 10

**Data Splits:**
- Training Set: 171,009 samples
- Validation Set: 56,888 samples
- Test Set: 56,910 samples

### Logistic Regression

Logistic regression serves as our supervised baseline. Given a transaction feature vector **x** ∈ ℝ²⁹, the model computes a fraud probability: p(y = fraud | **x**) = σ(**w**ᵀ**x** + b) where σ is the sigmoid function, **w** are learned weights, and b is the bias term. Training minimizes binary cross-entropy loss: ℒ = -[y log(p) + (1-y) log(1-p)]. To handle class imbalance, we apply class weighting, setting the weight for the fraud class inversely proportional to its frequency. This amplifies the loss for misclassified fraud cases. We use the scikit-learn implementation with `class_weight='balanced'`, LBFGS solver, and `max_iter=1000`. The model then outputs calibrated probabilities. We determine the optimal decision threshold on the validation set by maximizing F1 score. This threshold is then applied to test set predictions to generate binary labels and confusion matrices.

### Isolation Forest

Isolation Forest is an ensemble method that builds multiple isolation trees on random subsamples of the data. Each tree recursively partitions the feature space by randomly selecting a feature and a split value. Anomalies, being "few and different," require fewer splits to isolate and thus have shorter average path lengths. The anomaly score for a transaction is derived from its average path length across all trees, normalized by the expected path length for a dataset of that size. We train Isolation Forest primarily on normal transactions (though it can handle small contamination). Key hyperparameters include the number of trees (`n_estimators=100`), subsample size (`max_samples=256`), and contamination rate (`contamination=0.002`, roughly matching the fraud rate). The output anomaly scores are inverted and scaled to align with our evaluation framework where higher scores indicate higher fraud probability.

### Deep Autoencoder

Our deep autoencoder uses a symmetric fully-connected architecture with ReLU activations:

- **Encoder:** 29 → 64 → 32 → 8
- **Decoder:** 8 → 32 → 64 → 29

The 8-dimensional bottleneck forces the network to learn a compressed representation. The encoder compresses 29-dimensional transaction features into an 8-dimensional latent representation through two hidden layers (64 and 32 neurons). The decoder mirrors this structure to reconstruct the original input. MSE between input and reconstruction serves as both the training loss and the anomaly score at inference. The model is trained exclusively on normal transactions (no fraud labels used during training) to minimize mean squared error (MSE) reconstruction loss: ℒ_AE = ||**x** - **x̂**||²

### Denoising Autoencoder

The denoising autoencoder extends the standard AE by introducing Gaussian noise during training. For each clean input **x**, we generate a corrupted version: **x̃** = **x** + **ε**, where **ε** ~ N(0, σ²I). The network is trained to reconstruct the clean input **x** from the noisy input **x̃**. This denoising objective encourages the model to learn more robust representations that capture the underlying structure of normal transactions rather than memorizing specific patterns. At inference, we do not add noise; the anomaly score is the reconstruction error of the clean input. The noise level σ is a critical hyperparameter. Too little noise provides minimal regularization benefit, while too much noise makes the reconstruction task overly difficult and may prevent the model from learning meaningful patterns. To determine the optimal σ, we conducted an ablation study over σ ∈ {0.0, 0.01, 0.05, 0.1, 0.2}, training separate DAE models for each value and evaluating PR-AUC on the validation set.

### Hybrid Model (AE → LR)

The hybrid model aims to combine the representational power of autoencoders with the discriminative capability of supervised classifiers. The approach proceeds in two stages:

**Stage 1: Unsupervised Feature Learning.** We train a standard autoencoder on normal transactions. After training, we freeze the encoder weights and use the encoder as a fixed feature extractor. For any transaction **x**, the encoder produces an 8-dimensional latent representation **z** = encoder(**x**).

**Stage 2: Supervised Classification.** We extract latent features **z** for all transactions in the training set (both normal and fraud) and train a logistic regression classifier on these 8D latent features using the fraud labels. The LR classifier uses `class_weight='balanced'` to handle imbalance.

The hypothesis is that the autoencoder learns a nonlinear transformation that makes the vector data more robust, making it easier for a linear classifier to separate fraud from normal transactions.

## Results

We evaluate all five models on the held-out test set using metrics appropriate for extreme class imbalance. The table below summarizes performance across all models.

| Model | PR-AUC | F1 | Precision | Recall | ROC-AUC | F1 Threshold |
|-------|--------|-----|-----------|--------|---------|--------------|
| **Logistic Regression** | **0.744** | **0.787** | 0.962 | 0.667 | 0.982 | 0.792 |
| Isolation Forest | 0.034 | 0.097 | 0.054 | 0.520 | 0.952 | 0.100 |
| Deep Autoencoder | 0.205 | 0.389 | 0.327 | 0.480 | 0.938 | 0.390 |
| Denoising Autoencoder | 0.146 | 0.317 | 0.221 | 0.560 | 0.939 | 0.319 |
| Hybrid (AE→LR) | 0.029 | 0.094 | 0.058 | 0.240 | 0.889 | 0.090 |

Logistic regression dominates on PR-AUC and F1 score despite similar ROC-AUC values for autoencoders. F1 thresholds are determined by maximizing F1 on the validation set.

### Precision-Recall Curves

The PR curves reveal the key difference between models. Logistic regression achieves PR-AUC of 0.744, maintaining above 80% precision until approximately 70% recall. In contrast, the best autoencoder (Deep AE) achieves only 0.205 PR-AUC, with precision dropping rapidly as recall increases. The curve shapes demonstrate that autoencoders cannot operate at the high-precision regime required for practical fraud detection. Isolation Forest (0.034 PR-AUC) and the Hybrid model (0.029 PR-AUC) are essentially unusable.

### F1 Scores and Confusion Matrices

For deployment, we select operating thresholds by maximizing F1 score on the validation set. Logistic regression achieves F1 = 0.787, more than double any other model. Autoencoders achieve F1 = 0.389 (Deep AE) and 0.317 (DAE). Isolation Forest and Hybrid are below 0.10.

The confusion matrices reveal the operational trade-offs:

- **Logistic Regression**: 50 true positives with only 2 false positives—96% precision with 67% recall, flagging only 52 transactions to catch 50 frauds.
- **Deep Autoencoder**: Flags 110 transactions (74 false + 36 true) for 33% precision.
- **Denoising Autoencoder**: Flags 190 transactions (148 false + 42 true) for 22% precision.
- **Isolation Forest and Hybrid**: 687 and 290 false positives respectively, making them operationally unacceptable.

In production, analysts reviewing autoencoder alerts would waste significant time on false positives.

### Cost-Sensitive Analysis

We evaluate models under realistic cost assumptions where missing fraud is more expensive than false alarms, using Cost = λ·FN + FP for λ ∈ {10, 50, 100}.

Logistic regression consistently minimizes cost regardless of the FN/FP cost ratio:
- At λ=10: LR costs 252 compared to 464 for Deep AE (1.84× improvement)
- At λ=100: LR costs 2,502 compared to 3,974 for Deep AE (1.59× improvement)

The autoencoders' higher false negative counts (due to lower recall at reasonable precision) and higher false positive counts (due to poor precision) result in uniformly worse costs. Isolation Forest (cost 4,287 at λ=100) and Hybrid (cost 5,990 at λ=100) are prohibitively expensive.

### Effect of Noise in Denoising Autoencoder

We conducted a noise ablation study for the denoising autoencoder, training models with σ ∈ {0.0, 0.01, 0.05, 0.1, 0.2}. PR-AUC improved monotonically from 0.088 at σ=0 (standard AE) to 0.202 at σ=0.2, a 2.3× improvement. ROC-AUC remained stable around 0.94 across all noise levels. The optimal noise level (σ=0.2) was used for the final DAE model. However, even with optimal noise regularization, the DAE's PR-AUC (0.146) remains far below LR's 0.744, indicating that the fundamental limitation is the unsupervised reconstruction objective rather than insufficient regularization.

### Key Findings

Our results demonstrate a valuable negative result: on this PCA-transformed tabular fraud dataset, simple logistic regression with class weighting outperforms complex deep learning approaches across all meaningful metrics. This can be attributed to two factors:

1. **Feature representation**: The PCA-transformed features are approximately linearly separable for the fraud detection task. The deep autoencoders' nonlinear transformations do not provide additional discriminative power.

2. **Objective misalignment**: Autoencoders minimize reconstruction error on normal transactions, but this objective is not aligned with the goal of separating fraud from normal transactions. Logistic regression directly optimizes for discriminative classification using labeled data, making it inherently better suited to this supervised task despite the class imbalance.

The Hybrid model's poor performance (worse than both standalone LR and AE) suggests that the 8D latent representation discards information critical for fraud detection, confirming that unsupervised dimensionality reduction can harm supervised performance on already well-preprocessed features.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended but not required)
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/moumit-b/EC523-Credit-Card-Fraud-Detection
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

**Train all models and generate results:**
```bash
python scripts/run_all_models.py
```

This script will:
- Load and preprocess the dataset
- Create time-ordered train/val/test splits (60/20/20)
- Standardize features using training statistics
- Train all 5 models (LR, IF, AE, DAE, Hybrid)
- Optimize F1 thresholds on validation set
- Evaluate on test set and compute all metrics
- Generate visualizations and save to `artifacts/figures/`
- Save results tables to `artifacts/tables/`

Training takes approximately 23-25 minutes on an RTX 3070 Ti GPU.

**Run noise ablation study:**
```bash
python scripts/ablation_noise.py
```

## Contributions

| Group Member | Report/Slide Contributions | Lines of Code |
|--------------|---------------------------|---------------|
| Moumit Bhattacharjee | Slides: Experience gained, Pipeline, Task, Metrics, Evaluation, Architecture, Results<br>Report: Task, Approach, Dataset, Results | ~4500 |
| Rohan Hegde | Slides: Experience gained, Pipeline, Task, Metrics, Evaluation, Architecture, Results<br>Report: Task, Approach, Dataset, Results | ~2500 |
| Emmanuel Herold | Slides: Motivation, Related Work, Future Work<br>Report: Related Work | ~1300 |

## References

1. A. Vishnu Vardhan, M.P.V.S.N.M.L. Ankitha, P. Divya Sri, M. Battula, M.V.T.S. Priyanka. "Anomaly Detection in Credit Card Transactions using Autoencoders." *IJARCCE (International Journal of Advanced Research in Computer and Communication Engineering)*, 2024.

2. Y.T. Lei, C.Q. Ma, Y.S. Ren, X.Q, Chen, S. Narayan, A.Huynh. "A distributed deep neural network model for credit card fraud detection." *ScienceDirect.com (Financial Research Letters)*, 2023.

3. M. Singh, R. Prasad, G. Michael, N.K. Kaphungkui, N. Singh. "Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection." *arXiv.org*, 2024.

4. A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, G. Bontepi. "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy." *Polimi Research Repository*, 2017.
