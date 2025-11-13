# Credit Card Fraud Detection with Deep Autoencoders

## Group Members:
Moumit Bhattacharjee - moumitb@bu.edu

Rohan Hegde - rohanh@bu.edu

Emmanuel Herold - ebherold@bu.edu

## Task
Detect fraudulent credit-card transactions in highly imbalanced, time-ordered data via unsupervised/semi-supervised anomaly detection: train a deep autoencoder on legitimate transactions and flag high reconstruction error as fraud. Key challenges: extreme imbalance (~0.17% fraud), concept drift, and asymmetric costs.

## Approach
Train AE and denoising AE on legitimate transactions (PyTorch). Calibrate anomaly threshold with a cost function (λ·FN + FP) and evaluate on time-ordered splits. Compare to Isolation Forest and logistic-regression baselines.

## Dataset & metrics
Public European Credit Card Fraud dataset; create time-ordered train/val/test splits (no leakage), standardize V1–V28, Amount, Time. Primary metric: PR-AUC; also report ROC-AUC and Recall@90% Precision (threshold set on validation).

## Baselines
Isolation Forest, class-weighted Logistic Regression (and optionally XGBoost).


