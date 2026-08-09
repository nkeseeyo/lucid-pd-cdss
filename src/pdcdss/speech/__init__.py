"""Speech pipeline — the scientific core (leakage-free, laptop-friendly).

Planned modules (build in this order; see PROJECT_PLAN.md weeks 1-3):
  eda.py             — class balance, feature families, PCA/t-SNE, leakage demo.
  cross_validation.py — person-level GroupKFold harness (the anti-leakage core).
  preprocess.py      — in-fold scaling, NZV/correlation pruning, MI selection,
                       class-imbalance handling (class-weight / SMOTE).
  models.py          — LogReg, SVM, RandomForest, XGBoost/LightGBM (+ small NN
                       variants: BN, residual, LSTM — to mirror the literature).
  fusion.py          — multi-view fusion of UCI #470 feature families
                       (baseline jitter/shimmer vs MFCC vs TQWT) — paired, honest.
  evaluate.py        — accuracy, balanced-acc, sensitivity, specificity, F1,
                       ROC-AUC, PR-AUC, MCC; confusion/ROC/calibration curves.
  generalize.py      — cross-dataset test (train UCI #470, test UCI #174 etc.).
"""
