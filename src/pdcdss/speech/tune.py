"""Honest, nested, subject-level hyperparameter search for the speech detectors.

Run:  python -m pdcdss.speech.tune

The point of this module is to tune WITHOUT cheating. An INNER subject-level
``StratifiedGroupKFold`` drives a randomised hyperparameter search (model selection),
wrapped inside an OUTER subject-level ``StratifiedGroupKFold`` that produces the
reported metrics. The reported score is therefore always an out-of-fold score on data
the search never saw, so tuning cannot inflate the result the way a naive
``GridSearchCV(...).best_score_`` would.

Outputs:
  results/tables/tuned_comparison.csv   per-model metrics after nested tuning
  results/tables/tuned_vs_default.csv   tuned vs default (from model_comparison.csv)
  results/figures/tuned_comparison.png  tuned ROC-AUC + balanced accuracy per model
"""
from __future__ import annotations

import warnings
from math import prod

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from pdcdss.config import (  # noqa: E402
    N_SPLITS,
    RANDOM_SEED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
)
from pdcdss.speech.experiments import _metrics, load_uci470  # noqa: E402

SEED = RANDOM_SEED
N_ITER = 20          # randomised-search candidates per outer fold (capped at grid size)
INNER_SPLITS = 3     # inner subject-level folds for model selection
SCORING = "roc_auc"  # threshold-independent; the imbalance is handled by class weights


def _pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _spaces() -> dict[str, tuple[Pipeline, dict]]:
    """Each entry: (pipeline, hyperparameter grid keyed by ``model__<param>``)."""
    spaces = {
        "LogReg": (
            _pipe(LogisticRegression(max_iter=5000, class_weight="balanced",
                                     random_state=SEED)),
            {"model__C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]},
        ),
        "SVM (RBF)": (
            _pipe(SVC(kernel="rbf", probability=True, class_weight="balanced",
                      random_state=SEED)),
            {"model__C": [0.1, 1, 10, 50, 100],
             "model__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1]},
        ),
        "RandomForest": (
            _pipe(RandomForestClassifier(class_weight="balanced_subsample",
                                         random_state=SEED, n_jobs=1)),
            {"model__n_estimators": [200, 400, 800],
             "model__max_depth": [None, 5, 10, 20],
             "model__max_features": ["sqrt", "log2", 0.3],
             "model__min_samples_leaf": [1, 2, 4]},
        ),
    }
    try:
        from xgboost import XGBClassifier

        spaces["XGBoost"] = (
            _pipe(XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1)),
            {"model__n_estimators": [200, 400, 800],
             "model__max_depth": [3, 4, 6],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__subsample": [0.7, 0.9, 1.0],
             "model__colsample_bytree": [0.5, 0.7, 1.0],
             "model__reg_lambda": [1, 5, 10]},
        )
    except Exception:  # noqa: BLE001
        pass
    return spaces


def nested_cv(name: str, pipe: Pipeline, space: dict, X, y, groups) -> dict:
    """Outer subject-level CV reporting; inner subject-level CV model selection."""
    outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    n_iter = min(N_ITER, prod(len(v) for v in space.values()))
    fold_metrics = []
    for tr, te in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True,
                                     random_state=SEED)
        search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=inner,
                                    scoring=SCORING, random_state=SEED, n_jobs=-1,
                                    refit=True)
        search.fit(X[tr], y[tr], groups=groups[tr])
        best = search.best_estimator_
        proba = best.predict_proba(X[te])[:, 1]
        fold_metrics.append(_metrics(y[te], (proba >= 0.5).astype(int), proba))
    m = pd.DataFrame(fold_metrics)
    agg = {"model": name}
    for col in m.columns:
        agg[f"{col}_mean"] = m[col].mean()
        agg[f"{col}_std"] = m[col].std()
    return agg


def _plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["roc_auc_mean"], w, yerr=df["roc_auc_std"], label="ROC-AUC",
           color="#0C5C5E", capsize=4)
    ax.bar(x + w / 2, df["balanced_acc_mean"], w, yerr=df["balanced_acc_std"],
           label="Balanced accuracy", color="#2C6E9B", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Nested subject-level hyperparameter tuning (UCI #470)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "tuned_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    X, y, groups, _ = load_uci470()
    print(f"Nested subject-level tuning: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(np.unique(groups))} subjects "
          f"({N_SPLITS} outer x {INNER_SPLITS} inner folds, n_iter<={N_ITER})\n")

    rows = []
    for name, (pipe, space) in _spaces().items():
        print(f"  tuning {name} ...", flush=True)
        agg = nested_cv(name, pipe, space, X, y, groups)
        rows.append(agg)
        print(f"    {name:14s} AUC {agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f} | "
              f"bal-acc {agg['balanced_acc_mean']:.3f} | sens {agg['sensitivity_mean']:.3f} | "
              f"spec {agg['specificity_mean']:.3f} | MCC {agg['mcc_mean']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_TABLES / "tuned_comparison.csv", index=False)
    _plot(df)

    # tuned vs default, if the untuned baseline exists
    default_path = RESULTS_TABLES / "model_comparison.csv"
    if default_path.exists():
        keep = ["model", "roc_auc_mean", "balanced_acc_mean", "mcc_mean"]
        base = pd.read_csv(default_path)[keep].rename(
            columns=lambda c: c if c == "model" else f"default_{c}")
        merged = df[keep].merge(base, on="model")
        for met in ("roc_auc_mean", "balanced_acc_mean", "mcc_mean"):
            merged[f"delta_{met}"] = merged[met] - merged[f"default_{met}"]
        merged.to_csv(RESULTS_TABLES / "tuned_vs_default.csv", index=False)
        print("\nTuned vs default (positive delta = tuning helped):")
        print(merged.to_string(index=False))

    print(f"\nSaved tables -> {RESULTS_TABLES}\nSaved figures -> {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
