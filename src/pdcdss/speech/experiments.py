"""Speech experiments on UCI #470 — RQ1 (leakage demo) + RQ2 (model comparison).

Run:  python -m pdcdss.speech.experiments

Everything uses person-level StratifiedGroupKFold (group = subject `id`); the only
exception is the deliberately-wrong record-level split used for the leakage demo.
Preprocessing (StandardScaler) is fitted INSIDE each training fold via a Pipeline.

Outputs:
  results/tables/leakage_demo.csv         RQ1: record-level vs subject-level gap
  results/tables/model_comparison.csv     RQ2: per-model metrics (mean ± std over folds)
  results/figures/leakage_gap.png         RQ1 headline figure
  results/figures/model_comparison.png    RQ2 figure (ROC-AUC + balanced accuracy)
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from pdcdss.config import (  # noqa: E402
    GROUP_KEY,
    N_SPLITS,
    RANDOM_SEED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    TARGET,
    UCI174_CSV,
    UCI470_CSV,
)

SEED = RANDOM_SEED


def load_uci470() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(UCI470_CSV)
    groups = df[GROUP_KEY].to_numpy()
    y = df[TARGET].astype(int).to_numpy()
    X = df.drop(columns=[c for c in (GROUP_KEY, TARGET) if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    return X.to_numpy(dtype=float), y, groups, list(X.columns)


def load_uci174() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Oxford Parkinson's dataset (UCI #174): group = subject, target = status.

    Independent second corpus for the RQ1 replication: it shares the
    repeated-measures structure (several recordings per subject) that makes
    record-level validation leak, but uses a different, classical feature set.
    """
    df = pd.read_csv(UCI174_CSV)
    groups = df["subject"].to_numpy()
    y = df["status"].astype(int).to_numpy()
    drop = [c for c in ("name", "subject", "status") if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    return X.to_numpy(dtype=float), y, groups, list(X.columns)


def _models() -> dict[str, Pipeline]:
    def pipe(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    models = {
        "LogReg": pipe(LogisticRegression(max_iter=4000, class_weight="balanced",
                                          random_state=SEED)),
        "SVM (RBF)": pipe(SVC(probability=True, class_weight="balanced",
                              random_state=SEED)),
        "RandomForest": pipe(RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1)),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = pipe(XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.7, eval_metric="logloss", random_state=SEED, n_jobs=-1))
    except Exception:  # noqa: BLE001
        pass
    return models


def _metrics(y_true, y_pred, proba) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": spec,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def _cv_metrics(model, X, y, groups) -> pd.DataFrame:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        rows.append(_metrics(y[te], (proba >= 0.5).astype(int), proba))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- RQ1: leakage ---
def leakage_demo(X, y, groups) -> pd.DataFrame:
    """Same model, two splits: record-level (leaky) vs subject-level (honest)."""
    base = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=4000, random_state=SEED))])
    out = []
    protocols = [
        ("Record-level split (leaky)",
         StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED), None),
        ("Subject-level GroupKFold (leakage-free)",
         StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED), groups),
    ]
    for name, cv, grp in protocols:
        accs, aucs = [], []
        splits = cv.split(X, y, grp) if grp is not None else cv.split(X, y)
        for tr, te in splits:
            base.fit(X[tr], y[tr])
            proba = base.predict_proba(X[te])[:, 1]
            accs.append(float((proba >= 0.5).astype(int).__eq__(y[te]).mean()))
            aucs.append(roc_auc_score(y[te], proba))
        out.append({"protocol": name, "accuracy": np.mean(accs),
                    "accuracy_std": np.std(accs), "roc_auc": np.mean(aucs),
                    "roc_auc_std": np.std(aucs)})
    return pd.DataFrame(out)


def _plot_leakage(df: pd.DataFrame, fname: str = "leakage_gap.png",
                  title: str = "Data leakage inflates apparent performance (UCI #470)") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["accuracy"], w, yerr=df["accuracy_std"], label="Accuracy",
           color="#9A5B00", capsize=4)
    ax.bar(x + w / 2, df["roc_auc"], w, yerr=df["roc_auc_std"], label="ROC-AUC",
           color="#0C5C5E", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df["protocol"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / fname, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------ RQ2: comparison ---
def compare_models(X, y, groups) -> pd.DataFrame:
    rows = []
    for name, model in _models().items():
        m = _cv_metrics(model, X, y, groups)
        agg = {"model": name}
        for col in m.columns:
            agg[f"{col}_mean"] = m[col].mean()
            agg[f"{col}_std"] = m[col].std()
        rows.append(agg)
        print(f"  {name:14s} AUC {agg['roc_auc_mean']:.3f}±{agg['roc_auc_std']:.3f} | "
              f"bal-acc {agg['balanced_acc_mean']:.3f} | sens {agg['sensitivity_mean']:.3f} | "
              f"spec {agg['specificity_mean']:.3f}")
    return pd.DataFrame(rows)


def _plot_comparison(df: pd.DataFrame, fname: str = "model_comparison.png",
                     title: str = "Subject-level model comparison (UCI #470)") -> None:
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
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / fname, dpi=200)
    plt.close(fig)


# ----------------------------- classification report + confusion matrices ------
def _oof_predict(model, X, y, groups) -> np.ndarray:
    """Out-of-fold class predictions over the same subject-level folds (no leakage)."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=int)
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        pred[te] = (model.predict_proba(X[te])[:, 1] >= 0.5).astype(int)
    return pred


def classification_reports(X, y, groups) -> pd.DataFrame:
    """Per-class precision/recall/F1 and a confusion-matrix grid, out-of-fold.

    The 0.5-threshold sensitivity/specificity trade-off is the real story on this
    imbalanced dataset; a single accuracy or AUC number hides it.
    """
    labels = ["HC (0)", "PD (1)"]
    rows = []
    models = _models()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        pred = _oof_predict(model, X, y, groups)
        rep = classification_report(y, pred, target_names=labels,
                                    output_dict=True, zero_division=0)
        for key in (*labels, "macro avg", "weighted avg"):
            d = rep[key]
            rows.append({"model": name, "class": key, "precision": d["precision"],
                         "recall": d["recall"], "f1": d["f1-score"],
                         "support": int(d["support"])})
        ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]),
                               display_labels=labels).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{name} (out-of-fold)", fontsize=10, fontweight="bold")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle("Subject-level out-of-fold confusion matrices (UCI #470)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "confusion_matrices.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def _run(tag: str, label: str, X, y, groups) -> None:
    """Run the leakage demo and model comparison on one dataset, saving prefixed outputs."""
    print(f"\n=== {label}: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(np.unique(groups))} subjects, class balance {np.bincount(y).tolist()} ===")
    print("[RQ1] Leakage demonstration (same LogReg model, two split protocols):")
    leak = leakage_demo(X, y, groups)
    for _, r in leak.iterrows():
        print(f"  {r['protocol']:38s} acc {r['accuracy']:.3f} | AUC {r['roc_auc']:.3f}")
    leak.to_csv(RESULTS_TABLES / f"{tag}leakage_demo.csv", index=False)
    _plot_leakage(leak, f"{tag}leakage_gap.png",
                  f"Data leakage inflates apparent performance ({label})")
    print("[RQ2] Subject-level model comparison:")
    comp = compare_models(X, y, groups)
    comp.to_csv(RESULTS_TABLES / f"{tag}model_comparison.csv", index=False)
    _plot_comparison(comp, f"{tag}model_comparison.png",
                     f"Subject-level model comparison ({label})")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["uci470", "uci174", "both"], default="uci470")
    args = ap.parse_args()

    if args.dataset in ("uci470", "both"):
        X, y, groups, _ = load_uci470()
        _run("", "UCI #470", X, y, groups)
        print("[RQ2] Per-class classification report + confusion matrices:")
        rep = classification_reports(X, y, groups)
        rep.to_csv(RESULTS_TABLES / "classification_report.csv", index=False)
        for _, r in rep[rep["class"].isin(["HC (0)", "PD (1)"])].iterrows():
            print(f"  {r['model']:14s} {r['class']:8s} "
                  f"P {r['precision']:.3f} | R {r['recall']:.3f} | F1 {r['f1']:.3f} "
                  f"(n={r['support']})")

    if args.dataset in ("uci174", "both"):
        X, y, groups, _ = load_uci174()
        _run("uci174_", "UCI #174", X, y, groups)

    print(f"\nSaved tables -> {RESULTS_TABLES}\nSaved figures -> {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
