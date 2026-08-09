"""RQ2: SHAP feature-influence and cross-fold stability for the voice detector.

Run:  python -m pdcdss.speech.shap_analysis

Uses the XGBoost detector (the RQ2 winner) on UCI #470. Trees are scale-invariant,
so SHAP is computed on the raw named features for interpretability. Produces a global
beeswarm, a top-feature table tagged with its acoustic family, and a cross-fold
stability analysis (do the same features stay important across subject-level folds?).

Outputs:
  results/figures/shap_beeswarm.png        global feature influence (top 20)
  results/figures/shap_stability.png       per-fold mean|SHAP| for the top features
  results/tables/shap_top_features.csv      top features, family, mean|SHAP|, stability
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from pdcdss.config import N_SPLITS, RANDOM_SEED, RESULTS_FIGURES, RESULTS_TABLES  # noqa: E402
from pdcdss.speech.eda import feature_family  # noqa: E402
from pdcdss.speech.experiments import load_uci470  # noqa: E402

SEED = RANDOM_SEED
TOP_K = 15


def _xgb() -> XGBClassifier:
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.7, eval_metric="logloss",
                         random_state=SEED, n_jobs=-1)


def _shap_values(model, X) -> np.ndarray:
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):           # some shap versions return [class0, class1]
        sv = sv[1]
    return np.asarray(sv)


def main() -> None:
    X, y, groups, feats = load_uci470()
    feats = list(feats)
    fam = {f: feature_family(f) for f in feats}

    # --- global model + SHAP on the full dataset ---
    model = _xgb().fit(X, y)
    sv = _shap_values(model, X)
    global_imp = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)

    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

    # global beeswarm (top 20)
    shap.summary_plot(sv, X, feature_names=feats, max_display=20, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    fig.suptitle("Global SHAP feature influence (XGBoost, UCI #470)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- cross-fold stability: importance ranking per subject-level fold ---
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_imp = []
    for tr, te in cv.split(X, y, groups):
        m = _xgb().fit(X[tr], y[tr])
        sv_te = _shap_values(m, X[te])
        fold_imp.append(pd.Series(np.abs(sv_te).mean(axis=0), index=feats))
    imp = pd.concat(fold_imp, axis=1)
    imp.columns = [f"fold{i + 1}" for i in range(imp.shape[1])]

    # rank stability: mean pairwise Spearman of fold importance rankings
    ranks = imp.rank(ascending=False)
    corrs = [spearmanr(ranks.iloc[:, i], ranks.iloc[:, j]).correlation
             for i in range(ranks.shape[1]) for j in range(i + 1, ranks.shape[1])]
    mean_rho = float(np.mean(corrs))

    # how often each feature lands in the per-fold top-K
    in_topk = (ranks <= TOP_K).sum(axis=1)

    top = global_imp.head(TOP_K).index.tolist()
    table = pd.DataFrame({
        "feature": top,
        "family": [fam[f] for f in top],
        "mean_abs_shap": [round(float(global_imp[f]), 5) for f in top],
        "folds_in_top15": [int(in_topk[f]) for f in top],
    })
    table.to_csv(RESULTS_TABLES / "shap_top_features.csv", index=False)

    print(f"UCI #470 SHAP: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"mean pairwise cross-fold Spearman of importance ranks: {mean_rho:.3f}")
    n_stable = int((in_topk[top] == N_SPLITS).sum())
    print(f"{n_stable}/{TOP_K} top features appear in the top-{TOP_K} of all "
          f"{N_SPLITS} folds")
    print("\nTop features (mean|SHAP|, family, folds-in-top15):")
    for _, r in table.iterrows():
        print(f"  {r['feature']:32s} {r['mean_abs_shap']:.4f}  "
              f"{r['family']:28s} {r['folds_in_top15']}/{N_SPLITS}")

    # stability figure: per-fold mean|SHAP| for the global top features
    sub = imp.loc[top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(sub.to_numpy().T, vert=False, labels=sub.index, widths=0.6)
    ax.set_xlabel("mean |SHAP| per fold")
    ax.set_title(f"Cross-fold stability of the top {TOP_K} features "
                 f"(Spearman {mean_rho:.2f})", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "shap_stability.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved -> {RESULTS_TABLES / 'shap_top_features.csv'} and {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
