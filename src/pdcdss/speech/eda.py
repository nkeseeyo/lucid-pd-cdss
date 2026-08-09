"""Exploratory data analysis for UCI #470 (Stage 0) — real numbers + figures.

Run:  python -m pdcdss.speech.eda

Produces results/figures/eda_*.png and results/tables/eda_summary.csv: class
balance (recording vs subject),
missing/constant/duplicate columns, feature-family breakdown, correlation
redundancy, recordings-per-subject, and a record-level PCA scatter.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pdcdss.config import (  # noqa: E402
    GROUP_KEY,
    RESULTS_FIGURES,
    RESULTS_TABLES,
    TARGET,
    UCI470_CSV,
)

TEAL, GOLD, NAVY = "#0C5C5E", "#9A5B00", "#123047"


def feature_family(col: str) -> str:
    """Map a UCI #470 column to its Sakar et al. (2019) feature family.

    Order matters: the wavelet-transform block and the MFCC delta/delta-delta
    features are caught explicitly so they do not fall into a vague 'Other' bucket."""
    c = col.lower()
    if c.startswith("tqwt"):
        return "TQWT (tunable-Q wavelet)"
    if c.startswith(("ea", "ed", "det_", "app_")):
        return "Wavelet transform (WT)"
    if "mfcc" in c or "delta" in c or "log_energy" in c:
        return "MFCC / cepstral"
    if c.startswith(("gq_", "gne_", "vfer_", "imf_")):
        return "Vocal fold (GQ/GNE/VFER/IMF)"
    if c.startswith(("f1", "f2", "f3", "f4", "b1", "b2", "b3", "b4")) or "intensity" in c:
        return "Formant / intensity"
    if any(k in c for k in ("jitter", "shimmer", "harmonic", "rpde", "dfa", "ppe",
                            "pulse", "period")):
        return "Baseline (jitter/shimmer/...)"
    if c == "gender":
        return "Demographic"
    return "Other"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    df = pd.read_csv(UCI470_CSV)
    feat_cols = [c for c in df.columns if c not in (GROUP_KEY, TARGET)]
    n, nsub = len(df), df[GROUP_KEY].nunique()
    rec_bal = df[TARGET].value_counts().sort_index()
    sub_bal = df.groupby(GROUP_KEY)[TARGET].first().value_counts().sort_index()

    miss = int(df[feat_cols].isna().sum().sum())
    const = int((df[feat_cols].nunique() <= 1).sum())
    dup = int(df[feat_cols].T.duplicated().sum())

    Xnum = df[feat_cols].select_dtypes(include=[np.number])
    corr = Xnum.corr().abs().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    high = int(np.nansum(corr[iu] > 0.95))
    tot = len(iu[0])

    fam = pd.Series([feature_family(c) for c in feat_cols]).value_counts()

    print(f"UCI #470: rows={n}, subjects={nsub}, features={len(feat_cols)}, "
          f"recs/subject={n / nsub:.2f}")
    print(f"recordings by class {dict(rec_bal)} | subjects by class {dict(sub_bal)}")
    print(f"missing={miss}, constant cols={const}, duplicate cols={dup}")
    print(f"feature pairs |r|>0.95: {high}/{tot} ({100 * high / tot:.1f}%)")
    print("feature families:", dict(fam))

    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

    # class balance (recording vs subject)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, bal, title in [(ax[0], rec_bal, "Recordings by class"),
                          (ax[1], sub_bal, "Subjects by class")]:
        a.bar(["HC (0)", "PD (1)"], [bal.get(0, 0), bal.get(1, 0)], color=[TEAL, GOLD])
        a.set_title(title)
        _despine(a)
    fig.suptitle("UCI #470 class balance: recording vs subject level", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "eda_class_balance.png", dpi=200)
    plt.close(fig)

    # feature families
    fig, ax = plt.subplots(figsize=(7, 4))
    fam.sort_values().plot.barh(ax=ax, color=TEAL)
    ax.set_title("Feature families (UCI #470)", fontweight="bold")
    ax.set_xlabel("number of features")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "eda_feature_families.png", dpi=200)
    plt.close(fig)

    # PCA scatter (record-level — note the leakage caveat in the write-up)
    Xs = StandardScaler().fit_transform(Xnum.to_numpy(dtype=float))
    pc = PCA(n_components=2, random_state=42).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, co, lab in [(0, TEAL, "HC"), (1, GOLD, "PD")]:
        m = df[TARGET].to_numpy() == cls
        ax.scatter(pc[m, 0], pc[m, 1], s=10, alpha=0.5, c=co, label=lab)
    ax.set_title("PCA of UCI #470 features (record-level)", fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "eda_pca.png", dpi=200)
    plt.close(fig)

    pd.DataFrame({
        "metric": ["rows", "subjects", "features", "recs_per_subject", "rec_PD",
                   "rec_HC", "subj_PD", "subj_HC", "missing", "constant_cols",
                   "duplicate_cols", "high_corr_pairs_pct"],
        "value": [n, nsub, len(feat_cols), round(n / nsub, 2), int(rec_bal.get(1, 0)),
                  int(rec_bal.get(0, 0)), int(sub_bal.get(1, 0)), int(sub_bal.get(0, 0)),
                  miss, const, dup, round(100 * high / tot, 1)],
    }).to_csv(RESULTS_TABLES / "eda_summary.csv", index=False)
    print(f"\nsaved EDA figures -> {RESULTS_FIGURES}\nsummary -> {RESULTS_TABLES / 'eda_summary.csv'}")


if __name__ == "__main__":
    main()
