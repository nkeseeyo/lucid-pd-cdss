"""MRI dataset audit (RQ4, Stage 0) — why the headline accuracy is untrustworthy.

Run:  python -m pdcdss.mri.audit      (no GPU / no torch required)

Quantifies, before any modelling: class balance, the number of distinct acquisition
sequences (the effective number of 'scans'), exact-duplicate slices (which would
cross a naive random split), and the protocol confound --- whether the two classes
can be told apart by MRI acquisition sequence alone rather than by pathology.
Outputs results/tables/mri_audit_*.csv and results/figures/mri_audit_sequences.png.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdcdss.config import MRI_DIR, RESULTS_FIGURES, RESULTS_TABLES  # noqa: E402
from pdcdss.mri.dataset import file_hash, list_images  # noqa: E402

TEAL, GOLD = "#0C5C5E", "#9A5B00"


def main() -> None:
    rows = list_images(MRI_DIR)
    if not rows:
        raise SystemExit(f"No labelled images found under {MRI_DIR}")
    df = pd.DataFrame(rows)
    df["class"] = df["label"].map({0: "NonPD", 1: "PD"})

    n = len(df)
    by_class = df["class"].value_counts()

    # exact duplicates by content hash
    df["hash"] = [file_hash(p) for p in df["path"]]
    counts = df["hash"].value_counts()
    n_unique = int(df["hash"].nunique())
    n_dup_files = int((df["hash"].map(counts) > 1).sum())

    # acquisition sequences = recoverable 'scans'
    n_seq = int(df["seq"].nunique())
    seq_per_class = df.groupby("class")["seq"].nunique()
    seq_class = (df.groupby(["seq", "class"]).size().unstack(fill_value=0)
                 .reindex(columns=["NonPD", "PD"], fill_value=0))
    shared = int(((seq_class > 0).sum(axis=1) > 1).sum())     # sequences in BOTH classes
    confound = 100.0 * (n_seq - shared) / n_seq if n_seq else float("nan")

    print(f"MRI audit: {n} images, {n_unique} unique by content "
          f"({n_dup_files} duplicate files)")
    print(f"  by class: {dict(by_class)}")
    print(f"  distinct acquisition sequences ('scans'): {n_seq} {dict(seq_per_class)}")
    print(f"  sequences shared across classes: {shared} "
          f"-> {confound:.0f}% are class-exclusive (protocol confound)")

    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    seq_class.to_csv(RESULTS_TABLES / "mri_audit_sequences.csv")
    pd.DataFrame({
        "metric": ["images", "unique_by_hash", "duplicate_files", "PD", "NonPD",
                   "distinct_sequences", "sequences_shared_across_classes",
                   "pct_sequences_class_exclusive"],
        "value": [n, n_unique, n_dup_files, int(by_class.get("PD", 0)),
                  int(by_class.get("NonPD", 0)), n_seq, shared, round(confound, 1)],
    }).to_csv(RESULTS_TABLES / "mri_audit_summary.csv", index=False)

    # figure: slices per acquisition sequence, coloured by class
    order = seq_class.sum(axis=1).sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.22 * len(order))))
    left = np.zeros(len(order))
    for cls, col in [("NonPD", TEAL), ("PD", GOLD)]:
        vals = seq_class.reindex(order)[cls].to_numpy()
        ax.barh(y, vals, left=left, color=col, label=cls)
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("number of slices")
    ax.set_title("Slices per acquisition sequence (Kaggle PD brain-MRI)",
                 fontweight="bold")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "mri_audit_sequences.png", dpi=200)
    plt.close(fig)
    print(f"\nsaved tables -> {RESULTS_TABLES}\nsaved figure -> "
          f"{RESULTS_FIGURES / 'mri_audit_sequences.png'}")


if __name__ == "__main__":
    main()
