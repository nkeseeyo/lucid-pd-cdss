"""RQ4 ablation: fine-tuning depth vs the MRI leakage gap, with leaky-vs-honest Grad-CAM.

Trains one backbone (default ResNet50) under image-level (leaky) and scan-level
(honest) splits at three fine-tuning depths --- frozen head only, last conv block
unfrozen, and full fine-tuning. This answers two questions the single-depth study
cannot: (i) does deeper fine-tuning change the size of the leakage gap, and (ii) what
do Grad-CAM maps look like when the backbone is actually adapted to the data rather
than left as generic ImageNet features (removing the frozen-backbone caveat).

The Grad-CAM comparison uses the last-block-unfrozen models and shows the SAME slices
under the leaky and the honest model, so the two can be read side by side.

Run:
    python -m pdcdss.mri.ablation
    python -m pdcdss.mri.ablation --backbone resnet50 --epochs 30 --dedup

Outputs:
    results/tables/mri_finetune_ablation.csv    metrics per depth x split
    results/figures/mri_finetune_ablation.png   accuracy by depth and split
    results/figures/mri_gradcam_compare.png     leaky vs honest Grad-CAM (last block)
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdcdss.config import MRI_DIR, RESULTS_FIGURES, RESULTS_TABLES  # noqa: E402
from pdcdss.mri.dataset import file_hash, list_images  # noqa: E402
from pdcdss.mri.leakage import PRETTY, _splits, _train_eval, gradcam_compare  # noqa: E402

DEPTHS = ["head", "lastblock", "full"]
DEPTH_LABEL = {"head": "Frozen head", "lastblock": "Last block", "full": "Full network"}
# lower learning rate as more of the pretrained backbone is unfrozen
LR = {"head": 1e-3, "lastblock": 1e-4, "full": 1e-5}


def _plot(df: pd.DataFrame, backbone: str) -> None:
    x = np.arange(len(DEPTHS))
    w = 0.38

    def val(depth, kind):
        sub = df[(df["finetune"] == depth) & (df["protocol"].str.startswith(kind))]
        return float(sub["balanced_acc"].iloc[0]) if len(sub) else float("nan")

    leaky = [val(d, "Image") for d in DEPTHS]
    honest = [val(d, "Scan") for d in DEPTHS]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w / 2, leaky, w, label="Image-level (leaky)", color="#9A5B00")
    ax.bar(x + w / 2, honest, w, label="Scan-level (leakage-free)", color="#0C5C5E")
    ax.set_xticks(x)
    ax.set_xticklabels([DEPTH_LABEL[d] for d in DEPTHS])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Fine-tuning depth vs leakage gap ({PRETTY.get(backbone, backbone)})",
                 fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / "mri_finetune_ablation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dedup", action="store_true",
                    help="drop exact-duplicate slices before splitting")
    args = ap.parse_args()

    rows = list_images(MRI_DIR)
    if not rows:
        raise SystemExit(f"No labelled images under {MRI_DIR}")
    df = pd.DataFrame(rows)
    if args.dedup:
        df["hash"] = [file_hash(p) for p in df["path"]]
        df = df.drop_duplicates("hash").reset_index(drop=True)
    paths = df["path"].tolist()
    labels = df["label"].tolist()
    groups = df["seq"].tolist()
    idx = np.arange(len(df))
    splits = _splits(idx, labels, groups)

    print(f"MRI fine-tuning ablation ({args.backbone}): {len(df)} slices, "
          f"{df['seq'].nunique()} sequences, class balance {np.bincount(labels).tolist()}\n")

    out, cam_models = [], {}
    for depth in DEPTHS:
        for kind, (tr, va, te) in splits.items():
            tag = "leaky" if kind.startswith("Image") else "honest"
            m, model = _train_eval(args.backbone, tr, va, te, paths, labels,
                                    args.epochs, args.batch_size, LR[depth],
                                    args.patience, desc=f"{depth} {tag}", finetune=depth)
            out.append({"finetune": depth, "protocol": kind, **m})
            if depth == "lastblock":
                cam_models[tag] = model
            print(f"  {depth:10s} {kind:22s} acc {m['accuracy']:.3f} | "
                  f"AUC {m['roc_auc']:.3f} | sens {m['sensitivity']:.3f} | "
                  f"spec {m['specificity']:.3f} | epochs {m['epochs_run']}")

    res = pd.DataFrame(out)
    res.to_csv(RESULTS_TABLES / "mri_finetune_ablation.csv", index=False)
    _plot(res, args.backbone)
    if {"leaky", "honest"} <= set(cam_models):
        try:
            gradcam_compare(cam_models, args.backbone, df,
                            splits["Image-level (leaky)"][2])
        except Exception as e:  # noqa: BLE001
            print(f"[grad-cam compare skipped: {e}]")
    print(f"\nSaved -> {RESULTS_TABLES / 'mri_finetune_ablation.csv'} and {RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
