"""Train and serialise the MRI model used by the app's MRI critique demo.

This deliberately serialises the IMAGE-LEVEL model, the over-confident one whose
near-perfect accuracy was shown (RQ4) to be a leakage and acquisition-protocol artefact.
The app uses it only as a live demonstration of that critique: it returns a prediction and
a Grad-CAM map (showing the model attending to non-anatomical regions), loudly labelled as
a leakage-inflated research baseline that must not guide care.

Run:  python -m pdcdss.mri.deploy_mri
Output: models/deployed_mri.pt   {backbone, state_dict}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdcdss.config import MODELS_DIR, MRI_DIR
from pdcdss.mri.dataset import list_images
from pdcdss.mri.leakage import _splits, _train_eval

BACKBONE = "resnet50"
OUT = MODELS_DIR / "deployed_mri.pt"


def main() -> None:
    rows = list_images(MRI_DIR)
    if not rows:
        raise SystemExit(f"No labelled images under {MRI_DIR}")
    df = pd.DataFrame(rows)
    paths = df["path"].tolist()
    labels = df["label"].tolist()
    groups = df["seq"].tolist()
    idx = np.arange(len(df))
    tr, va, te = _splits(idx, labels, groups)["Image-level (leaky)"]

    print(f"Training image-level {BACKBONE} on {len(df)} slices "
          f"(class balance {np.bincount(labels).tolist()})")
    metrics, model = _train_eval(BACKBONE, tr, va, te, paths, labels, max_epochs=20,
                                 bs=32, lr=1e-3, patience=5, desc="deploy mri",
                                 finetune="head")
    print(f"image-level test: acc {metrics['accuracy']:.3f} | AUC {metrics['roc_auc']:.3f} "
          f"| sens {metrics['sensitivity']:.3f} | spec {metrics['specificity']:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": BACKBONE, "state_dict": model.state_dict()}, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
