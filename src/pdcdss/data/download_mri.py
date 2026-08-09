"""Acquire the MRI dataset for the CRITICAL BASELINE (not the scientific core).

The MRI strand reproduces the popular 'MRI-CNN for PD' pipeline on the SAME kind
of public Kaggle dataset the field uses, then exposes its weaknesses: accuracy
that collapses once you switch from a naive slice-level split to a subject-level
split, and Grad-CAM heatmaps that attend to non-brain artefacts. The dataset's
opaque provenance is itself part of the critique and must be documented.

Ethics: public, anonymised brain-image slices analysed as secondary data (no
recruitment). Record the dataset's licence/provenance and confirm with the
supervisor for the project file.

Needs a Kaggle API token at ~/.kaggle/kaggle.json
(kaggle.com -> Account -> Create New API Token). Then:
    pip install -e ".[data]"
    python src/pdcdss/data/download_mri.py --kaggle <owner>/<dataset-slug>

GPU: train the CNN on free Colab/Kaggle GPU — not required locally.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

MRI_RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "mri"
MRI_RAW.mkdir(parents=True, exist_ok=True)


def download_kaggle(slug: str) -> None:
    kdir = Path.home() / ".kaggle"
    has_cred = (
        (kdir / "kaggle.json").exists()
        or (kdir / "access_token").exists()        # new KGAT_... token format
        or os.getenv("KAGGLE_API_TOKEN")
        or os.getenv("KAGGLE_KEY")
    )
    if not has_cred:
        raise SystemExit(
            "No Kaggle credentials found. Either:\n"
            f"  - new token: save the KGAT_... token to {kdir / 'access_token'} (or env KAGGLE_API_TOKEN), or\n"
            f"  - classic:   put kaggle.json at {kdir / 'kaggle.json'}\n"
            "  then: pip install -e \".[data]\" and re-run."
        )
    try:
        import kagglehub  # type: ignore
    except ImportError as e:
        raise SystemExit('pip install -e ".[data]"') from e
    path = kagglehub.dataset_download(slug)
    print(f"[kaggle] downloaded '{slug}' -> {path}")
    print(f"         copy/symlink the images into {MRI_RAW} for the pipeline.")
    print("         REMEMBER: record the dataset licence + provenance (part of the critique).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaggle", metavar="owner/slug", required=True,
                    help="Kaggle PD-MRI dataset slug (the critique baseline)")
    args = ap.parse_args()
    download_kaggle(args.kaggle)


if __name__ == "__main__":
    main()
