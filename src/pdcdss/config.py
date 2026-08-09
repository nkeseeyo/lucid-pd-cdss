"""Central paths and constants for the pdcdss package.

Import these everywhere instead of hard-coding paths, so the code is portable and the
figures, tables and models land in predictable places.
"""
from __future__ import annotations

from pathlib import Path

# src/pdcdss/config.py  ->  parents[2] == project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_FIGURES: Path = RESULTS_DIR / "figures"
RESULTS_TABLES: Path = RESULTS_DIR / "tables"

# --- primary dataset (UCI #470, Sakar et al. 2019 / dataset 2018) ---
UCI470_CSV: Path = RAW_DIR / "uci470_pd_speech.csv"
# external speech benchmark (UCI #174, Little et al. 2007)
UCI174_CSV: Path = RAW_DIR / "uci174_parkinsons.csv"

# MRI critique baseline (Kaggle PD brain-MRI): folders of class-labelled slices.
# Images live under data/raw/mri/.../{PD,NonPD}/; discovery walks recursively.
MRI_DIR: Path = RAW_DIR / "mri"

# column conventions for UCI #470
GROUP_KEY: str = "id"      # subject id — ALWAYS group on this for CV (anti-leakage)
TARGET: str = "class"      # 1 = Parkinson's, 0 = healthy control

# reproducibility
RANDOM_SEED: int = 42
N_SPLITS: int = 5          # person-level GroupKFold folds

for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR,
           RESULTS_FIGURES, RESULTS_TABLES):
    _d.mkdir(parents=True, exist_ok=True)
