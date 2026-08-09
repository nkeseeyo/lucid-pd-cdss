# Data

All datasets are **public, de-identified, secondary** data — no data are collected
from people, and no identifiable information is handled (consistent with ethics
approval **P194723**). Re-downloadable with the scripts in `src/pdcdss/data/`.

Layout: `raw/` (as downloaded), `processed/` (model-ready), `external/` (extra corpora).
`raw/`, `processed/`, `external/` are git-ignored — re-create with the scripts.

## Speech (the scientific core)

| File | Dataset | Subjects | Rows | Role | Status |
|---|---|---|---|---|---|
| `raw/uci470_pd_speech.csv` | UCI #470 PD Classification (Sakar et al., 2019) | 252 (188 PD / 64 HC) | 756 (3/subj) | **PRIMARY** | ✅ present |
| `raw/uci174_parkinsons.csv` | UCI #174 Parkinsons (Little et al., 2007) | 32 (24 PD / 8 HC) | 195 | **EXTERNAL benchmark** | ✅ downloaded |
| `raw/uci301/` | UCI #301 multi-sound (Sakar et al., 2013) | 40 | 1040 | optional | on demand |
| `external/neurovoz/` | NeuroVoz (Spanish, 2024) | 108 | — | external (stretch) | ⏳ **DUA-gated** |

- **Grouping keys (critical):** UCI #470 → `id`; UCI #174 → derived `subject` (parsed from
  `name`). Always split cross-validation on these so no speaker leaks across folds.
- **Targets:** UCI #470 → `class` (1=PD); UCI #174 → `status` (1=PD).
- **Cross-dataset caveat:** the corpora use *different* feature pipelines (UCI #470 is
  ~95% TQWT/MFCC). A true cross-dataset test needs a **shared modest feature set
  (e.g. eGeMAPS) recomputed from raw audio** on the comparable (sustained-vowel) task —
  so the full-feature model and the cross-dataset model are two different models.
- **NeuroVoz** is **request-only (DUA, CC BY-NC-ND)** — not a one-click download.
  Request in week 1, treat as best-effort; UCI #174 is the *planned* external set.

Download:
```bash
python src/pdcdss/data/download_speech.py --dataset 470   # primary (needs 7-Zip for the RAR)
python src/pdcdss/data/download_speech.py --dataset 174   # external benchmark
```

## MRI (critique baseline — not the scientific core)

Used to **reproduce and expose** the popular MRI-CNN pipeline's weaknesses
(accuracy that collapses under subject-level splitting; Grad-CAM artefact attention),
never as a validated PD-detection claim.

- **Source:** a public Kaggle "Parkinson's MRI" image-classification dataset — the
  same kind the field uses. Its opaque provenance is *itself part of the critique*;
  record the licence/provenance and confirm with the supervisor.

```bash
pip install -e ".[data]"                                   # kagglehub (needs kaggle.json)
python src/pdcdss/data/download_mri.py --kaggle owner/slug
```

> No public dataset pairs MRI with voice from the **same** subjects, so any
> "voice+MRI" fusion is unpaired. The "combined" app mode is therefore an explicit
> decision-level illustration, not trained joint fusion. See the MRI section of the
> [project README](../README.md#the-mri-baseline-does-not-survive-a-scan-level-split).
