"""pdcdss — Parkinson's Disease CDSS (RESEARCH code).

This package holds the *research / science* only. The deployable web application
(FastAPI backend + React/Streamlit frontend) lives separately under ``/app`` at
the repo root, and consumes the trained artefacts this package writes to
``/models`` and ``/results``.

Research sub-packages:
  * data       — dataset download + loading (UCI #470 primary, #174 external).
  * speech     — scientific core: voice features, classical ML + small NN
                 variants under person-level GroupKFold, multi-view fusion, SHAP.
  * mri        — critical BASELINE: reproduces the popular MRI-CNN pipeline to
                 expose data-leakage and weak structural-MRI signal (Grad-CAM).
  * explain    — SHAP (speech) + Grad-CAM (MRI) + faithfulness/readability eval.
  * recommend  — rule-based risk-band -> care-route to specialist *types*.
"""

__version__ = "0.1.0"
