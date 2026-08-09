"""MRI strand (RQ4) — the critical baseline, not a clinical claim.

Reproduces the field's brain-MRI Parkinson's pipeline on the public Kaggle slice
dataset, then shows empirically why its headline accuracies (often 95-100%, e.g.
Pandey et al. 2025) do not survive honest validation. Trained locally on the
RTX 5060 Ti (cu128 wheels); no Colab/Kaggle GPU required.

Modules:
  dataset.py  — discover/label slices and recover the acquisition-sequence
                grouping that stands in for a scan/subject (no subject IDs exist).
  audit.py    — fast, no-GPU audit: class balance, exact-duplicate slices, and the
                protocol confound (are the classes separable by MRI sequence alone?).
  leakage.py  — compare transfer-learning CNNs (VGG16, ResNet50, DenseNet121,
                EfficientNet) plus a small from-scratch CNN under an image-level
                (leaky) split vs a scan-level (honest) split, with Grad-CAM.

Ethics: public, de-identified imaging analysed as secondary data; provenance is
documented as part of the critique (see PROJECT_PLAN.md).
"""
