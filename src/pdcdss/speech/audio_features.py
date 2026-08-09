"""Reproducible acoustic feature extraction for the DEPLOYED app voice model.

The UCI #470 research model is features-in (proprietary TQWT) and cannot score live
mic audio. The deployed NeuroVox voice model instead uses **eGeMAPS (~88 features)**
computed the SAME way on (a) the IPVS training corpus and (b) live microphone input,
so the recorder works end-to-end: record sustained vowel -> eGeMAPS -> model.

Extractor: openSMILE eGeMAPSv02 (functionals) via the `opensmile` Python package;
`parselmouth` (Praat) is the helper for basic jitter/shimmer/HNR/F0 sanity checks.
Install: pip install -e ".[audio]"

This module is shared by training (IPVS) and the FastAPI inference path.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

# eGeMAPS v02 functionals = 88 features (the standard compact affective/clinical set)
FEATURE_SET = "eGeMAPSv02"
FEATURE_LEVEL = "Functionals"


@functools.lru_cache(maxsize=1)
def _smile():
    """The openSMILE extractor, built on first use and reused afterwards.

    The import is deferred so that loading this module stays cheap, and the configured
    extractor is cached because building one per recording would dominate the cost of
    serving a request.
    """
    import opensmile  # type: ignore

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet[FEATURE_SET],
        feature_level=opensmile.FeatureLevel[FEATURE_LEVEL],
    )


def extract_file(wav_path: str | Path) -> pd.DataFrame:
    """eGeMAPS feature row for a single WAV (one sustained-vowel recording)."""
    return _smile().process_file(str(wav_path))


def extract_signal(signal, sampling_rate: int) -> pd.DataFrame:
    """eGeMAPS feature row for an in-memory waveform (live mic capture)."""
    return _smile().process_signal(signal, sampling_rate)


def extract_folder(folder: str | Path, label: int | None = None) -> pd.DataFrame:
    """eGeMAPS for every WAV under `folder` (recurses); optional class `label`.

    NOTE: keep a subject id per file (from filename/metadata) so the IPVS model is
    also evaluated with subject-level GroupKFold — same anti-leakage rule as UCI #470.
    """
    rows = []
    for wav in sorted(Path(folder).rglob("*.wav")):
        feats = extract_file(wav)
        feats.insert(0, "file", wav.name)
        if label is not None:
            feats["label"] = label
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
