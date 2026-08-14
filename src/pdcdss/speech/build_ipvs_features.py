"""Build the eGeMAPS feature table for the DEPLOYED voice model from IPVS audio.

Walks ``data/raw/ipvs/italian_parkinson/<group>/<person>/*.wav``, extracts openSMILE
eGeMAPSv02 functionals (~88 features) per file, and writes a tidy table with
``subject`` (person folder), ``label`` (1 = Parkinson's, 0 = control), ``group`` and
``file`` columns to ``data/processed/ipvs_egemaps.csv``.

The SAME extractor (pdcdss.speech.audio_features) runs on live microphone input in
the app, so train and inference share one pipeline. Train/eval must use subject-level
GroupKFold on ``subject`` (never split a person across folds).

    pip install -e ".[audio]"
    python src/pdcdss/data/download_ipvs.py          # get the audio first
    python -m pdcdss.speech.build_ipvs_features       # -> data/processed/ipvs_egemaps.csv
    python -m pdcdss.speech.build_ipvs_features --limit 6   # quick smoke test
"""
from __future__ import annotations

import argparse

import pandas as pd

from pdcdss.config import PROCESSED_DIR, RAW_DIR
from pdcdss.speech.audio_features import extract_file

IPVS_ROOT = RAW_DIR / "ipvs" / "italian_parkinson"
OUT = PROCESSED_DIR / "ipvs_egemaps.csv"


def _label(group_dir: str) -> int:
    """1 if the group folder names Parkinson's, else 0 (healthy control)."""
    return int("parkinson" in group_dir.lower())


def iter_records(limit: int | None = None):
    wavs = sorted(IPVS_ROOT.rglob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    for wav in wavs:
        rel = wav.relative_to(IPVS_ROOT)
        if len(rel.parts) < 3:
            continue  # expect at least <group>/<person>/<file>.wav
        # subject = the FULL folder path of the individual. The PD group both nests an
        # extra batch level (group/batch/person/file) AND reuses person names across
        # batches (e.g. two distinct "Nicola S"), so the bare folder name is not unique.
        # The full path is collision-free → one subject per real individual.
        group = rel.parts[0]
        subject = "/".join(rel.parts[:-1])
        yield wav, group, subject, _label(group)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="process only N files (smoke test)")
    args = ap.parse_args()

    if not IPVS_ROOT.exists():
        raise SystemExit(f"IPVS audio not found at {IPVS_ROOT}. Run download_ipvs.py first.")

    rows = []
    for wav, group, subject, label in iter_records(args.limit):
        try:
            feats = extract_file(wav)            # 1-row DataFrame of eGeMAPS functionals
        except Exception as e:                   # noqa: BLE001 — skip unreadable file, keep going
            print(f"  skip {wav.name}: {e}")
            continue
        feats = feats.reset_index(drop=True)
        feats.insert(0, "file", wav.name)
        feats.insert(0, "subject", subject)
        feats.insert(0, "group", group)
        feats.insert(0, "label", label)
        rows.append(feats)
        print(f"  {label} | {subject.split('/')[-1]:22s} | {wav.name}")

    if not rows:
        raise SystemExit("No features extracted.")
    df = pd.concat(rows, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    n_sub = df["subject"].nunique()
    print(f"\nwrote {OUT}\nrows: {len(df)} | subjects: {n_sub} | features: {df.shape[1]-4}")
    print("label balance (rows):", dict(df["label"].value_counts()))
    print("label balance (subjects):",
          dict(df.groupby("subject")["label"].first().value_counts()))


if __name__ == "__main__":
    main()
