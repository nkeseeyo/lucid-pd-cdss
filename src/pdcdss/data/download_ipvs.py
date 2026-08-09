"""Download IPVS (Italian Parkinson's Voice and Speech) from the public HF mirror.

Dimauro & Girardi (2019), CC BY 4.0 — raw audio (sustained vowels + reading) from
28 people with Parkinson's + 37 healthy controls, organised as
``italian_parkinson/<group>/<person>/*.wav``. The group folder gives the label
(PD vs healthy), the person folder is the subject id (for subject-level GroupKFold).

This is the corpus for the DEPLOYED voice model (eGeMAPS + classical ML), so the
live recorder works end-to-end. UCI #470 stays the research benchmark.

    pip install -e ".[audio,data]"   # needs huggingface_hub
    python src/pdcdss/data/download_ipvs.py
"""
from __future__ import annotations

from pathlib import Path

DEST = Path(__file__).resolve().parents[3] / "data" / "raw" / "ipvs"
REPO = "birgermoell/Italian_Parkinsons_Voice_and_Speech"


def main() -> None:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=str(DEST),
        allow_patterns=["italian_parkinson/**", "README.md"],
        ignore_patterns=["*.zip"],  # skip the bundled zip (duplicate of the tree)
    )
    wavs = list(Path(path).rglob("*.wav"))
    print(f"downloaded -> {path}\n{len(wavs)} wav files")


if __name__ == "__main__":
    main()
