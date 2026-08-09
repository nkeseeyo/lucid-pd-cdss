"""Download the public Parkinson's *speech* datasets used by this project.

All are public, de-identified, free to use for research (ethics ID P194723):

  470 : UCI #470 Parkinson's Disease Classification (Sakar et al., 2019)  [PRIMARY]
        252 subjects x 3 recordings = 756 rows, ~752 acoustic features.
        Ships a zip-containing-a-RAR; needs a RAR extractor (7-Zip) on PATH.
  174 : UCI #174 Parkinsons (Little et al., 2007)        [EXTERNAL BENCHMARK]
        31 subjects, ~197 recordings, 22 features. Direct zip, no extractor.
  301 : UCI #301 Parkinson Speech, multiple sound types (Sakar et al., 2013)
        40 subjects x 26 recordings. Best-effort raw extraction.

Usage:
    python src/pdcdss/data/download_speech.py --dataset 174
    python src/pdcdss/data/download_speech.py --dataset all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "470": "https://archive.ics.uci.edu/static/public/470/parkinson+s+disease+classification.zip",
    "174": "https://archive.ics.uci.edu/static/public/174/parkinsons.zip",
    "301": "https://archive.ics.uci.edu/static/public/301/parkinson+speech+dataset+with+multiple+types+of+sound+recordings.zip",
}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=180).read()


def _find_rar_extractor() -> str | None:
    for exe in ("7zz", "7z", "7za"):
        p = shutil.which(exe)
        if p:
            return p
    return None


# ----------------------------------------------------------------- UCI 470 ----
def download_470() -> Path:
    out = RAW / "uci470_pd_speech.csv"
    if out.exists():
        print(f"[470] already present -> {out}")
        return out
    tmp = Path(tempfile.mkdtemp(prefix="pd470_"))
    zipfile.ZipFile(BytesIO(_fetch(URLS["470"]))).extractall(tmp)
    rar = next(tmp.glob("*.rar"))
    ext = _find_rar_extractor()
    if not ext:
        print("[470] No RAR extractor (7-Zip) on PATH. Extract manually:", rar)
        return out
    subprocess.run([ext, "x", str(rar), f"-o{tmp}", "-y"], check=True,
                   stdout=subprocess.DEVNULL)
    csv = next(tmp.glob("*.csv"))
    df = pd.read_csv(csv, header=1)  # flatten the two-line header
    df.to_csv(out, index=False)
    _report("470", df, group="id", target="class")
    return out


# ----------------------------------------------------------------- UCI 174 ----
def download_174() -> Path:
    """Little et al. (2007). 'name' encodes subject: phon_R01_S01_1 -> S01."""
    out = RAW / "uci174_parkinsons.csv"
    tmp = Path(tempfile.mkdtemp(prefix="pd174_"))
    zipfile.ZipFile(BytesIO(_fetch(URLS["174"]))).extractall(tmp)
    data = next(tmp.glob("parkinsons.data"))
    df = pd.read_csv(data)  # comma-separated, has a header row
    # derive subject id from the 'name' column for leakage-free GroupKFold
    df.insert(1, "subject", df["name"].str.split("_").str[2])
    df.to_csv(out, index=False)
    _report("174", df, group="subject", target="status")
    return out


# ----------------------------------------------------------------- UCI 301 ----
def download_301() -> Path:
    """Best-effort: extract the raw txt files; parsing is documented in the zip."""
    dest = RAW / "uci301"
    dest.mkdir(exist_ok=True)
    zipfile.ZipFile(BytesIO(_fetch(URLS["301"]))).extractall(dest)
    print(f"[301] extracted raw files -> {dest}")
    print("      train_data.txt / test_data.txt are headerless; col 0 = subject id,")
    print("      then 26 features, then UPDRS + class. See the included docs to map.")
    return dest


def _report(tag: str, df: pd.DataFrame, *, group: str, target: str) -> None:
    print(f"=== UCI {tag} ===")
    print("shape:", df.shape)
    if group in df.columns:
        n = df[group].nunique()
        print(f"subjects ('{group}'): {n} | rows: {len(df)} | recs/subject: {len(df)/n:.2f}")
    if target in df.columns:
        print(f"class balance (rows, '{target}'):", dict(df[target].value_counts()))
        if group in df.columns:
            sub = df.groupby(group)[target].first().value_counts()
            print("class balance (subjects):", dict(sub))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["470", "174", "301", "all"], default="174")
    args = ap.parse_args()
    todo = ["470", "174", "301"] if args.dataset == "all" else [args.dataset]
    fns = {"470": download_470, "174": download_174, "301": download_301}
    for d in todo:
        try:
            fns[d]()
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"[{d}] FAILED: {e}")


if __name__ == "__main__":
    main()
