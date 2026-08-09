"""Discovery, labelling and scan-grouping for the MRI critique baseline.

The Kaggle brain-MRI set is folders of class-labelled 2D slices with no subject
IDs. The only recoverable grouping is the acquisition-sequence prefix embedded in
each filename (e.g. ``Ax_T1_SE_003.png`` -> sequence ``Ax_T1_SE``), which stands in
for a scan/subject so a leakage-free split can keep all slices of one scan together.
Duplicate markers (``_copy1``, `` (1)``) and the trailing slice index are stripped.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pdcdss.config import MRI_DIR

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def label_from_path(p: Path) -> int | None:
    """1 = Parkinson's, 0 = control. NonPD is checked first (it contains 'pd')."""
    parts = {x.lower() for x in p.parts}
    if {"nonpd", "non-pd", "normal", "healthy", "control", "hc"} & parts:
        return 0
    if "pd" in parts or any("parkinson" in x for x in parts):
        return 1
    return None


def sequence_group(name: str) -> str:
    """Filename -> acquisition-sequence prefix (drops slice index + duplicate marks)."""
    s = Path(name).stem
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)                 # " (1)", " (2)"
    s = re.sub(r"[ _-]*copy ?\d*$", "", s, flags=re.I)   # "_copy1", " - Copy"
    s = re.sub(r"[ _-]+\d+$", "", s)                     # trailing slice number
    return s.strip() or Path(name).stem


def list_images(root: Path = MRI_DIR) -> list[dict]:
    """Walk ``root`` for labelled slices; return [{path, label, seq}, ...]."""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            lab = label_from_path(p)
            if lab is None:
                continue
            rows.append({"path": str(p), "label": lab, "seq": sequence_group(p.name)})
    return rows


def file_hash(path: str | Path) -> str:
    """MD5 of file bytes — catches exact-duplicate slices regardless of filename."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
