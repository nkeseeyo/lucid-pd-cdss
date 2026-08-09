"""FastAPI backend for the LUCID-PD CDSS.

Run: uvicorn app.backend.main:app --reload   ->  http://127.0.0.1:8000/docs

The voice route serves real inference: an uploaded recording is turned into eGeMAPS
features, scored by the deployed model, and returned with SHAP evidence, a grounded
small-language-model explanation and a rule-based recommendation. The MRI route runs the
real (image-level) CNN with Grad-CAM but is a research-baseline critique, flagged as
leakage-inflated and not for clinical use. The combined route is an illustrative
decision-level combination of the two. See app/backend/inference.py.

Uploads are written to a temporary directory that is removed as soon as the response is
built, so no recording or image outlives the request that carried it.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import inference
from .inference import AudioDecodeError
from .schemas import CombinedResult, Health

app = FastAPI(
    title="LUCID-PD CDSS API",
    version="0.1.0",
    description="Decision support only: not a medical device and not a diagnosis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

#: Container signatures, checked ahead of the upload name because browsers post recordings
#: under generic filenames whose extension does not describe the bytes.
_SIGNATURES: dict[bytes, str] = {
    b"RIFF": ".wav",
    b"OggS": ".ogg",
    b"fLaC": ".flac",
    b"\x1aE\xdf\xa3": ".webm",
}


def _suffix(payload: bytes, filename: str | None, default: str) -> str:
    """Extension to store an upload under: container signature first, then its name."""
    return _SIGNATURES.get(payload[:4]) or Path(filename or "").suffix or default


@contextmanager
def _staged(uploads: dict[str, tuple[bytes, str | None, str]]) -> Iterator[dict[str, Path]]:
    """Write each upload into one temporary directory and yield the paths by name."""
    with tempfile.TemporaryDirectory(prefix="lucidpd-upload-") as workdir:
        paths = {}
        for name, (payload, filename, default) in uploads.items():
            path = Path(workdir) / f"{name}{_suffix(payload, filename, default)}"
            path.write_bytes(payload)
            paths[name] = path
        yield paths


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")


@app.post("/predict/voice", response_model=CombinedResult)
async def predict_voice(file: UploadFile = File(...)) -> CombinedResult:
    """Real prediction: uploaded recording -> eGeMAPS -> deployed model -> result."""
    payload = await file.read()
    with _staged({"voice": (payload, file.filename, ".wav")}) as staged:
        return _decoded(inference.predict_voice, staged["voice"])


@app.post("/predict/mri", response_model=CombinedResult)
async def predict_mri(file: UploadFile = File(...)) -> CombinedResult:
    """Research-baseline MRI critique: real CNN prediction and Grad-CAM, flagged."""
    payload = await file.read()
    with _staged({"mri": (payload, file.filename, ".png")}) as staged:
        return inference.predict_mri(staged["mri"])


@app.post("/predict/combined", response_model=CombinedResult)
async def predict_combined(voice: UploadFile = File(...),
                           image: UploadFile = File(...)) -> CombinedResult:
    """Illustrative decision-level combination of the voice and MRI estimates."""
    uploads = {
        "voice": (await voice.read(), voice.filename, ".wav"),
        "mri": (await image.read(), image.filename, ".png"),
    }
    with _staged(uploads) as staged:
        return _decoded(inference.predict_combined, staged["voice"], staged["mri"])


def _decoded(predict, *paths: Path) -> CombinedResult:
    """Run a prediction that reads audio, reporting an unreadable recording as 415."""
    try:
        return predict(*paths)
    except AudioDecodeError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error


# The single-container deployment serves the compiled React bundle from the API itself, so
# the browser talks to one origin and no separate web host is needed. STATIC_DIR points at
# the build output; when it is unset (local development) the frontend runs on its own dev
# server and only the API routes above are exposed. The mount is declared last so that it
# never shadows a route.
_static = Path(os.environ.get("STATIC_DIR", ""))
if _static.name and _static.is_dir():
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
