"""Real inference for the deployed voice model.

Audio (an uploaded recording) -> eGeMAPS features -> XGBoost -> probability, risk band,
SHAP evidence, a grounded small-language-model explanation and a rule-based
recommendation. The model artefact is produced by ``pdcdss.speech.deploy_voice``; the
chat model behind the explanation is selected in ``explain_backend``.

The MRI strand is a research critique rather than a validated detector: its prediction and
Grad-CAM map are returned only to make the leakage argument visible, and the combined mode
averages two unpaired estimates for illustration.
"""
from __future__ import annotations

import base64
import functools
import io
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np

from pdcdss.config import MODELS_DIR
from pdcdss.recommend.rules import recommend
from pdcdss.speech.audio_features import extract_file

from .explain_backend import explanation
from .schemas import CombinedResult, Explanation, Feature, Recommendation

_MODEL_PATH = MODELS_DIR / "deployed_voice.joblib"
_MRI_PATH = MODELS_DIR / "deployed_mri.pt"

#: Sample rate and channel count openSMILE expects for the eGeMAPS functionals.
_TARGET_RATE = 16_000
_TARGET_CHANNELS = 1

#: Risk band boundaries on the predicted probability, shared by every modality.
_MODERATE_FROM, _HIGH_FROM = 0.40, 0.70

#: Number of distinct feature families reported back to the interface.
_TOP_FEATURES = 5

# eGeMAPS feature name -> plain-language family (first matching substring wins)
_PLAIN: list[tuple[str, str]] = [
    ("jitter", "pitch stability (jitter)"),
    ("shimmer", "loudness stability (shimmer)"),
    ("hnr", "voice clarity (harmonics-to-noise)"),
    ("formant", "vowel resonance (formants)"),
    ("f1", "vowel resonance (formants)"),
    ("f2", "vowel resonance (formants)"),
    ("f3", "vowel resonance (formants)"),
    ("loudness", "loudness"),
    ("f0", "pitch level and variation"),
    ("semitone", "pitch level and variation"),
    ("pitch", "pitch level and variation"),
    ("mfcc", "articulation and spectral shape"),
    ("spectral", "articulation and spectral shape"),
    ("alpha", "articulation and spectral shape"),
    ("hammarberg", "articulation and spectral shape"),
    ("slope", "articulation and spectral shape"),
    ("voiced", "speech timing and rhythm"),
    ("segment", "speech timing and rhythm"),
]


class AudioDecodeError(RuntimeError):
    """An uploaded recording could not be converted to the WAV form openSMILE reads."""


def _family(name: str) -> str:
    lowered = name.lower()
    for key, family in _PLAIN:
        if key in lowered:
            return family
    return "other acoustic measures"


def _band(probability: float) -> str:
    if probability < _MODERATE_FROM:
        return "low"
    return "moderate" if probability < _HIGH_FROM else "high"


@functools.lru_cache(maxsize=1)
def _load_voice() -> tuple[object, list[str]]:
    artefact = joblib.load(_MODEL_PATH)
    return artefact["pipeline"], list(artefact["features"])


@contextmanager
def _as_wav(path: Path) -> Iterator[Path]:
    """Yield a WAV rendering of ``path``, transcoding with ffmpeg when necessary.

    Browser captures arrive as WebM/Opus, which openSMILE cannot read, so anything that is
    not already a WAV is converted to mono 16 kHz. The converted copy lives in a temporary
    directory that is removed when the block exits.
    """
    if path.suffix.lower() == ".wav":
        yield path
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioDecodeError(
            f"decoding a {path.suffix or 'headerless'} recording needs ffmpeg, "
            "which is not installed on this host")
    with tempfile.TemporaryDirectory(prefix="lucidpd-") as workdir:
        wav = Path(workdir) / "converted.wav"
        # -hide_banner and the error log level keep stderr to the decode failure itself,
        # so the message that reaches the client describes what went wrong.
        result = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-ac", str(_TARGET_CHANNELS), "-ar", str(_TARGET_RATE), str(wav)],
            capture_output=True, text=True, check=False)
        if result.returncode != 0 or not wav.exists():
            raise AudioDecodeError(
                f"the recording could not be decoded: {result.stderr.strip()[-300:]}")
        yield wav


def _shap_evidence(pipeline, features: list[str],
                   scaled: np.ndarray) -> tuple[list[Feature], list[str]]:
    """Rank feature families by SHAP magnitude, keeping the strongest member of each.

    Returns the bars the interface draws and the matching phrases handed to the
    explanation layer, so the text and the chart cannot describe different evidence.
    """
    import shap

    values = np.asarray(shap.TreeExplainer(pipeline["model"]).shap_values(scaled))
    values = values[0] if values.ndim > 1 else values
    largest = float(np.abs(values).max()) or 1.0

    bars: list[Feature] = []
    phrases: list[str] = []
    seen: set[str] = set()
    for index in np.argsort(np.abs(values))[::-1]:
        family = _family(features[index])
        if family in seen:
            continue
        seen.add(family)
        raises = values[index] > 0
        bars.append(Feature(name=family, weight=round(abs(values[index]) / largest, 3),
                            dir="up" if raises else "down"))
        phrases.append(f"{family} ({'raising' if raises else 'lowering'} the estimate)")
        if len(bars) == _TOP_FEATURES:
            break
    return bars, phrases


def _display_pct(probability: float) -> int:
    """Render a probability as the whole percentage the interface shows.

    This must agree with the gauge in ``app/frontend/src/components/Gauge.tsx``, which
    reads the same probability and would otherwise print a different number beside this
    sentence. Two rules are shared. Halves round up, because Python's ``round`` rounds
    them to even and JavaScript's ``Math.round`` does not, so 0.005 would print as 0% here
    and 1% on the gauge. And the result is held inside 1-99, because a screening estimate
    should never assert absolute certainty.
    """
    return min(99, max(1, int(100 * probability + 0.5)))


def _as_recommendation(probability: float) -> Recommendation:
    rule = recommend(probability)
    return Recommendation(band=rule.band, route=rule.route,
                          specialist_type=rule.specialist_type,
                          secondary=rule.secondary, disclaimer=rule.disclaimer)


def predict_voice(audio_path: str | Path) -> CombinedResult:
    """Score one recording: eGeMAPS features, deployed model, SHAP evidence, explanation."""
    pipeline, features = _load_voice()
    with _as_wav(Path(audio_path)) as wav:
        row = extract_file(wav)
    x = row.reindex(columns=features).to_numpy(dtype=float)
    probability = float(pipeline.predict_proba(x)[0, 1])
    band = _band(probability)

    bars, phrases = _shap_evidence(pipeline, features, pipeline["scaler"].transform(x))
    prediction = (f"Estimated probability of Parkinson's disease: "
                  f"{_display_pct(probability)}%")
    text = explanation(prediction, band, phrases[:3])

    return CombinedResult(
        modality="voice", probability=round(probability, 3), risk_band=band,
        explanation=Explanation(features=bars, plain_text=text, method="SHAP"),
        recommendation=_as_recommendation(probability),
        caveat="Screening estimate from the deployed voice model; decision support only.",
    )


@functools.lru_cache(maxsize=1)
def _load_mri():
    import torch

    from pdcdss.mri.leakage import DEVICE, build_model

    artefact = torch.load(_MRI_PATH, map_location=DEVICE)
    model = build_model(artefact["backbone"]).to(DEVICE)
    model.load_state_dict(artefact["state_dict"])
    model.eval()
    return model, artefact["backbone"]


def _gradcam_overlay(image, cam) -> str:
    """Encode the Grad-CAM heat map over the greyscale slice as a data URL."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(3.4, 3.4))
    axes.imshow(image.convert("L"), cmap="gray")
    axes.imshow(cam, cmap="jet", alpha=0.45)
    axes.axis("off")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def predict_mri(image_path: str | Path) -> CombinedResult:
    """Research-baseline MRI critique: real CNN prediction plus a Grad-CAM overlay.

    The model is the over-confident image-level network from RQ4. Its output is returned
    only to demonstrate the critique, flagged as leakage-inflated and not clinical.
    """
    import torch
    from PIL import Image

    from pdcdss.mri.leakage import DEVICE, _gradcam, _target_layer, _tf

    model, backbone = _load_mri()
    transform = _tf(False)
    image, cam, _ = _gradcam(model, _target_layer(model, backbone), transform,
                             str(image_path))
    with torch.no_grad():
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        probability = float(torch.softmax(model(x), 1)[0, 1])

    text = ("This is a research baseline, not a validated detector. The brain-MRI model returns "
            "a prediction and a Grad-CAM map of where it looked, but its accuracy was shown to be "
            "a leakage and scanner-protocol artefact, so it must not guide care. The highlighted "
            "regions often fall outside the brain, which is the point of the critique.")
    return CombinedResult(
        modality="mri", probability=round(probability, 3), risk_band=_band(probability),
        explanation=Explanation(features=[], plain_text=text, method="Grad-CAM"),
        recommendation=_as_recommendation(probability),
        caveat="Leakage-inflated research baseline (RQ4): not a validated detector, not clinical.",
        gradcam=_gradcam_overlay(image, cam),
    )


def predict_combined(audio_path: str | Path, image_path: str | Path) -> CombinedResult:
    """Illustrative decision-level combination of the voice and MRI estimates."""
    voice = predict_voice(audio_path)
    mri = predict_mri(image_path)
    probability = round((voice.probability + mri.probability) / 2, 3)
    text = ("Illustrative decision-level combination of the voice and MRI estimates. The two "
            "inputs are not from the same person and are not paired, so this is not a true "
            "multimodal diagnosis; the MRI component is a leakage-inflated research baseline.")
    return CombinedResult(
        modality="combined", probability=probability, risk_band=_band(probability),
        explanation=Explanation(features=voice.explanation.features, plain_text=text,
                                method="SHAP + Grad-CAM"),
        recommendation=_as_recommendation(probability),
        caveat="Illustrative only: voice and MRI are not paired; the MRI component is a "
               "leakage-inflated research baseline.",
        gradcam=mri.gradcam,
    )
