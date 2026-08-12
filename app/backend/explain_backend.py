"""Chat-model backend for the deployed grounded explanation layer (RQ3).

The explanation layer needs a small language model at request time, and which one it
reaches depends on where the service runs. ``SLM_BACKEND`` selects it:

``ollama`` (default)
    The local Ollama daemon, that is, the configuration the RQ3 evaluation measured. Used
    during development on a machine that has the model pulled.
``hf``
    The same model family served through Hugging Face Inference Providers. Used by the
    hosted deployment, where there is no local daemon and CPU-only generation would be far
    too slow to sit in a request.
``off``
    No model call; every request is answered with the deterministic grounded summary.

All three paths draw on the same retrieved guideline text, and the two model paths send the
system prompt exported by the research module, so the served explanation is constrained by
the faithfulness guardrail the evaluation describes.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections.abc import Callable

from pdcdss.explain.guidance import retrieved_block
from pdcdss.explain.slm_explain import (
    DEPLOYED_MODEL,
    OLLAMA_HOST,
    SYSTEM_PROMPT_GROUNDED,
    build_prompt,
    generate,
    strip_reasoning,
)

_LOG = logging.getLogger(__name__)

#: Deployment-only additions to the evaluated system prompt. The research module's prompt
#: is left untouched because the RQ3 results were measured under it; these lines address
#: two drifts observed in served output. The model paraphrased the neutral family name
#: "loudness (lowering the estimate)" as "reduced loudness lowers it", which reads as if a
#: known Parkinson's sign lowered the risk. And it emits em dashes, which the interface
#: style forbids.
_DEPLOYED_PROMPT_ADDENDUM = (
    "\nName each voice feature exactly as supplied in the evidence. Do not add qualifiers "
    "such as 'reduced', 'increased', 'poor' or 'improved' to a feature name: the evidence "
    "states which direction a feature moved the estimate, not how the voice sounded. Do "
    "not use em dashes."
)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
#: Instruct build of the RQ3 winner (Qwen3-4B), served by Hugging Face Inference Providers.
HF_MODEL_DEFAULT = "Qwen/Qwen3-4B-Instruct-2507"
#: Environment variables a Hugging Face token may arrive under, most specific first.
_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
#: Guideline snippets retrieved to ground one explanation.
_SNIPPETS = 3
_REQUEST_TIMEOUT = 60


def _grounded_prompt(prediction: str, risk_band: str, top_features: list[str]) -> str:
    query = f"{prediction} {risk_band} {' '.join(top_features)}"
    return build_prompt(prediction, risk_band, top_features,
                        retrieved_block(query, k=_SNIPPETS))


def _hf_token() -> str | None:
    return next((os.environ[var].strip() for var in _TOKEN_VARS if os.environ.get(var)),
                None)


def _via_inference_providers(prediction: str, risk_band: str,
                             top_features: list[str]) -> str:
    """Generate through the OpenAI-compatible Hugging Face router.

    Returns an empty string when the provider cannot be reached, which the caller reads as
    a request to fall back to the deterministic summary. The handler is narrow on purpose:
    a transport failure is the one condition that may be absorbed here, because a language
    model outage must not turn a valid prediction into a failed request.
    """
    token = _hf_token()
    if token is None:
        _LOG.warning("SLM_BACKEND=hf but no Hugging Face token is set (%s)",
                     ", ".join(_TOKEN_VARS))
        return ""

    payload = {
        "model": os.environ.get("SLM_MODEL", HF_MODEL_DEFAULT),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_GROUNDED + _DEPLOYED_PROMPT_ADDENDUM},
            {"role": "user", "content": _grounded_prompt(prediction, risk_band,
                                                         top_features)},
        ],
        "temperature": 0.2,
        "max_tokens": 320,
    }
    request = urllib.request.Request(
        HF_ROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except OSError as error:  # URLError, HTTPError and socket timeouts all derive from this
        _LOG.warning("inference provider unreachable, using the deterministic summary: %s",
                     error)
        return ""
    return strip_reasoning(body["choices"][0]["message"]["content"]).strip()


def _via_ollama(prediction: str, risk_band: str, top_features: list[str]) -> str:
    """Generate through the local Ollama daemon, exactly as the RQ3 evaluation did.

    Absorbs a transport failure for the same reason as the hosted path: the daemon is not
    running on every machine that serves the interface.
    """
    try:
        text, _ = generate(prediction, risk_band, top_features, grounded=True,
                           model=DEPLOYED_MODEL)
    except OSError as error:
        _LOG.warning("Ollama unreachable at %s, using the deterministic summary: %s",
                     OLLAMA_HOST, error)
        return ""
    return text


def _no_model(prediction: str, risk_band: str, top_features: list[str]) -> str:
    return ""


_BACKENDS: dict[str, Callable[[str, str, list[str]], str]] = {
    "hf": _via_inference_providers,
    "ollama": _via_ollama,
    "off": _no_model,
}


def _grounded_summary(prediction: str, risk_band: str, top_features: list[str]) -> str:
    """Explanation assembled from the model output and the retrieved guidance alone.

    It states the estimate, the band and the features that drove it, quotes the retrieved
    guideline text verbatim and repeats the decision-support notice, so a response produced
    without a generated paragraph is still faithful and still grounded.
    """
    drivers = f" The estimate was driven mainly by {', '.join(top_features)}." \
        if top_features else ""
    guidance = retrieved_block(f"{prediction} {risk_band}", k=1).lstrip("- ").strip()
    return (f"{prediction}, which falls in the {risk_band} risk band.{drivers} "
            "This is decision support, not a diagnosis, and a specialist should review the "
            f"result. Relevant guidance: {guidance}")


def _tidy(text: str) -> str:
    """Replace generated em and en dashes with commas.

    The prompt asks the model not to emit them, but a probabilistic instruction is not a
    guarantee; this is. Spaced dashes become a comma and a space, unspaced ones a comma,
    which reads naturally in the clause-break positions the model uses them for.
    """
    for dash in ("—", "–"):
        text = text.replace(f" {dash} ", ", ").replace(dash, ", ")
    return text.replace(" ,", ",").replace(",,", ",")


def explanation(prediction: str, risk_band: str, top_features: list[str]) -> str:
    """Plain-language, grounded explanation of one prediction."""
    name = os.environ.get("SLM_BACKEND", "ollama").strip().lower()
    backend = _BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"unknown SLM_BACKEND {name!r}; expected one of "
                         f"{', '.join(sorted(_BACKENDS))}")
    return _tidy(backend(prediction, risk_band, top_features) or _grounded_summary(
        prediction, risk_band, top_features))
