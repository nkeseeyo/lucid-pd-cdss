"""SLM (Small Language Model) explanation layer.

Turns {prediction, risk band, top SHAP features} into plain, faithful language,
grounded by single-pass RAG over a tiny NICE/NHS corpus.

RESEARCH (RQ3): compare a small set of current on-device SLMs on the explanation
task (faithfulness / readability / consistency / latency), under a grounded prompt
(retrieved NICE/NHS guidance + faithfulness guardrail) versus an ungrounded baseline,
then DEPLOY the single winner. The comparison is research; the deployment is one model
(no runtime fallback).

NO FINE-TUNING. Models are used prompted/zero-shot with few-shot + RAG grounding +
a faithfulness guardrail (the SLM may only reference the provided features and the
retrieved guideline text). This is a deliberate differentiator vs fine-tuned-GPT-4
report generators (e.g. Sci Rep s41598-025-22448-7): cheaper, safer (no hallucinated
clinical facts), and fully reproducible on a local model.

Backend: Ollama (local, free, offline, quantised GGUF). Models run on the RTX 5060 Ti.
    install https://ollama.com  ->  `ollama pull gemma4:e4b phi4-mini qwen3:4b`  ->  :11434
"""
from __future__ import annotations

# RQ6 comparison set (ollama tag -> note). All on-device class, vendor-diverse.
CANDIDATES: dict[str, str] = {
    "gemma4:e4b": "Google Gemma 4 E4B — on-device/edge, safety-tuned (current default)",
    "phi4-mini":  "Microsoft Phi-4-mini (3.8B) — strong small reasoner",
    "qwen3:4b":   "Alibaba Qwen3-4B — strong multilingual small model",
}

# The deployed model = the RQ3 comparison winner: Qwen3-4B gave perfect grounded
# faithfulness (1.00), zero unsupported claims and the most readable output (FK ~11).
DEPLOYED_MODEL = "qwen3:4b"
OLLAMA_HOST = "http://localhost:11434"

# Grounded condition: retrieval + an explicit faithfulness guardrail.
_SYSTEM_PROMPT = (
    "You are a careful clinical-decision-support assistant. Rephrase the provided "
    "structured result into one short, plain-English paragraph (reading age ~11-13). "
    "Use ONLY the given prediction, risk band, listed features, and the retrieved "
    "guideline text. Do NOT add new clinical claims, numbers, or advice. State clearly "
    "that this is decision support, not a diagnosis. "
    "Output only the paragraph itself, with no preamble, headings, lists or notes."
)

# Ungrounded baseline: a naive 'just explain it' prompt with no retrieval and no
# guardrail, representing typical unconstrained LLM use. This is the RQ3 contrast.
_SYSTEM_PROMPT_UNGROUNDED = (
    "You are a helpful medical assistant. Explain the following Parkinson's screening "
    "result to the patient in one short, plain-English paragraph. "
    "Output only the paragraph itself, with no preamble, headings, lists or notes."
)

import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def _strip_think(text: str) -> str:
    """Remove any reasoning that a model inlines into its answer (e.g. qwen3 emits a
    <think>...</think> block in content when its hidden channel is off)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def build_prompt(prediction: str, risk_band: str, top_features: list[str],
                 retrieved: str) -> str:
    """Grounded user prompt: structured result + verbatim retrieved guideline text."""
    feats = ", ".join(top_features) if top_features else "none provided"
    return (
        f"Prediction: {prediction}\nRisk band: {risk_band}\n"
        f"Top contributing voice features (from SHAP): {feats}\n"
        f"Guideline context (use verbatim, do not extend):\n{retrieved}\n\n"
        "Write the explanation now."
    )


def build_prompt_ungrounded(prediction: str, risk_band: str,
                            top_features: list[str]) -> str:
    """Ungrounded user prompt: same structured result, but no retrieved guidance."""
    feats = ", ".join(top_features) if top_features else "none provided"
    return (
        f"Prediction: {prediction}\nRisk band: {risk_band}\n"
        f"Top contributing voice features: {feats}\n\n"
        "Write the explanation now."
    )


def _chat(model: str, system: str, user: str, *, temperature: float = 0.2,
          seed: int | None = None, num_predict: int = 320,
          timeout: int = 240, think: bool | None = None) -> tuple[str, float]:
    """Low-level Ollama /api/chat call. Returns (text, latency_seconds).

    Pass think=False to disable the hidden reasoning pass on thinking-capable models
    (e.g. the gemma4 judge), so the answer lands in `content` directly rather than being
    consumed by the thinking budget.
    """
    options: dict = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    if think is not None:
        payload["think"] = think

    def _post(p: dict) -> dict:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/chat",
            data=json.dumps(p).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    t0 = time.perf_counter()
    try:
        data = _post(payload)
    except urllib.error.HTTPError as e:
        # non-thinking models (e.g. phi4-mini) reject the think flag with HTTP 400
        if e.code == 400 and "think" in payload:
            payload.pop("think")
            data = _post(payload)
        else:
            raise
    return data["message"]["content"].strip(), time.perf_counter() - t0


def generate(prediction: str, risk_band: str, top_features: list[str], *,
             grounded: bool, model: str = DEPLOYED_MODEL, temperature: float = 0.2,
             seed: int | None = None) -> tuple[str, float]:
    """Generate one explanation under the grounded or ungrounded condition.

    Returns (text, latency_seconds). The grounded condition retrieves guideline
    snippets relevant to the case and applies the faithfulness guardrail; the
    ungrounded condition uses a naive prompt with no retrieval.
    """
    if grounded:
        from pdcdss.explain.guidance import retrieved_block
        query = f"{prediction} {risk_band} {' '.join(top_features)}"
        retrieved = retrieved_block(query, k=3)
        user = build_prompt(prediction, risk_band, top_features, retrieved)
        system = _SYSTEM_PROMPT
    else:
        user = build_prompt_ungrounded(prediction, risk_band, top_features)
        system = _SYSTEM_PROMPT_UNGROUNDED
    # think=True routes any reasoning to the model's hidden channel so `content` is the
    # answer only and complete (the candidates include reasoning models). A small budget
    # can be exhausted by thinking, leaving empty content, so retry with a larger one.
    text, lat = "", 0.0
    for budget in (1024, 2048):
        text, lat = _chat(model, system, user, temperature=temperature, seed=seed,
                          think=True, num_predict=budget)
        text = _strip_think(text)
        if text:
            break
    return text, lat


def explain(prediction: str, risk_band: str, top_features: list[str],
            retrieved: str | None = None, model: str = DEPLOYED_MODEL) -> str:
    """Public API used by the web app: grounded plain-language explanation (text only)."""
    text, _ = generate(prediction, risk_band, top_features, grounded=True, model=model)
    return text


#: Exported for the deployment layer (app/backend/explain_backend.py), which reaches the
#: same model through a hosted provider. Sharing these rather than restating them keeps the
#: served prompt identical to the one the RQ3 evaluation measured.
SYSTEM_PROMPT_GROUNDED = _SYSTEM_PROMPT
strip_reasoning = _strip_think
