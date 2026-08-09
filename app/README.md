# app/ — LUCID-PD web application

The deployable CDSS, separate from the research code (`src/pdcdss`). The app loads the
trained artefacts in `models/` and serves them through a FastAPI backend and a React UI.

```
app/
├── backend/          FastAPI service
│   ├── main.py       routes: /health, /predict/voice, /predict/mri, /predict/combined
│   ├── schemas.py    Pydantic response models
│   └── inference.py  loads models/, extracts features, runs SHAP + Grad-CAM + the SLM
└── frontend/         React (Vite + TypeScript) UI — the three-mode interface
```

## What each mode does
- **Voice** (primary): a recording is turned into eGeMAPS features and scored by the
  deployed model, then returned with SHAP evidence, a grounded small-language-model
  explanation and a rule-based care route.
- **MRI** (research baseline): a slice is run through the image-level CNN with a Grad-CAM
  overlay, a live demonstration of the leakage critique (RQ4). It is flagged throughout as
  leakage-inflated and is not a validated detector.
- **Combined**: an illustrative decision-level combination of the two; voice and MRI are not
  paired, so it is not true multimodal fusion.

Every screen carries a "decision support only, not a diagnosis, not a medical device" notice.

## Prerequisites
- The `parkinsons_cdss` conda environment (research dependencies, including `opensmile`,
  `torch`, `shap`).
- Trained artefacts present in `models/`: `deployed_voice.joblib` and `deployed_mri.pt`.
  Rebuild them with `python -m pdcdss.speech.deploy_voice` and
  `python -m pdcdss.mri.deploy_mri`.
- An explanation backend, selected by `SLM_BACKEND`. `hf` calls Hugging Face Inference
  Providers and needs `HF_TOKEN` in the environment; `ollama` reaches a local daemon
  (`ollama pull qwen3:4b`), which is the development setting; `off` disables generation.
  None of these is required: with no backend available the service returns a deterministic
  grounded summary and every other part of the response is unchanged.

## Run the demo
```bash
# 1. backend — from the repo root, with src on the path
PYTHONPATH=src conda run -n parkinsons_cdss uvicorn app.backend.main:app --port 8000
# 2. frontend — separate terminal
cd app/frontend && npm install && npm run dev
```
Open the localhost URL Vite prints (e.g. http://localhost:5173), choose a mode, add an input
and click **Analyse**. Interactive API docs are at http://127.0.0.1:8000/docs.

For the voice mode, upload a WAV/MP3/M4A clip or record a sustained vowel; for MRI, upload an
axial slice (PNG/JPG). No uploaded audio or image is stored.
