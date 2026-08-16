---
title: LUCID-PD Parkinson's CDSS
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice-based Parkinson's decision support with SHAP evidence
---

# LUCID-PD: a clinical decision support demonstrator for Parkinson's disease

Decision support and research tool. **Not a medical device, and not a diagnosis.** Nothing
shown here should be used to make or delay a clinical decision, and no uploaded audio or
image is stored.

This Space hosts the prototype built for an MSc dissertation on leakage-controlled
validation and dependable explanation in Parkinson's screening. The interface offers three
modes.

**Voice** is the working system. A sustained-vowel recording, uploaded or captured in the
browser, is converted to eGeMAPS acoustic features by openSMILE and scored by a gradient
boosted classifier trained on the Italian Parkinson's Voice and Speech corpus and evaluated
at subject level. The result carries the SHAP features that drove it, a plain-language
explanation grounded in retrieved NICE and NHS guidance, and a rule-based care route.

**MRI** is a research baseline that exists to be criticised. The convolutional network
behind it reaches a high accuracy that the dissertation shows to be an artefact of
image-level splitting and acquisition protocol rather than pathology. Its prediction and
Grad-CAM map are displayed so that the failure is visible: the highlighted regions
frequently fall outside the brain. It is not a detector.

**Combined** averages the two estimates for illustration. The voice and imaging corpora are
not paired at subject level, so this is not multimodal fusion and is labelled as such
throughout.

## Interpreting the voice result

The deployed voice model is evaluated at subject level, so no recording from one person
appears on both sides of a split. Its reported figure is nevertheless read with caution,
because the control and Parkinson's recordings originate in different cohorts whose age and
recording conditions plausibly inflate it. The validated results of the dissertation are
those obtained on the feature-based UCI corpora, not the figure this demonstrator produces.

## Interfaces

The user interface is served at the root of the Space. The API it calls is open and
documented at `/docs`, with `POST /predict/voice`, `POST /predict/mri`,
`POST /predict/combined` and `GET /health`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Space secret. Authenticates the explanation layer against Hugging Face Inference Providers. Without it the response falls back to a deterministic grounded summary. |
| `SLM_BACKEND` | `hf` in this deployment. `ollama` reaches a local daemon during development, `off` disables generation. |
| `SLM_MODEL` | Chat model behind the explanation, `Qwen/Qwen3-4B-Instruct-2507` by default. |

## Build

The container compiles the React frontend, installs the CPU build of PyTorch, caches the
ImageNet backbone weights and serves the compiled bundle and the API from one process on
port 7860.
