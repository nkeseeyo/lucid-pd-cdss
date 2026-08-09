"""Explainability + its evaluation (no human participants required).

Planned modules:
  shap_speech.py   — global (beeswarm) + local SHAP; feature-family attribution;
                     stability across folds (weak if top features change).
  faithfulness.py  — insertion/deletion curves; does the LLM explanation only
                     reference features SHAP flagged?  readability (Flesch-Kincaid);
                     consistency (similar cases -> similar explanations).
  slm_explain.py   — Gemma 4 (E4B) via Ollama rephrases {prediction, risk band,
                     top SHAP features} into plain language, grounded by single-pass
                     RAG over a tiny NICE/NHS corpus. Rephrase only — add no clinical
                     facts. A small on-device model keeps it local, free and no-GPU.
"""
