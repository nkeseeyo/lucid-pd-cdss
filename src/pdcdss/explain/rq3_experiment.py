"""RQ3: grounded vs ungrounded small-language-model explanation, compared across models.

For each case (from cases.py), each model (CANDIDATES) and each condition
(grounded RAG + guardrail vs an ungrounded baseline prompt), this generates an
explanation and scores it on:

  * faithfulness  -- share of the explanation's claims that the supplied context
                     supports, judged by an LLM-as-judge (gemma4:12b), following the
                     RAGAS faithfulness definition;
  * unsupported   -- number of claims NOT supported by the supplied context
                     (the hallucination / overclaim count);
  * readability   -- Flesch-Kincaid grade level;
  * consistency   -- mean pairwise similarity across repeated generations;
  * latency       -- seconds per generation.

The judge model (gemma4:12b, already local) is deliberately larger than the systems
under test (the ~4B candidates), so the evaluator is not grading its own family.

Run:
    python -m pdcdss.explain.rq3_experiment                  # full run
    python -m pdcdss.explain.rq3_experiment --models phi4-mini --max-cases 2 --repeats 2

Outputs:
    results/tables/rq3_slm_runs.csv          every generation + its scores
    results/tables/rq3_slm_faithfulness.csv  aggregated per model x condition
    results/figures/rq3_faithfulness.png     faithfulness + unsupported-claim bars
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from pdcdss.config import RESULTS_FIGURES, RESULTS_TABLES  # noqa: E402
from pdcdss.explain.cases import build_cases  # noqa: E402
from pdcdss.explain.guidance import retrieved_block  # noqa: E402
from pdcdss.explain.slm_explain import CANDIDATES, _chat, generate  # noqa: E402

JUDGE_MODEL = "gemma4:12b"
REPEAT_SEEDS = [42, 43, 44]
PRETTY = {"gemma4:e4b": "Gemma 4 E4B", "phi4-mini": "Phi-4-mini", "qwen3:4b": "Qwen3-4B"}

_JUDGE_SYSTEM = (
    "You are a strict clinical fact-checker. You are given CONTEXT (the only information "
    "the explanation was allowed to use) and an EXPLANATION. List every distinct factual "
    "or clinical claim the explanation makes. For each claim, first give a brief reason, "
    "then set supported=true only if the claim is directly stated in, or directly entailed "
    "by, the CONTEXT; otherwise supported=false. Treat added clinical facts, numbers, "
    "symptoms or advice not in the CONTEXT as unsupported. Respond with JSON only, no "
    'other prose: {"claims":[{"claim":"...","reason":"...","supported":true}]}'
)


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch_kincaid_grade(text: str) -> float:
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sents or not words:
        return float("nan")
    syl = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / len(sents) + 11.8 * syl / len(words) - 15.59


def consistency(texts: list[str]) -> float:
    """Mean pairwise TF-IDF cosine over repeated generations (1.0 = identical)."""
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return float("nan")
    m = TfidfVectorizer().fit_transform(texts)
    sims = cosine_similarity(m)
    iu = np.triu_indices_from(sims, k=1)
    return float(sims[iu].mean())


def _context(case: pd.Series, grounded: bool) -> str:
    """The information the explanation was allowed to use (judge sees the same)."""
    block = (f"Prediction: {case['prediction']}\nRisk band: {case['risk_band']}\n"
             f"Top contributing voice features: {case['top_features']}")
    if grounded:
        query = f"{case['prediction']} {case['risk_band']} {case['top_features']}"
        block += "\nRetrieved guideline text:\n" + retrieved_block(query, k=3)
    return block


def judge_faithfulness(text: str, context: str) -> tuple[int, int]:
    """Return (n_claims, n_supported) via the LLM-as-judge. (0, 0) on parse failure.

    The judge (gemma4:12b) is a reasoning model whose answer follows a hidden thinking
    pass, so it needs a generous token budget or it returns empty content; we retry with
    a larger budget if the first attempt yields no parseable JSON.
    """
    user = f"CONTEXT:\n{context}\n\nEXPLANATION:\n{text}\n\nReturn the JSON now."
    for budget in (4096, 8192):
        raw, _ = _chat(JUDGE_MODEL, _JUDGE_SYSTEM, user, temperature=0.0,
                       num_predict=budget, think=False)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            continue
        try:
            claims = json.loads(m.group(0)).get("claims", [])
        except json.JSONDecodeError:
            continue
        if claims:
            supported = sum(1 for c in claims if c.get("supported") is True)
            return len(claims), supported
    return 0, 0


def run(models: list[str], cases: pd.DataFrame, repeats: int) -> pd.DataFrame:
    rows = []
    seeds = REPEAT_SEEDS[:repeats]
    for model in models:
        for _, case in cases.iterrows():
            for grounded in (True, False):
                cond = "grounded" if grounded else "ungrounded"
                outs, lats = [], []
                for sd in seeds:
                    txt, lat = generate(case["prediction"], case["risk_band"],
                                        [case["top_features"]], grounded=grounded,
                                        model=model, seed=sd)
                    outs.append(txt)
                    lats.append(lat)
                primary = outs[0]
                n_claims, supported = judge_faithfulness(primary, _context(case, grounded))
                faith = supported / n_claims if n_claims else float("nan")
                rows.append({
                    "model": model, "condition": cond, "case_id": case["case_id"],
                    "risk_band": case["risk_band"],
                    "faithfulness": faith, "n_claims": n_claims,
                    "unsupported": n_claims - supported,
                    "fk_grade": round(flesch_kincaid_grade(primary), 2),
                    "consistency": round(consistency(outs), 3),
                    "latency_s": round(float(np.mean(lats)), 2),
                    "words": len(re.findall(r"[A-Za-z]+", primary)),
                    "text": primary.replace("\n", " "),
                })
                print(f"  {PRETTY.get(model, model):12s} {cond:10s} {case['case_id']} "
                      f"faith={faith if faith==faith else float('nan'):.2f} "
                      f"unsup={n_claims - supported} fk={rows[-1]['fk_grade']:.1f} "
                      f"lat={rows[-1]['latency_s']:.1f}s")
    return pd.DataFrame(rows)


def _aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    agg = (runs.groupby(["model", "condition"])
           .agg(faithfulness=("faithfulness", "mean"),
                unsupported=("unsupported", "mean"),
                fk_grade=("fk_grade", "mean"),
                consistency=("consistency", "mean"),
                latency_s=("latency_s", "mean"),
                n=("case_id", "count"))
           .round(3).reset_index())
    return agg


def _plot(agg: pd.DataFrame) -> None:
    models = [m for m in PRETTY if m in set(agg["model"])]
    x = np.arange(len(models))
    w = 0.38

    def vals(metric, cond):
        return [float(agg[(agg["model"] == m) & (agg["condition"] == cond)][metric].iloc[0])
                if len(agg[(agg["model"] == m) & (agg["condition"] == cond)]) else np.nan
                for m in models]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [
        (ax[0], "faithfulness", "Faithfulness (claims supported by context)",
         "mean faithfulness", 1.16),
        (ax[1], "unsupported", "Unsupported claims per explanation",
         "mean unsupported claims", None),
    ]
    for a, metric, title, ylab, ytop in panels:
        gb = a.bar(x - w / 2, vals(metric, "grounded"), w, label="Grounded (RAG)",
                   color="#0C5C5E")
        ub = a.bar(x + w / 2, vals(metric, "ungrounded"), w, label="Ungrounded",
                   color="#9A5B00")
        a.bar_label(gb, fmt="%.2f", padding=2, fontsize=8)
        a.bar_label(ub, fmt="%.2f", padding=2, fontsize=8)
        a.set_xticks(x)
        a.set_xticklabels([PRETTY[m] for m in models])
        a.set_title(title, fontweight="bold")
        a.set_ylabel(ylab)
        top = ytop or max(vals(metric, "grounded") + vals(metric, "ungrounded")) * 1.20
        a.set_ylim(0, top)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS_FIGURES / "rq3_faithfulness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cases_csv = RESULTS_TABLES / "rq3_cases.csv"
    cases = pd.read_csv(cases_csv) if cases_csv.exists() else build_cases()
    if args.max_cases:
        cases = cases.head(args.max_cases)

    print(f"RQ3: {len(args.models)} models x 2 conditions x {len(cases)} cases "
          f"x {args.repeats} repeats; judge = {JUDGE_MODEL}\n")
    runs = run(args.models, cases, args.repeats)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RESULTS_TABLES / "rq3_slm_runs.csv", index=False)
    agg = _aggregate(runs)
    agg.to_csv(RESULTS_TABLES / "rq3_slm_faithfulness.csv", index=False)
    _plot(agg)

    print("\nAggregated (mean per model x condition):")
    print(agg.to_string(index=False))
    print(f"\nSaved -> {RESULTS_TABLES / 'rq3_slm_faithfulness.csv'} and "
          f"{RESULTS_FIGURES / 'rq3_faithfulness.png'}")


if __name__ == "__main__":
    main()
