"""Single-pass retrieval over the tiny NICE/NHS guidance corpus (RQ3).

The retrieval is deliberately simple and reproducible: TF-IDF cosine similarity over
a handful of curated snippets, returning the top-k for a query. This is the grounding
source for the SLM explanation layer. Keeping it small and transparent means a reader
can see exactly which guideline text any explanation was allowed to draw on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdcdss.config import EXTERNAL_DIR

GUIDANCE_JSON = EXTERNAL_DIR / "guidance" / "parkinsons_guidance.json"


@dataclass(frozen=True)
class Snippet:
    id: str
    source: str
    text: str

    def cite(self) -> str:
        return f"[{self.source}] {self.text}"


@lru_cache(maxsize=1)
def _load() -> tuple[tuple[Snippet, ...], TfidfVectorizer, object]:
    data = json.loads(GUIDANCE_JSON.read_text(encoding="utf-8"))
    snippets = tuple(Snippet(s["id"], s["source"], s["text"]) for s in data["snippets"])
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([s.text for s in snippets])
    return snippets, vec, matrix


def retrieve(query: str, k: int = 3) -> list[Snippet]:
    """Return the k guidance snippets most similar to the query, best first."""
    snippets, vec, matrix = _load()
    k = min(k, len(snippets))
    sims = cosine_similarity(vec.transform([query]), matrix).ravel()
    order = sims.argsort()[::-1][:k]
    return [snippets[i] for i in order]


def retrieved_block(query: str, k: int = 3) -> str:
    """Format the retrieved snippets as a verbatim, citable context block."""
    return "\n".join(f"- {s.cite()}" for s in retrieve(query, k))


def all_snippets() -> list[Snippet]:
    return list(_load()[0])
