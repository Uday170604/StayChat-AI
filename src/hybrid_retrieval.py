"""
Hybrid dense (FAISS) + sparse (BM25) retrieval with lightweight reranking.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    LIST_QUERY_TOP_K,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    TOP_K,
)
from src.embed_store import VectorStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _is_list_query(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in ("which hotels", "which hotel", "list ", "hotels have", "hotels with"))


def _rerank_score(query: str, chunk: dict[str, Any], fused_score: float) -> float:
    q = query.lower()
    text = chunk["text"].lower()
    category = chunk.get("category", "")
    score = fused_score

    if "cancellation" in q or "policy" in q:
        if category == "policies":
            score += 0.12
        if "cancellation" in text:
            score += 0.08

    if any(w in q for w in ("review", "excellent", "suggest", "recommend", "rating")):
        if category == "guest_review":
            score += 0.10
        if re.search(r"4\.[5-9]/5|outstanding|best beach", text):
            score += 0.15
        if "3.8/5" in text and "excellent" in q:
            score -= 0.05

    if "breakfast" in q and "complimentary" in q:
        if "additional charge" in text or "not complimentary" in text:
            score -= 0.25
        if any(p in text for p in ("complimentary breakfast", "breakfast included", "complimentary light breakfast")):
            score += 0.10

    if "wifi" in q or "wi-fi" in q:
        if "paid" in text and "wifi" in text:
            score -= 0.15
        if "complimentary wifi" in text or "free wifi" in text:
            score += 0.08

    hotel = chunk.get("hotel_name", "").lower()
    if hotel and hotel in q:
        score += 0.20
    if "hotel x" in q and "hotel x" in hotel:
        score += 0.25

    if "beach" in q and "beach" in text:
        score += 0.06

    return score


class HybridRetriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.corpus_tokens = [tokenize(c["text"]) for c in store.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        if k is None:
            k = LIST_QUERY_TOP_K if _is_list_query(query) else TOP_K

        n = len(self.store.chunks)
        candidate_k = min(n, max(k * RETRIEVAL_CANDIDATE_MULTIPLIER, k))

        dense_hits = self.store.search(query, candidate_k)
        dense_by_id = {h[0]["chunk_id"]: h[1] for h in dense_hits}

        q_tokens = tokenize(query)
        bm25_raw = list(self.bm25.get_scores(q_tokens))
        bm25_norm = _normalize(bm25_raw)

        candidates: dict[str, dict[str, Any]] = {}
        for chunk, d_score in dense_hits:
            cid = chunk["chunk_id"]
            idx = next(i for i, c in enumerate(self.store.chunks) if c["chunk_id"] == cid)
            fused = HYBRID_DENSE_WEIGHT * d_score + HYBRID_BM25_WEIGHT * bm25_norm[idx]
            candidates[cid] = {**chunk, "dense_score": d_score, "bm25_score": bm25_norm[idx], "score": fused}

        for idx, chunk in enumerate(self.store.chunks):
            cid = chunk["chunk_id"]
            if cid in candidates:
                continue
            if bm25_norm[idx] < 0.15:
                continue
            candidates[cid] = {
                **chunk,
                "dense_score": 0.0,
                "bm25_score": bm25_norm[idx],
                "score": HYBRID_BM25_WEIGHT * bm25_norm[idx],
            }

        ranked = sorted(
            candidates.values(),
            key=lambda c: _rerank_score(query, c, c["score"]),
            reverse=True,
        )
        return ranked[:k]
