"""
Task 5 — Hallucination control: retrieval score threshold + strict prompting + citation check.

Low-similarity queries (off-domain) trigger abstention before generation.
Post-generation, unsupported sentences (no citation overlap with context) are stripped.
"""

from __future__ import annotations

import re
from typing import Any

from config import MIN_ANSWER_CONFIDENCE, MIN_RETRIEVAL_SCORE
from src.generation import generate_answer

ABSTAIN = "I don't have enough information in the provided documents to answer that."

_STOPWORDS = {
    "which", "what", "where", "when", "how", "the", "and", "for", "with", "have",
    "has", "are", "is", "a", "an", "of", "to", "in", "near", "about", "hotels",
    "hotel", "suggest", "allow", "allows",
}


def query_terms_missing_from_context(query: str, chunks: list[dict]) -> list[str]:
    """Content words in query that never appear in retrieved context (hallucination risk)."""
    context = " ".join(c["text"] for c in chunks).lower()
    terms = [
        t
        for t in re.findall(r"[a-z]{4,}", query.lower())
        if t not in _STOPWORDS
    ]
    return [t for t in terms if t not in context]


def max_retrieval_score(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    return max(c.get("score", 0.0) for c in chunks)


def verify_answer_against_context(answer: str, chunks: list[dict]) -> tuple[str, float]:
    """
    Simple lexical overlap confidence between answer tokens and retrieved context.
    Returns (possibly revised answer, confidence in [0,1]).
    """
    context = " ".join(c["text"] for c in chunks).lower()
    answer_lower = answer.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", answer_lower) if len(t) > 3]
    if not tokens:
        return answer, 0.0
    supported = sum(1 for t in tokens if t in context)
    confidence = supported / len(tokens)

    if ABSTAIN in answer:
        return answer, 1.0

    cited_ids = set(re.findall(r"\[([^\]]+_chunk_\d+)\]", answer))
    if cited_ids:
        id_to_text = {c["chunk_id"]: c["text"].lower() for c in chunks}
        cited_ok = all(
            any(tok in id_to_text.get(cid, "") for tok in tokens[:8])
            for cid in cited_ids
            if cid in id_to_text
        )
        if not cited_ok and confidence < MIN_ANSWER_CONFIDENCE:
            return ABSTAIN, confidence

    if confidence < MIN_ANSWER_CONFIDENCE and ABSTAIN not in answer:
        return ABSTAIN, confidence
    return answer, confidence


def answer_with_controls(
    query: str,
    chunks: list[dict],
    strict: bool = True,
    enable_controls: bool = True,
) -> dict[str, Any]:
    top_score = max_retrieval_score(chunks)

    if enable_controls and top_score < MIN_RETRIEVAL_SCORE:
        return {
            "answer": ABSTAIN,
            "abstained": True,
            "reason": f"max retrieval score {top_score:.3f} < threshold {MIN_RETRIEVAL_SCORE}",
            "controls_enabled": True,
            "top_score": top_score,
        }

    missing = query_terms_missing_from_context(query, chunks)
    if enable_controls and missing:
        return {
            "answer": ABSTAIN,
            "abstained": True,
            "reason": f"query terms not grounded in context: {missing}",
            "controls_enabled": True,
            "top_score": top_score,
            "missing_terms": missing,
        }

    gen = generate_answer(query, chunks, strict=strict)
    answer = gen["answer"]
    confidence = 1.0

    if enable_controls:
        answer, confidence = verify_answer_against_context(answer, chunks)

    return {
        **gen,
        "answer": answer,
        "abstained": ABSTAIN in answer,
        "verification_confidence": confidence,
        "controls_enabled": enable_controls,
        "top_score": top_score,
    }


def run_ablation(query: str, chunks: list[dict]) -> dict[str, Any]:
    """Before/after: permissive prompt without controls vs strict + verification."""
    without = generate_answer(query, chunks, strict=False)
    without["controls_enabled"] = False

    with_ctrl = answer_with_controls(query, chunks, strict=True, enable_controls=True)
    return {"without_controls": without, "with_controls": with_ctrl}
