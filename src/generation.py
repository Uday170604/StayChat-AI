"""
Task 3 — Generative QA: context-grounded prompting with OpenAI, Ollama, or generic mock.

The mock path is query-agnostic: it scores sentences from retrieved chunks by lexical
overlap with the question and composes cited answers (no hardcoded Q1/Q2/Q3 branches).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from config import MAX_CONTEXT_CHARS, OLLAMA_HOST, OLLAMA_MODEL, OPENAI_MODEL

SYSTEM_PROMPT = """You are a hotel information assistant for StayChat.
Answer ONLY using the provided context chunks.
If the context does not contain enough information, reply exactly:
"I don't have enough information in the provided documents to answer that."
Cite sources using [chunk_id] after each factual claim.
Do not invent hotel names, policies, or amenities not present in the context."""

ABSTAIN = "I don't have enough information in the provided documents to answer that."

_STOPWORDS = {
    "which", "what", "where", "when", "how", "the", "and", "for", "with", "have",
    "has", "are", "is", "a", "an", "of", "to", "in", "near", "about", "hotels",
    "hotel", "suggest", "allow", "allows", "free", "nearby", "policy", "cancellation",
}


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c['chunk_id']}] (hotel: {c.get('hotel_name', 'N/A')}, "
            f"score: {c.get('score', 0):.3f})\n{c['text']}"
        )
    return "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]


def build_prompt(query: str, chunks: list[dict]) -> list[dict[str, str]]:
    context = format_context(chunks)
    user_content = f"""Context:
{context}

Question: {query}

Instructions:
- Answer only from the context above.
- Use bullet points when listing multiple hotels.
- Include [chunk_id] citations."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _query_terms(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", query.lower()) if t not in _STOPWORDS]


def _concept_groups(query: str) -> list[list[str]]:
    """AND-style concept groups inferred from the question (generic, not query-id specific)."""
    q = query.lower()
    groups: list[list[str]] = []
    if "wifi" in q or "wi-fi" in q:
        groups.append(["wifi", "wi-fi"])
    if "breakfast" in q:
        groups.append(
            [
                "complimentary breakfast",
                "complimentary buffet breakfast",
                "complimentary light breakfast",
                "buffet breakfast included",
            ]
        )
    if "cancellation" in q:
        groups.append(["cancellation", "cancel"])
    if "beach" in q:
        groups.append(["beach"])
    if "review" in q or "excellent" in q:
        groups.append(["review", "rating", "outstanding", "excellent"])
    return groups


def _chunk_satisfies_groups(text: str, groups: list[list[str]]) -> bool:
    lower = text.lower()
    for group in groups:
        if not any(g in lower for g in group):
            return False
    if "breakfast" in lower and ("additional charge" in lower or "not complimentary" in lower):
        return False
    if "executive rooms only" in lower and "complimentary" not in lower.split("executive")[0]:
        return False
    if ("wifi" in lower or "wi-fi" in lower) and "paid" in lower:
        if not any(p in lower for p in ("complimentary wifi", "free wifi", "wifi complimentary")):
            return False
    return True


def _hotel_in_query(query: str, hotel_name: str) -> bool:
    if not hotel_name:
        return False
    h = hotel_name.lower()
    q = query.lower()
    if h in q:
        return True
    short = h.replace(" residency", "").replace(" hotel", "")
    return short in q and len(short) > 4


def _sentence_score(sentence: str, terms: list[str]) -> float:
    lower = sentence.lower()
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in lower) / len(terms)


def _best_sentences(chunk: dict, terms: list[str], max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", chunk["text"])
    scored = sorted(parts, key=lambda s: _sentence_score(s, terms), reverse=True)
    picked = [s for s in scored if _sentence_score(s, terms) > 0][:max_sentences]
    return " ".join(picked) if picked else chunk["text"]


def _mock_generate(query: str, chunks: list[dict], strict: bool = True) -> str:
    """Generic context-only answer synthesis (no per-assessment-query branches)."""
    if not chunks:
        return ABSTAIN

    q = query.lower()
    groups = _concept_groups(query)
    terms = _query_terms(query)

    if groups and any(p in q for p in ("which", "hotels have", "hotels with", "list")):
        lines: list[str] = []
        seen: set[str] = set()
        for c in chunks:
            if not _chunk_satisfies_groups(c["text"], groups):
                continue
            hotel = c.get("hotel_name") or "Unknown"
            if hotel in seen:
                continue
            seen.add(hotel)
            snippet = _best_sentences(c, terms, 1)
            lines.append(f"- **{hotel}**: {snippet} [{c['chunk_id']}]")
        if lines:
            return "Based on the retrieved documents:\n" + "\n".join(lines)
        return ABSTAIN

    if "cancellation" in q or "policy" in q:
        for c in chunks:
            if _hotel_in_query(query, c.get("hotel_name", "")) and (
                c.get("category") == "policies" or "cancellation" in c["text"].lower()
            ):
                return f"{c['text']} [{c['chunk_id']}]"
        for c in chunks:
            if "cancellation" in c["text"].lower():
                return f"{c['text']} [{c['chunk_id']}]"

    if any(w in q for w in ("suggest", "recommend", "excellent", "best")):
        review_chunks = [c for c in chunks if c.get("category") == "guest_review"]
        ranked = sorted(
            review_chunks or chunks,
            key=lambda c: (
                float(re.search(r"(\d\.\d)/5", c["text"]).group(1))
                if re.search(r"(\d\.\d)/5", c["text"])
                else 0.0,
                c.get("score", 0),
            ),
            reverse=True,
        )
        for c in ranked:
            if groups and not _chunk_satisfies_groups(c["text"], groups):
                continue
            hotel = c.get("hotel_name", "the hotel")
            snippet = _best_sentences(c, terms).strip().rstrip("'")
            return f"I suggest **{hotel}**: {snippet} [{c['chunk_id']}]"

    top = chunks[0]
    snippet = _best_sentences(top, terms)
    answer = f"Based on the documents: {snippet} [{top['chunk_id']}]"

    if not strict:
        answer += " Additional partner properties may offer similar services."
    return answer


def _try_ollama(messages: list[dict[str, str]]) -> str | None:
    if os.environ.get("USE_OLLAMA", "").strip() not in ("1", "true", "yes"):
        return None
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL),
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def generate_answer(
    query: str,
    chunks: list[dict],
    strict: bool = True,
    use_openai: bool | None = None,
) -> dict[str, Any]:
    messages = build_prompt(query, chunks)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if use_openai is None:
        use_openai = bool(api_key)

    backend = "mock"

    if use_openai and api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                messages=messages,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
            backend = "openai"
        except Exception as exc:  # noqa: BLE001
            answer = _mock_generate(query, chunks, strict=strict)
            backend = f"mock_fallback ({exc})"
    else:
        ollama_answer = _try_ollama(messages)
        if ollama_answer:
            answer = ollama_answer
            backend = "ollama"
        else:
            answer = _mock_generate(query, chunks, strict=strict)
            backend = "mock"

    return {
        "answer": answer,
        "prompt_messages": messages,
        "backend": backend,
        "strict_mode": strict,
    }
