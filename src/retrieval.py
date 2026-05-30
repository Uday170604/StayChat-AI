"""Query-time hybrid retrieval wrapper."""

from __future__ import annotations

from config import INDEX_DIR, TOP_K
from src.embed_store import VectorStore
from src.hybrid_retrieval import HybridRetriever

_RETRIEVER_CACHE: HybridRetriever | None = None


def clear_retriever_cache() -> None:
    global _RETRIEVER_CACHE
    _RETRIEVER_CACHE = None


def get_retriever(index_dir=INDEX_DIR) -> HybridRetriever:
    global _RETRIEVER_CACHE
    if _RETRIEVER_CACHE is None:
        store = VectorStore.load(index_dir)
        _RETRIEVER_CACHE = HybridRetriever(store)
    return _RETRIEVER_CACHE


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """
    Hybrid dense + BM25 retrieval with reranking.

    k=7 (10 for list queries): improves recall for multi-hotel amenity questions
    without exceeding the LLM context budget.
    """
    retriever = get_retriever()
    if k is None:
        return retriever.retrieve(query)
    return retriever.retrieve(query, k=k)
