"""
Task 2 — Embeddings and FAISS vector store.

Model: all-MiniLM-L6-v2 (384-dim) — strong sentence-level semantics, small footprint,
no API cost, suitable for evaluator mock runs.
Index: FAISS IndexFlatIP on L2-normalized vectors (cosine similarity via inner product).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, EMBEDDING_MODEL, INDEX_DIR


_STORE_CACHE: dict[str, "VectorStore"] = {}


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")

    def build_from_chunks(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        matrix = self.embed_texts(texts)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(matrix)

    def save(self, directory: Path = INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "faiss.index"))
        meta = {"model_name": self.model_name, "dimension": self.dimension, "chunks": self.chunks}
        (directory / "metadata.pkl").write_bytes(pickle.dumps(meta))

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "VectorStore":
        key = str(directory.resolve())
        if key in _STORE_CACHE:
            return _STORE_CACHE[key]
        store = cls()
        store.index = faiss.read_index(str(directory / "faiss.index"))
        meta = pickle.loads((directory / "metadata.pkl").read_bytes())
        store.model_name = meta["model_name"]
        store.dimension = meta["dimension"]
        store.chunks = meta["chunks"]
        store.model = SentenceTransformer(store.model_name)
        _STORE_CACHE[key] = store
        return store

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_chunks first.")
        q_vec = self.embed_texts([query])
        scores, indices = self.index.search(q_vec, k)
        results: list[tuple[dict, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


def build_index(chunks_path: Path = CHUNKS_PATH, index_dir: Path = INDEX_DIR) -> VectorStore:
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload["chunks"]
    store = VectorStore()
    store.build_from_chunks(chunks)
    store.save(index_dir)
    return store


if __name__ == "__main__":
    store = build_index()
    print(f"Indexed {len(store.chunks)} chunks with {store.model_name}")
