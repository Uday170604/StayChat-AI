"""
Task 1 — Preprocessing: clean text, chunk with overlap, persist chunks.

Chunking strategy: sentence-aware fixed-size windows (character-based).
Rationale: hotel corpus mixes short policies, bullet-like amenities, and narrative
reviews. Sentence boundaries reduce mid-sentence splits for policy clauses and
review quotes. Overlap preserves cross-boundary context (e.g., WiFi in one chunk
and breakfast in the next for the same hotel).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, RAW_DOCS_PATH


def clean_text(raw: str) -> str:
    """Remove HTML, boilerplate markers, encoding noise, and excess whitespace."""
    text = raw
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"(?i)(click here|lorem ipsum|all rights reserved)", " ", text)
    text = re.sub(r"[^\w\s.,;:'\"!?()\-/&@#%+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Build overlapping chunks from a single document."""
    cleaned = clean_text(doc["text"])
    sentences = _split_sentences(cleaned)
    chunks: list[dict[str, Any]] = []
    buffer = ""
    chunk_index = 0

    def flush(buf: str, idx: int) -> dict[str, Any]:
        chunk_id = f"{doc['id']}_chunk_{idx}"
        return {
            "chunk_id": chunk_id,
            "doc_id": doc["id"],
            "hotel_name": doc.get("hotel_name", ""),
            "category": doc.get("category", ""),
            "title": doc.get("title", ""),
            "text": buf.strip(),
        }

    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(flush(buffer, chunk_index))
                chunk_index += 1
                tail = buffer[-overlap:] if overlap < len(buffer) else buffer
                buffer = f"{tail} {sentence}".strip()
            else:
                # Single very long sentence — hard split
                start = 0
                while start < len(sentence):
                    piece = sentence[start : start + chunk_size]
                    chunks.append(flush(piece, chunk_index))
                    chunk_index += 1
                    start += chunk_size - overlap
                buffer = ""

    if buffer:
        chunks.append(flush(buffer, chunk_index))

    return chunks


def load_raw_documents(path: Path = RAW_DOCS_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["documents"]


def build_chunks(
    docs_path: Path = RAW_DOCS_PATH,
    out_path: Path = CHUNKS_PATH,
) -> list[dict[str, Any]]:
    docs = load_raw_documents(docs_path)
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "strategy": "sentence-aware fixed-size with overlap",
                "chunks": all_chunks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Created {len(chunks)} chunks -> {CHUNKS_PATH}")
