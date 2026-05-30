"""Central configuration for the StayChat Hotel RAG assessment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_PATH = DATA_DIR / "hotel_documents.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
INDEX_DIR = PROJECT_ROOT / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Chunking (Task 1) — smaller windows so policies/reviews can split across chunks
CHUNK_SIZE = 280
CHUNK_OVERLAP = 60

# Retrieval (Task 2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 7
LIST_QUERY_TOP_K = 10
HYBRID_DENSE_WEIGHT = 0.65
HYBRID_BM25_WEIGHT = 0.35
RETRIEVAL_CANDIDATE_MULTIPLIER = 4

# Generation (Task 3)
OPENAI_MODEL = "gpt-3.5-turbo"
OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
MAX_CONTEXT_CHARS = 6000

# Hallucination control (Task 5)
MIN_RETRIEVAL_SCORE = 0.35
MIN_ANSWER_CONFIDENCE = 0.25
