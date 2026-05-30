# Submission — StayChat AI/ML RAG Assessment

**Candidate:** Uday Garodhara 
**Role:** AI / ML Engineer  
**Date:** 30/5/2026

## One-minute summary

End-to-end RAG pipeline for hotel Q&A: sentence-aware chunking, **hybrid dense+BM25 retrieval** with reranking, context-only generation (OpenAI / Ollama / generic mock), Precision@k + reciprocal rank evaluation, and layered hallucination controls with ablation.

## How to run (evaluators)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py --all
python -m unittest tests.test_rag -v
```

Outputs: `outputs/sample_outputs.md`, `outputs/hallucination_ablation.md`

**No API key required** — uses local embeddings + generic mock LLM. Optional:

```bash
set OPENAI_API_KEY=sk-...
set USE_OLLAMA=1
set OLLAMA_MODEL=llama3.2
python main.py --evaluate
```

## Design highlights

| Area | Decision |
|------|----------|
| Chunking | 280 chars, 60 overlap — splits longer amenity/policy text |
| Embeddings | `all-MiniLM-L6-v2` (384-d), cosine via FAISS IP |
| Retrieval | 65% dense + 35% BM25, rerank boosts policies/reviews, penalizes paid breakfast |
| k | 7 default, 10 for list queries |
| Hallucination | Score threshold + query-term grounding + lexical verification + strict prompt |
| Generation | No hardcoded Q1/Q2/Q3 branches; mock synthesizes from retrieved chunks only |

## Dataset

40 synthetic documents (MIT). Categories: 9 descriptions, 7 amenities, 10 reviews, 7 policies, 7 location — see `data/hotel_documents.json`.

## Known trade-offs

- Mock LLM is lexical, not neural — use OpenAI/Ollama for production-quality phrasing.
- Ground-truth labels are manual for three queries; metrics are indicative.
- Cross-encoder reranking listed as future work in evaluation notes.
