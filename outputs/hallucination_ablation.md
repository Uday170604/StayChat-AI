# Hallucination control ablation

**Query:** Which hotels allow pet tigers?

## Without controls (permissive prompt)
Based on the documents: Pet-friendly rooms available on request (surcharge applies). [pol_003_chunk_0] Additional partner properties may offer similar services.

## With controls (threshold + strict prompt + verification)
I don't have enough information in the provided documents to answer that.

### Why this reduces hallucination
- **Layer 1 — Retrieval threshold:** blocks generation when no chunk is sufficiently similar.
- **Layer 2 — Query-term grounding:** abstains when key question terms (e.g. *tigers*) are absent from context.
- **Layer 3 — Strict prompt:** instructs the model to use only provided chunks.
- **Layer 4 — Lexical verification:** downgrades answers whose tokens lack support in retrieved text.
