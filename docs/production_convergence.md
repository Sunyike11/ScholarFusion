# Production Convergence

Phase 3 replaces the earlier adapter-only validation with a traced production validation path:

```text
query
→ multimodal dense
→ Chroma dense
→ LlamaIndex BM25
→ optional page visual
→ protected Weighted RRF
→ reranker
→ evidence verifier
```

## Trace Evidence

`evaluation_results/phase3/production_trace.json` reports:

```json
{
  "chroma_collection": "phase3_research_text_dense",
  "chroma_count": 45,
  "bm25_enabled": true,
  "bm25_node_count": 615
}
```

The fixed trace set covers text, figure, image, pair, and cross-paper query types.

## Default Routing

- Primary: multimodal dense retrieval.
- Secondary: Chroma dense text retrieval.
- Auxiliary: LlamaIndex BM25.
- Page Visual: enabled only for image-heavy query types.
- Graph: disabled by default; retained as an optional cross-paper enhancement.

## Why Protected RRF

The earlier hybrid run could degrade below dense-only. Protected RRF preserves the primary dense top candidates during final truncation while still allowing BM25 and Chroma hits to enter the ranking. Current debug result:

```json
{
  "dense_recall@10": 0.24285714285714285,
  "hybrid_recall@10": 0.24285714285714285,
  "hybrid_not_worse_than_dense": true
}
```

## Remaining Boundary

The Chroma collection is rebuilt from ScholarFusion structured vectors because the original persisted legacy Chroma/BM25 cache was not present. This still validates real Chroma and real LlamaIndex BM25 execution in the production validation path.
