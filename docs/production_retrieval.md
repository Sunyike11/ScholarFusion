# Production Retrieval

ScholarFusion Phase2 routes ResearchService search and QA calls through `HybridResearchRetriever`.

## Entry Point

`core/hybrid_research_retriever.py`

Unified result fields:

- `result_id`
- `paper_id`
- `element_id`
- `page_number`
- `element_type`
- `modality`
- `raw_scores`
- `rank_positions`
- `retrieval_sources`
- `fusion_score`
- `rerank_score`

## Retrieval Sources

- Legacy Chroma/BM25 adapter: optional, used only when an initialized legacy workflow is supplied.
- ScholarFusion lexical retriever: structured element text retrieval.
- ScholarFusion dense text vectors: `research_text_elements`.
- Multimodal dense vectors: `research_multimodal_elements`.
- Page visual vectors: `research_page_visual`.
- Paper-level vectors: `research_paper_level`.
- Graph retriever: optional boost, not a replacement for vector retrieval.

## Fusion

The current fusion method is Weighted RRF. Per-result metadata preserves source names, raw scores and rank positions for debugging and ablation.

## Default Safety

Default pytest and default service construction do not call paid model APIs. Real reranking requires:

```bash
SCHOLARFUSION_ENABLE_REAL_RERANKER=true
```

## Known Boundary

The legacy Chroma/BM25 path is now represented as an adapter, but Phase2 smoke validation used the ScholarFusion SQLite vector index. A production deployment should pass the initialized legacy workflow into `LegacyProductionRetrieverAdapter` before claiming full old-pipeline reuse in metrics.
