# Production Retrieval Trace

`scripts/trace_scholarfusion_production_retrieval.py` records safe traces for fixed model-reviewed queries.

## Recorded Fields

- `query_id`
- `query_type`
- `query_modality`
- `retrievers_invoked`
- `retrievers_in_top_results`
- `collections_used`
- `bm25_enabled`
- `chroma_text_dense_enabled`
- `multimodal_dense_enabled`
- `page_visual_enabled`
- `paper_level_enabled`
- `graph_enabled`
- `fusion_method`
- `reranker_actual_provider`
- `post_fusion_ranks`
- `post_rerank_ranks`
- `latency_by_stage`

The trace does not store full papers, prompts, reasoning, API keys or image Base64.

## Current Phase2.1 Result

```text
trace_count=5
fusion_method=WeightedRRF
bm25_enabled=true for text/pair queries
multimodal_dense_enabled=true
page_visual_enabled=true
paper_level_enabled=true
graph_enabled=true for text/pair queries
chroma_text_dense_enabled=false
chroma_text_dense_status=legacy_adapter_not_initialized
```

## Boundary

The trace validates the ScholarFusion unified path. It does not prove that the original Chroma collection is active inside ScholarFusion, because the legacy adapter was not initialized in the smoke.
