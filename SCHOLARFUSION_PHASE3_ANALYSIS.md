# ScholarFusion Phase 3 Analysis

Phase name: Production Retrieval Convergence + Calibration + Small Gold Evaluation.

## Actual Current Production Path

Current ResearchService path is not:

```text
query -> Chroma dense -> BM25 -> multimodal -> page visual -> RRF -> reranker -> verifier -> answer
```

It is closer to:

```text
query -> ScholarFusion SQLite vectors + structured lexical retrieval -> Weighted RRF -> reranker fallback -> verifier -> answer
```

`HybridResearchRetriever` is the unified entry, but its legacy adapter is empty by default.

## Why `legacy_adapter_not_initialized` Appears

The legacy RAG path requires an initialized `RAGWorkflow` from `RAGApplication`, backed by Chroma and BM25 persistence. In the current working tree:

- `file/chroma_db` does not exist.
- `file/storage_bm25` does not exist.
- The ScholarFusion research service is constructed independently from `RAGApplication`.

Therefore the legacy adapter has no workflow instance to call.

## Chroma / BM25 Reality

The original code supports:

- Chroma through `DocumentIngestionPipeline` and `ChromaVectorStore`.
- LlamaIndex BM25 through `RAGWorkflow._initialize_bm25`.

But those persisted indexes are absent for the current ScholarFusion real-paper corpus. Phase 3 must rebuild a small production validation path from the structured research corpus:

- Chroma dense over research text elements.
- LlamaIndex BM25 over research text elements.
- Existing SQLite multimodal/page/paper indexes.

This is a real production convergence path using Chroma and LlamaIndex BM25, but it is not a restored old cache.

## Fusion Problem

Phase 2.1 ablation showed:

```text
B2 Recall@10 < B0 Recall@10
```

Cause: RRF can push dense hits below top-k when a noisy BM25 list contributes candidates. Hybrid fusion needs a monotonic protection rule: adding auxiliary retrievers should not remove strong primary dense hits from the final top-k.

## Retriever Priority

Observed best module:

```text
qwen3-vl-embedding / multimodal dense
```

Phase 3 production policy should be:

- Primary: multimodal dense.
- Secondary: Chroma text dense.
- Auxiliary: BM25.
- Page visual: only image-heavy queries.
- Graph: off by default.

## Evidence Verifier Problem

Phase 2.1:

```text
false_refusal_rate=0.6857
```

The verifier is too conservative for excerpt-style questions. Calibration must lower refusal without increasing false answers substantially.

## Small Gold Constraint

Opened CSV files currently have empty human review fields:

- `review_decision`
- `spot_check_status`
- `human_notes`

No reliable human-confirmed rows exist. Phase 3 can prepare `small_gold_v1` candidates and import true human rows if filled later, but must not mark rows as `human_verified=true` without user edits.

## Phase 3 Files

Planned:

- `core/phase3_production_retrieval.py`
- `scripts/trace_phase3_production_path.py`
- `scripts/debug_rrf_pipeline.py`
- `scripts/query_type_breakdown.py`
- `scripts/calibrate_answerability.py`
- `scripts/build_small_gold_v1.py`
- `scripts/run_phase3_final_ablation.py`
- `SCHOLARFUSION_PHASE3_RESULT.md`
- `docs/production_convergence.md`
- `docs/calibration.md`
- `docs/query_breakdown.md`
- `docs/small_gold_evaluation.md`
- `docs/final_ablation.md`
