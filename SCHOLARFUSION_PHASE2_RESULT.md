# ScholarFusion Phase 2 Result

Phase name: Production Retrieval Unification, Visual Retrieval and Quality Evaluation.

## What Changed

- Added `HybridResearchRetriever` as the unified ResearchService retrieval entry.
- Reconnected ResearchService text, figure, any-to-any and QA requests through the unified retriever.
- Added real rendered-page visual indexing through `PageVisualRenderer`.
- Added page image vectors in `research_page_visual` with `input_modality=page_image`.
- Split citation output into `primary_citations` and `context_citations`.
- Added claim-level citation planning and Phase2 citation metrics.
- Added configurable real `qwen3-vl-rerank` provider with safe deterministic fallback.
- Added real LLM Paper Card builder with schema and evidence validation.
- Added Phase2 human review queue generation and import tooling.
- Added index consistency checker.

## Modified Files

- `config/settings.py`
- `core/hybrid_research_retriever.py`
- `core/page_visual.py`
- `core/phase2_citations.py`
- `core/phase2_reranker_provider.py`
- `core/paper_card_llm.py`
- `core/research_context.py`
- `core/research_service.py`
- `core/scholarfusion_figures.py`
- `core/scholarfusion_indexer.py`
- `scripts/build_llm_paper_cards.py`
- `scripts/build_phase2_review_queue.py`
- `scripts/build_scholarfusion_lite_eval.py`
- `scripts/check_research_index_consistency.py`
- `scripts/evaluate_paper_parsing.py`
- `scripts/import_human_review.py`
- `scripts/run_scholarfusion_lite_evaluation.py`
- `scripts/run_scholarfusion_smoke.py`

## Real Validation

Default unit tests do not call paid APIs:

```bash
conda run -n rag python -m pytest -q
```

Result:

```text
20 passed, 1 skipped, 5 warnings
```

Real/fake mixed smoke over 3 local papers:

```bash
conda run -n rag python scripts/run_scholarfusion_smoke.py \
  --pdf-dir data/research_papers \
  --max-papers 3 \
  --max-requests 45 \
  --output-dir evaluation_results/scholarfusion_phase2 \
  --force
```

Result summary:

```json
{
  "status": "ok",
  "provider": "fake",
  "paper_count": 3,
  "embedding_request_count": 115,
  "index_counts": {
    "text": 45,
    "multimodal": 40,
    "page": 27,
    "paper": 3
  },
  "qa_supported_answerable": true,
  "qa_unanswerable_answerable": false
}
```

The smoke used the fake embedding provider for cost-safe repeatability. A separate small real `qwen3-vl-rerank` call succeeded with provider `qwen3-vl-rerank`.

## Page Visual

Page visual indexing now renders PDFs to stable PNGs under the research asset directory and embeds those images. SQLite vector metadata confirms:

```text
research_page_visual | 27 vectors | input_modality=page_image | page_image_path present
```

This replaces the previous page-text proxy for normal PDF-backed indexing. A proxy remains only for tests or missing-source fallback.

## Figure Quality

```json
{
  "paper_count": 3,
  "figure_count": 394,
  "unique_figure_count": 65,
  "duplicate_figure_rate": 0.835,
  "figures_with_caption_rate": 0.3385,
  "high_confidence_caption_rate": 0.3385,
  "explicit_reference_rate": 0.2923,
  "fallback_link_rate": 1.0,
  "decorative_filtered_count": 16,
  "duplicate_filtered_count": 313,
  "indexed_figure_count": 65
}
```

The previous `figures_with_linked_paragraph_rate=1.0` is explained by broad same-page fallback links. Phase2 now records `explicit_reference`, `caption_neighbor`, `same_page_neighbor`, and `same_section_fallback` separately. Evidence Bundle and Figure-Text Pair construction prefer only `explicit_reference` and `caption_neighbor`, falling back only when no stronger link exists.

## Paper Cards

Real DashScope text LLM Paper Card extraction was run for 3 papers. Raw model output is saved only in ignored evaluation results and was not printed.

Summary:

```text
paper_count=3
field_coverage=0.8571 for each paper
evidence_validity=1.0 for each paper
unknown_rate=0.1429 for each paper
validation_errors=0 for each paper
```

The builder uses selected structured evidence only, validates element IDs, and resets unsupported values to `unknown`.

## Evaluation

Provisional candidate generation:

```text
candidate_count=92
dataset_status=provisional
not_gold=true
```

Phase2 human review queue:

```text
review_queue_count=30
text_fact=6
method_detail=3
text_to_figure=5
image_to_text=4
image_to_image=3
pair_to_any=3
unanswerable=2
cross_paper_comparison=4
```

Provisional metrics over non-gold data:

```json
{
  "recall@5": 0.1882,
  "recall@10": 0.1938,
  "mrr": 0.3327,
  "citation_precision": 0.0187,
  "primary_citation_precision": 0.0112,
  "citation_page_accuracy": 1.0,
  "claim_support_rate": 0.9447,
  "answerable_accuracy": 0.2826,
  "p50_retrieval_latency_ms": 59.377,
  "p95_retrieval_latency_ms": 61.173,
  "p50_qa_latency_ms": 61.961,
  "p95_qa_latency_ms": 65.622
}
```

These are not final quality numbers. Citation precision is especially low because auto-generated relevant evidence labels are sparse and incomplete. The queue is ready for human verification before resume-grade metrics are claimed.

## Consistency

```json
{
  "paper_count": 3,
  "element_count": 1018,
  "vector_count": 115,
  "orphan_vector_count": 0,
  "missing_element_reference_count": 0,
  "duplicate_vector_id_count": 0,
  "stale_page_image_count": 0,
  "graph_missing_evidence_count": 0
}
```

## Remaining Boundaries

- Legacy Chroma/BM25 can be adapted through `LegacyProductionRetrieverAdapter`, but this smoke still primarily validates the ScholarFusion SQLite vector path plus lexical retriever.
- Real page visual vectors are implemented; the smoke used fake embeddings for repeatability, while earlier real qwen3-vl embedding smoke validated the provider.
- Dedicated `qwen3-vl-rerank` API smoke succeeded, but production service uses it only when `SCHOLARFUSION_ENABLE_REAL_RERANKER=true`.
- OCR/Layout fallback remains a triggerable fallback path, not a benchmark-grade OCR pipeline.
- Human verified evaluation data has not been imported yet.
