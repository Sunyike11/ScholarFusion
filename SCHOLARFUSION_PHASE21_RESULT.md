# ScholarFusion Phase 2.1 Result

Phase name: Production Path Validation, Model-Reviewed Evaluation and Ablation.

## Summary

This phase froze a model-reviewed provisional dataset, validated the production retrieval trace, ran a real `qwen3-vl-rerank` smoke, and produced B0-M5 ablation results on the same dataset.

All evaluation outputs remain:

```text
dataset_status=provisional
review_status=model_reviewed
not_gold=true
human_verified=false
```

No metric in this phase is a human Gold result.

## Files Added

- `SCHOLARFUSION_PHASE21_ANALYSIS.md`
- `SCHOLARFUSION_PHASE21_PLAN.md`
- `config/scholarfusion_phase21_experiments.yaml`
- `core/scholarfusion_model_review.py`
- `core/scholarfusion_phase21_metrics.py`
- `scripts/review_scholarfusion_eval_with_model.py`
- `scripts/trace_scholarfusion_production_retrieval.py`
- `scripts/run_scholarfusion_real_reranker_smoke.py`
- `scripts/run_scholarfusion_phase21_ablation.py`
- `tests/test_scholarfusion_phase21.py`
- `docs/model_reviewed_evaluation_protocol.md`
- `docs/production_retrieval_trace.md`
- `docs/model_reviewed_ablation_results.md`

## Model-Reviewed Dataset

Command:

```bash
conda run -n rag python scripts/review_scholarfusion_eval_with_model.py \
  --db-path evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3 \
  --candidates evaluation_data/scholarfusion_lite_v1/candidates.jsonl \
  --output-dir evaluation_data/scholarfusion_model_reviewed_v1 \
  --max-items 48 \
  --max-requests 48 \
  --confidence-threshold 0.85 \
  --real-reviewer \
  --force
```

Result:

```json
{
  "source_candidate_count": 92,
  "selected_candidate_count": 39,
  "accepted_count": 38,
  "rejected_count": 1,
  "reviewer_model": "qwen-plus",
  "real_reviewer": true,
  "confidence_threshold": 0.85,
  "request_count": 39,
  "accepted_type_counts": {
    "text_fact": 6,
    "method_detail": 5,
    "text_to_figure": 5,
    "image_to_text": 5,
    "image_to_image": 4,
    "pair_to_any": 4,
    "unanswerable": 3,
    "cross_paper_comparison": 6
  }
}
```

Outputs:

- `evaluation_data/scholarfusion_model_reviewed_v1/raw_candidates.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/review_results.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/rejected.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/review_report.json`
- `evaluation_data/scholarfusion_model_reviewed_v1/spot_check.csv`

Spot-check rows: 8.

## Production Retrieval Trace

Command:

```bash
conda run -n rag python scripts/trace_scholarfusion_production_retrieval.py \
  --db-path evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3 \
  --index-db-path evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3 \
  --dataset evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl \
  --output evaluation_results/scholarfusion_phase21/production_retrieval_trace.json
```

Result:

```text
trace_count=5
fusion_method=WeightedRRF
retrievers configured include bm25, text_dense, multimodal_dense, page_visual, paper_level, graph
chroma_text_dense_enabled=false
chroma_text_dense_status=legacy_adapter_not_initialized
```

Important boundary: the trace validates the unified ScholarFusion path, but not the original Chroma text dense collection. The legacy adapter exists but was not initialized in this run.

## Real Reranker Smoke

Command:

```bash
conda run -n rag python scripts/run_scholarfusion_real_reranker_smoke.py \
  --db-path evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3 \
  --index-db-path evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3 \
  --dataset evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl \
  --output evaluation_results/scholarfusion_phase21/real_reranker_smoke.json
```

Result:

```json
{
  "requested_provider": "qwen3-vl-rerank",
  "request_count": 4,
  "fallback_count": 0,
  "rank_change_count": 4
}
```

The provider was actually called. Default tests still do not call this API.

## Ablation

Command:

```bash
conda run -n rag python scripts/run_scholarfusion_phase21_ablation.py \
  --db-path evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3 \
  --index-db-path evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3 \
  --dataset evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl \
  --output evaluation_results/scholarfusion_phase21/model_reviewed_ablation.json \
  --top-k 10
```

Result summary:

| Group | Recall@10 | MRR | Balanced Acc. | False Answer | False Refusal | P95 Retrieval ms |
|---|---:|---:|---:|---:|---:|---:|
| B0 Text Dense | 0.0857 | 0.0417 | 1.0000 | 0.0000 | 0.0000 | 19.542 |
| B1 BM25 | 0.0429 | 0.0208 | 1.0000 | 0.0000 | 0.0000 | 14.736 |
| B2 Text Dense + BM25 | 0.0571 | 0.0228 | 1.0000 | 0.0000 | 0.0000 | 30.367 |
| B3 + Text Reranker | 0.0571 | 0.0592 | 1.0000 | 0.0000 | 0.0000 | 30.865 |
| B4 + Evidence Bundle | 0.0571 | 0.0592 | 1.0000 | 0.0000 | 0.0000 | 30.994 |
| M1 + Multimodal Dense | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 | 45.711 |
| M2 + Real Page Visual | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 | 56.785 |
| M3 + Multimodal Reranker | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 | 56.796 |
| M4 + Evidence Verifier | 0.2286 | 0.2158 | 0.6571 | 0.0000 | 0.6857 | 56.345 |
| M5 + Graph Retrieval | 0.2286 | 0.2158 | 0.6571 | 0.0000 | 0.6857 | 56.379 |

Interpretation:

- Multimodal Dense improves Recall@10 and MRR on this model-reviewed provisional set.
- Page Visual adds latency in this small run but does not improve aggregate Recall@10.
- Reranker smoke shows rank changes, but the ablation's deterministic reranker does not improve aggregate retrieval metrics.
- Evidence Verifier eliminates false answers but causes high false refusal on this dataset.
- Graph Retrieval does not improve aggregate metrics in the current fixed sample.
- Primary citation precision remains 0 in the ablation, meaning QA citations are not yet aligned to the model-reviewed primary evidence fields.

## Tests

Command:

```bash
conda run -n rag python -m pytest -q
```

Result:

```text
24 passed, 1 skipped, 5 warnings
```

## Still Not Resume-Grade

- No human-verified Gold set yet.
- Citation Precision is not ready to claim.
- Original Chroma text dense path is not proven inside ScholarFusion Phase2.1 trace.
- Evidence Verifier requires calibration; current false refusal is too high.
- Some text_fact samples are still simple excerpt facts rather than deep scientific questions.
