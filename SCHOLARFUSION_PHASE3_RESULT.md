# ScholarFusion Phase 3 Result

Phase name: Production Retrieval Convergence + Calibration + Small Gold Evaluation

## Summary

Phase 3 connected the production validation retrieval path around Chroma dense, LlamaIndex BM25, multimodal dense, optional page visual retrieval, protected Weighted RRF, reranking, and calibrated answerability. The implementation is intentionally conservative: it does not introduce new databases, vector stores, agents, planners, or frontend work.

Small Gold evaluation is now available with 8 explicitly approved human-verified rows. This is enough for a smoke-level Gold check, but still below the original 20-30 row target for stable resume metrics.

## Changed Files

- `core/phase3_production_retrieval.py`
- `core/scholarfusion_index.py`
- `scripts/trace_phase3_production_path.py`
- `scripts/debug_rrf_pipeline.py`
- `scripts/calibrate_answerability.py`
- `scripts/run_phase3_final_ablation.py`
- `scripts/query_type_breakdown.py`
- `scripts/build_small_gold_v1.py`
- `tests/test_scholarfusion_phase3.py`
- `SCHOLARFUSION_PHASE3_ANALYSIS.md`
- `SCHOLARFUSION_PHASE3_PLAN.md`
- `docs/production_convergence.md`
- `docs/calibration.md`
- `docs/query_breakdown.md`
- `docs/small_gold_evaluation.md`
- `docs/final_ablation.md`
- `README.md`
- `RESUME_CAPABILITY_MATRIX.md`

## Production Path

The Phase 3 trace validates this path:

```text
query
→ multimodal dense SQLite index
→ Chroma dense collection: phase3_research_text_dense
→ LlamaIndex BM25 over structured paper elements
→ optional page visual retrieval for image-heavy query types
→ protected Weighted RRF
→ optional reranker
→ calibrated answerability verifier
```

Trace output:

```json
{
  "trace_count": 5,
  "chroma_collection": "phase3_research_text_dense",
  "chroma_count": 45,
  "bm25_enabled": true,
  "bm25_node_count": 615
}
```

For text queries, the trace invokes `multimodal_dense`, `chroma_dense`, and `bm25`. For `text_to_figure`, `image_to_text`, and `pair_to_any`, it also invokes `page_visual`.

## Fusion Debug

The previous abnormal behavior was `B2 < B0`. Phase 3 uses protected Weighted RRF so auxiliary retrievers cannot erase strong primary dense hits from the final top-k.

Result:

```json
{
  "dense_recall@10": 0.24285714285714285,
  "hybrid_recall@10": 0.24285714285714285,
  "hybrid_not_worse_than_dense": true
}
```

This satisfies the convergence target `Recall@10(B2) >= Recall@10(B0)`.

## Calibration

The calibrated answerability policy keeps retrieval thresholds permissive and refuses private or undisclosed questions explicitly:

```json
{
  "minimum_primary_evidence": 1,
  "retrieval_threshold": 0.0,
  "rerank_threshold": 0.0,
  "verifier_threshold": 0.34,
  "private_or_undisclosed_queries_refuse": true
}
```

On the model-reviewed provisional set:

```json
{
  "false_refusal_rate": 0.0,
  "false_answer_rate": 0.0,
  "balanced_accuracy": 1.0
}
```

This is not a human Gold result. The sample contains 38 model-reviewed provisional items.

## Final Ablation

Dataset:

```json
{
  "dataset_status": "provisional",
  "review_status": "model_reviewed",
  "not_gold": true,
  "human_verified": false,
  "sample_count": 38
}
```

| Experiment | Recall@10 | MRR | False Answer | False Refusal | P95 Latency ms |
|---|---:|---:|---:|---:|---:|
| B0 Dense | 0.2429 | 0.2551 | 0.0 | 0.0 | 15.829 |
| B1 BM25 | 0.2143 | 0.2222 | 0.0 | 0.0 | 0.463 |
| B2 Hybrid | 0.2429 | 0.2551 | 0.0 | 0.0 | 18.384 |
| B3 + Text Reranker | 0.2429 | 0.3092 | 0.0 | 0.0 | 18.623 |
| M1 + Multimodal Dense | 0.2429 | 0.2551 | 0.0 | 0.0 | 18.294 |
| M2 + Page Visual | 0.2429 | 0.2551 | 0.0 | 0.0 | 29.935 |
| M3 + Multimodal Reranker | 0.2429 | 0.3092 | 0.0 | 0.0 | 30.228 |
| M4 + Evidence Verifier | 0.2429 | 0.3092 | 0.0 | 0.0 | 29.684 |

Conclusion: reranking improves MRR on this provisional set. Page Visual increases latency without improving aggregate Recall@10, so it should stay dynamically enabled for image-heavy queries rather than default-on for all queries. Graph retrieval remains off by default for Phase 3.

## Small Gold Status

Command output:

```json
{
  "status": "ok",
  "sample_count": 8,
  "candidate_count": 30,
  "human_verified": true
}
```

Generated files:

- `evaluation_data/small_gold_v1/gold_dataset.jsonl`
- `evaluation_data/small_gold_v1/candidates_pending_human.jsonl`
- `evaluation_data/small_gold_v1/small_gold_report.json`

`gold_dataset.jsonl` contains 8 human-verified rows approved from `spot_check.csv`.

Distribution:

```text
text_fact: 1
text_to_figure: 1
image_to_text: 1
image_to_image: 1
pair_to_any: 1
cross_paper_comparison: 1
unanswerable: 2
```

## Small Gold Ablation

Dataset:

```json
{
  "dataset_status": "gold",
  "review_status": "human_verified",
  "not_gold": false,
  "human_verified": true,
  "sample_count": 8
}
```

| Experiment | Recall@10 | MRR | False Answer | False Refusal | P95 Latency ms |
|---|---:|---:|---:|---:|---:|
| B0 Dense | 0.4167 | 0.2991 | 0.0 | 0.0 | 20.703 |
| B1 BM25 | 0.2500 | 0.2500 | 0.0 | 0.0 | 0.676 |
| B2 Hybrid | 0.4167 | 0.2991 | 0.0 | 0.0 | 19.721 |
| B3 + Text Reranker | 0.4167 | 0.4062 | 0.0 | 0.0 | 19.555 |
| M1 + Multimodal Dense | 0.4167 | 0.2991 | 0.0 | 0.0 | 19.319 |
| M2 + Page Visual | 0.4167 | 0.2991 | 0.0 | 0.0 | 31.577 |
| M3 + Multimodal Reranker | 0.4167 | 0.4062 | 0.0 | 0.0 | 29.705 |
| M4 + Evidence Verifier | 0.4167 | 0.4062 | 0.0 | 0.0 | 30.731 |

## Test Results

Commands:

```bash
conda run -n rag env PYTHONPATH=. python -m py_compile core/phase3_production_retrieval.py scripts/trace_phase3_production_path.py scripts/debug_rrf_pipeline.py scripts/calibrate_answerability.py scripts/run_phase3_final_ablation.py scripts/query_type_breakdown.py scripts/build_small_gold_v1.py
conda run -n rag env PYTHONPATH=. pytest -q tests/test_scholarfusion_phase3.py
conda run -n rag env PYTHONPATH=. pytest -q
```

Results:

```text
3 passed, 1 warning
27 passed, 1 skipped, 6 warnings
```

## Honest Boundaries

- The main 38-row ablation remains model-reviewed provisional.
- Small Gold exists with 8 human-verified rows, which is useful for smoke-level validation but too small for strong resume metrics.
- Citation precision is still not resume-grade because this Phase 3 ablation uses retrieval/evidence labels, not full generated answer citation judging.
- Chroma/BM25 are now truly invoked in the Phase 3 production validation path, but Chroma is rebuilt from the structured ScholarFusion corpus because the original persisted legacy Chroma/BM25 directories were not available.
- Page Visual is real rendered-page vector retrieval, but it is useful only for image-heavy queries in the current small dataset.
- Graph retrieval remains a capability, not a default production component.
