# ScholarFusion Phase 2.1 Plan

## Scope

This phase validates and evaluates the existing ScholarFusion production path. It does not add new APIs, databases, vector stores, frontends or large OCR changes.

## Steps

1. Freeze a model-reviewed provisional dataset under `evaluation_data/scholarfusion_model_reviewed_v1/`.
2. Add independent reviewer classes and a reviewer script.
3. Run deterministic validation after model review.
4. Keep accepted rows as `review_status=model_reviewed`, `dataset_status=provisional`, `not_gold=true`, `human_verified=false`.
5. Generate a 6-8 row spot-check CSV for later human inspection.
6. Add production retrieval trace script with safe retriever/collection/provider trace.
7. Add real reranker smoke script gated by explicit command.
8. Add B0-M5 ablation runner over the fixed model-reviewed dataset.
9. Add strict and alternative-aware citation metrics plus answerability confusion metrics.
10. Write result docs and update the capability matrix.

## Constraints

- Reviewer never receives system answers, system retrieval results, ranking or metrics.
- Reviewer input is limited to question, candidate answer, candidate answerable flag, paper IDs and evidence previews.
- Model-reviewed data is not human verified.
- Default pytest must not call paid APIs.
- Real provider runs are explicit scripts and record actual provider or fallback.

## Expected Outputs

- `evaluation_data/scholarfusion_model_reviewed_v1/raw_candidates.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/review_results.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/rejected.jsonl`
- `evaluation_data/scholarfusion_model_reviewed_v1/review_report.json`
- `evaluation_data/scholarfusion_model_reviewed_v1/spot_check.csv`
- `evaluation_results/scholarfusion_phase21/production_retrieval_trace.json`
- `evaluation_results/scholarfusion_phase21/real_reranker_smoke.json`
- `evaluation_results/scholarfusion_phase21/model_reviewed_ablation.json`
