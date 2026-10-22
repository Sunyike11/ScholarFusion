# Answerability Calibration

Phase 3 calibrates answerability to reduce the previous high false-refusal behavior.

## Policy

```json
{
  "minimum_primary_evidence": 1,
  "retrieval_threshold": 0.0,
  "rerank_threshold": 0.0,
  "verifier_threshold": 0.34,
  "entity_coverage_threshold": 0.0,
  "attribute_coverage_threshold": 0.0,
  "private_or_undisclosed_queries_refuse": true
}
```

The rule is intentionally simple: answer when retrieval evidence exists, but refuse private, undisclosed, or proprietary questions.

## Result

`evaluation_results/phase3/calibration.json` reports:

```json
{
  "dataset_status": "provisional",
  "review_status": "model_reviewed",
  "not_gold": true,
  "false_refusal_rate": 0.0,
  "false_answer_rate": 0.0,
  "balanced_accuracy": 1.0
}
```

The sample has 35 answerable examples and 3 unanswerable examples. This is not a human Gold calibration result.

## Reject Reasons

```json
{
  "has_retrieval_evidence": 35,
  "private_or_undisclosed": 3
}
```

## Boundary

The calibration is useful for production convergence but should be rechecked after human Gold review. It should not be tuned repeatedly against the same provisional set for resume numbers.
