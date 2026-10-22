# Model-Reviewed Evaluation Protocol

This protocol creates a provisional, model-reviewed benchmark. It is not human Gold.

## Labels

Every accepted item must keep:

```text
review_status=model_reviewed
dataset_status=provisional
not_gold=true
human_verified=false
```

## Separation

The reviewer receives:

- question
- candidate reference answer
- candidate answerable flag
- paper IDs
- candidate evidence IDs
- evidence previews
- page metadata
- element types

The reviewer does not receive:

- system retrieval results
- system answers
- rankings
- metrics
- generator reasoning

## Reviewer

Current real reviewer:

```text
qwen-plus
temperature=0.0
confidence_threshold=0.85
```

The reviewer returns fixed JSON with approve/revise/reject, confidence, corrected fields, evidence IDs, reason codes and a short safe summary.

## Deterministic Validation

After model review, the validator checks:

- query ID
- paper IDs
- element IDs
- evidence paper consistency
- primary evidence for answerable rows
- figure requirements
- cross-paper coverage
- unanswerable consistency
- duplicate questions
- answer leakage

Failed rows go to `rejected.jsonl`.

## Outputs

- `raw_candidates.jsonl`
- `review_results.jsonl`
- `provisional.jsonl`
- `rejected.jsonl`
- `review_report.json`
- `spot_check.csv`

The spot-check CSV is for later human inspection and does not block model-reviewed provisional evaluation.
