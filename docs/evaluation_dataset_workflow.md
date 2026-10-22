# Evaluation Dataset Workflow

Phase 1.5 creates candidate evaluation data for human review. Automatically generated records are never marked as gold.

## Build Candidates

```bash
PYTHONPATH=. python scripts/build_research_eval_candidates.py \
  --output-dir evaluation_data/research_v1 \
  --force
```

Outputs:

- `evaluation_data/research_v1/candidates.jsonl`
- `evaluation_data/research_v1/review_template.csv`

## Candidate Schema

Each record contains:

- `query_id`
- `paper_ids`
- `query_type`
- `question`
- `reference_answer_candidate`
- `relevant_element_ids`
- `relevant_page_numbers`
- `answerable`
- `difficulty`
- `generation_model`
- `generation_timestamp`
- `review_status`

`review_status` is always `pending_review` for generated candidates.

## Validation Rules

- Every relevant element ID must exist.
- Element paper IDs must match `paper_ids`.
- Element pages must match `relevant_page_numbers`.
- Answerable records need at least one evidence element.
- Unanswerable records must not include evidence IDs.
- Duplicate questions are removed.
- Questions should not directly reveal the candidate answer.

## Provisional Evaluation

```bash
PYTHONPATH=. python scripts/run_research_mvp_evaluation.py \
  --dataset evaluation_data/research_v1/candidates.jsonl \
  --output-dir evaluation_results/research_mvp_eval
```

When records are not `human_verified`, output metrics include:

```json
{
  "not_gold": true
}
```

These metrics are for debugging only and must not be reported as final accuracy.
