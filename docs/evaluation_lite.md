# Evaluation Lite

ScholarFusion Lite evaluation uses provisional candidates by default.

Generated files:

- `evaluation_data/scholarfusion_lite_v1/candidates.jsonl`
- `evaluation_data/scholarfusion_lite_v1/review_queue.csv`

All generated candidates are `pending_review`. Metrics from these candidates are marked:

```json
{
  "not_gold": true,
  "dataset_status": "provisional"
}
```

Only human-reviewed candidates should be used for final resume metrics.
