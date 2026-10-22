# Small Gold Evaluation

Phase 3 adds a strict Small Gold builder, but it does not fabricate human labels.

## Current Status

`evaluation_data/small_gold_v1/small_gold_report.json` reports:

```json
{
  "status": "ok",
  "sample_count": 8,
  "candidate_count": 30,
  "human_verified": true
}
```

Eight rows in `evaluation_data/scholarfusion_model_reviewed_v1/spot_check.csv` have explicit `approved` status, so they are exported as Small Gold.

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

## Files

- `evaluation_data/small_gold_v1/gold_dataset.jsonl`
- `evaluation_data/small_gold_v1/candidates_pending_human.jsonl`
- `evaluation_data/small_gold_v1/small_gold_report.json`

## Approval Rule

Rows become Gold only when `spot_check_status` or `review_decision` is explicitly one of:

```text
approve, approved, yes, y, true, pass, human_verified
```

Then the builder sets:

```json
{
  "human_verified": true,
  "review_status": "human_verified",
  "dataset_status": "gold",
  "not_gold": false
}
```

## Next Step

To strengthen this beyond a smoke-level Gold set, approve 20-30 rows in the spot-check CSV, then rerun:

```bash
conda run -n rag env PYTHONPATH=. python scripts/build_small_gold_v1.py
conda run -n rag env PYTHONPATH=. python scripts/run_phase3_final_ablation.py --dataset evaluation_data/small_gold_v1/gold_dataset.jsonl --output evaluation_results/phase3/final_ablation_gold.json
```

The current 8-row Gold result is valid as a small smoke check, but it is still too small to use as a stable headline metric.
