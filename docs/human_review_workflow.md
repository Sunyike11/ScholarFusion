# Human Review Workflow

ScholarFusion Lite evaluation uses automatic candidate generation followed by small human review.

## Generate Candidates

```bash
conda run -n rag python scripts/build_scholarfusion_lite_eval.py \
  --db-path evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3 \
  --output-dir evaluation_data/scholarfusion_lite_v1 \
  --target-count 100 \
  --force
```

Generated rows are always:

```text
review_status=pending_review
```

## Build Phase2 Review Queue

```bash
conda run -n rag python scripts/build_phase2_review_queue.py \
  --candidates evaluation_data/scholarfusion_lite_v1/candidates.jsonl \
  --output evaluation_data/scholarfusion_lite_v1/phase2_review_queue.csv
```

Current queue:

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

## Human Fields

The reviewer edits:

- `review_decision`
- `review_notes`
- reference answer or evidence fields if needed

Only explicit approvals are imported as `human_verified`.

## Import

```bash
conda run -n rag python scripts/import_human_review.py \
  --candidates evaluation_data/scholarfusion_lite_v1/candidates.jsonl \
  --review-csv evaluation_data/scholarfusion_lite_v1/phase2_review_queue.csv \
  --output evaluation_data/scholarfusion_lite_v1/candidates_with_human_review.jsonl
```

Rows without approval remain provisional.
