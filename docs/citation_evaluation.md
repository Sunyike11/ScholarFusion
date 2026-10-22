# Citation Evaluation

Phase2 separates citation validity from evidence correctness.

## Output Types

- `primary_citations`: evidence that directly supports answer claims.
- `context_citations`: surrounding section, figure, caption or neighbor context.

Only primary citations should be used for citation evidence precision.

## Metrics

The evaluation runner reports:

- `citation_precision` and `citation_recall`: legacy compatibility over all citations.
- `primary_citation_precision` and `primary_citation_recall`: direct evidence only.
- `citation_id_validity`: whether cited IDs exist.
- `citation_page_accuracy`: whether cited page metadata exists.
- `citation_evidence_precision` and `citation_evidence_recall`: primary evidence against labeled evidence.
- `claim_support_rate`: fraction of generated claims marked supported by the deterministic verifier.

## Current Phase2 Result

The current dataset is provisional, not human verified:

```text
item_count=92
not_gold=true
primary_citation_precision=0.0112
citation_page_accuracy=1.0
claim_support_rate=0.9447
```

The low evidence precision is not yet a final system quality number. The auto-generated labels often include only one narrow evidence ID while QA may cite a valid but unlabeled neighboring paragraph or caption.

## Next Step

Use `evaluation_data/scholarfusion_lite_v1/phase2_review_queue.csv` to add human decisions, then import verified rows before reporting resume-grade citation precision.
