# Model-Reviewed Ablation Results

Dataset:

```text
scholarfusion_model_reviewed_v1
sample_count=38
dataset_status=provisional
review_status=model_reviewed
not_gold=true
```

## Summary

| Group | Recall@10 | MRR | Balanced Acc. | False Answer | False Refusal |
|---|---:|---:|---:|---:|---:|
| B0 Text Dense | 0.0857 | 0.0417 | 1.0000 | 0.0000 | 0.0000 |
| B1 BM25 | 0.0429 | 0.0208 | 1.0000 | 0.0000 | 0.0000 |
| B2 Text Dense + BM25 | 0.0571 | 0.0228 | 1.0000 | 0.0000 | 0.0000 |
| B3 + Text Reranker | 0.0571 | 0.0592 | 1.0000 | 0.0000 | 0.0000 |
| B4 + Evidence Bundle | 0.0571 | 0.0592 | 1.0000 | 0.0000 | 0.0000 |
| M1 + Multimodal Dense | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 |
| M2 + Real Page Visual | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 |
| M3 + Multimodal Reranker | 0.2286 | 0.2158 | 1.0000 | 0.0000 | 0.0000 |
| M4 + Evidence Verifier | 0.2286 | 0.2158 | 0.6571 | 0.0000 | 0.6857 |
| M5 + Graph Retrieval | 0.2286 | 0.2158 | 0.6571 | 0.0000 | 0.6857 |

## Observations

- Multimodal Dense is the main retrieval gain in this fixed provisional set.
- Page Visual currently adds latency but not aggregate Recall@10.
- Reranker real smoke changes rankings, but the ablation's deterministic default reranker does not improve aggregate Recall@10.
- Evidence Verifier is conservative: it reduces false answers but causes many false refusals.
- Graph Retrieval did not improve this fixed sample.
- Citation strict and alternative-aware precision remain 0 in ablation, so citation alignment is not yet solved.

## Artifact

Full JSON:

```text
evaluation_results/scholarfusion_phase21/model_reviewed_ablation.json
```
