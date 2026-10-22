# Final Ablation

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

## Results

| Experiment | Recall@5 | Recall@10 | MRR | nDCG@10 | False Answer | False Refusal | P95 Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Dense | 0.2143 | 0.2429 | 0.2551 | 0.2333 | 0.0 | 0.0 | 15.829 |
| B1 BM25 | 0.2000 | 0.2143 | 0.2222 | 0.2094 | 0.0 | 0.0 | 0.463 |
| B2 Hybrid | 0.2143 | 0.2429 | 0.2551 | 0.2333 | 0.0 | 0.0 | 18.384 |
| B3 + Text Reranker | 0.2429 | 0.2429 | 0.3092 | 0.2618 | 0.0 | 0.0 | 18.623 |
| M1 + Multimodal Dense | 0.2143 | 0.2429 | 0.2551 | 0.2333 | 0.0 | 0.0 | 18.294 |
| M2 + Page Visual | 0.2143 | 0.2429 | 0.2551 | 0.2333 | 0.0 | 0.0 | 29.935 |
| M3 + Multimodal Reranker | 0.2429 | 0.2429 | 0.3092 | 0.2618 | 0.0 | 0.0 | 30.228 |
| M4 + Evidence Verifier | 0.2429 | 0.2429 | 0.3092 | 0.2618 | 0.0 | 0.0 | 29.684 |

## Reading

- B2 no longer regresses relative to B0 on Recall@10.
- Text/multimodal reranking improves MRR from 0.2551 to 0.3092.
- Page Visual does not improve aggregate Recall@10 on this small provisional set and adds latency, so it is enabled only for image-heavy query types.
- Evidence Verifier keeps false-answer and false-refusal rates at 0.0 on model-reviewed provisional data.

## Boundary

These are not final resume metrics. They are reproducible Phase 3 provisional metrics and must be rerun after Small Gold approval.

## Small Gold Ablation

After explicit approval in `spot_check.csv`, `evaluation_data/small_gold_v1/gold_dataset.jsonl` contains 8 human-verified rows.

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

| Experiment | Recall@5 | Recall@10 | MRR | False Answer | False Refusal | P95 Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B0 Dense | 0.3333 | 0.4167 | 0.2991 | 0.0 | 0.0 | 20.703 |
| B1 BM25 | 0.2500 | 0.2500 | 0.2500 | 0.0 | 0.0 | 0.676 |
| B2 Hybrid | 0.3333 | 0.4167 | 0.2991 | 0.0 | 0.0 | 19.721 |
| B3 + Text Reranker | 0.4167 | 0.4167 | 0.4062 | 0.0 | 0.0 | 19.555 |
| M1 + Multimodal Dense | 0.3333 | 0.4167 | 0.2991 | 0.0 | 0.0 | 19.319 |
| M2 + Page Visual | 0.3333 | 0.4167 | 0.2991 | 0.0 | 0.0 | 31.577 |
| M3 + Multimodal Reranker | 0.4167 | 0.4167 | 0.4062 | 0.0 | 0.0 | 29.705 |
| M4 + Evidence Verifier | 0.4167 | 0.4167 | 0.4062 | 0.0 | 0.0 | 30.731 |

This confirms the Phase 3 pattern on human-verified samples: hybrid does not regress from dense, reranking improves MRR, and page visual adds latency without aggregate gain on this tiny set.
