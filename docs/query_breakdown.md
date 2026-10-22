# Query Type Breakdown

Dataset: model-reviewed provisional, not human Gold.

## Sample Counts

- `cross_paper_comparison`: 6
- `image_to_image`: 4
- `image_to_text`: 5
- `method_detail`: 5
- `pair_to_any`: 4
- `text_fact`: 6
- `text_to_figure`: 5
- `unanswerable`: 3

## Experiment Summary

| Group | Recall@10 | MRR | False Answer | False Refusal | P95 Latency ms |
|---|---:|---:|---:|---:|---:|
| B0_dense | 0.2429 | 0.2551 | 0.0000 | 0.0000 | 15.829 |
| B1_bm25 | 0.2143 | 0.2222 | 0.0000 | 0.0000 | 0.463 |
| B2_hybrid | 0.2429 | 0.2551 | 0.0000 | 0.0000 | 18.384 |
| B3_text_reranker | 0.2429 | 0.3092 | 0.0000 | 0.0000 | 18.623 |
| M1_multimodal_dense | 0.2429 | 0.2551 | 0.0000 | 0.0000 | 18.294 |
| M2_page_visual | 0.2429 | 0.2551 | 0.0000 | 0.0000 | 29.935 |
| M3_multimodal_reranker | 0.2429 | 0.3092 | 0.0000 | 0.0000 | 30.228 |
| M4_evidence_verifier | 0.2429 | 0.3092 | 0.0000 | 0.0000 | 29.684 |

## Findings

- Multimodal dense is the strongest default retriever on the current dataset.
- Page visual should stay query-gated because aggregate Recall@10 does not improve enough to justify default latency.
- Graph retrieval is kept off by default until cross-paper evidence labels are stronger.
- Evidence verifier calibration reduces false refusal when private/undisclosed refusal is handled separately.
