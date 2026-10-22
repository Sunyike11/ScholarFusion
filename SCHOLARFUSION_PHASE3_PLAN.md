# ScholarFusion Phase 3 Plan

## Goal

Converge existing modules into a practical production path, calibrate answerability, and prepare small Gold evaluation without fabricating human labels.

## Steps

1. Build a Phase3 production retriever from existing research data:
   - Chroma dense over research text elements.
   - LlamaIndex BM25 over research text elements.
   - Existing multimodal/page/paper SQLite vectors.
   - Weighted RRF with primary-hit protection.
2. Add production trace script.
3. Add RRF debug script and verify B2 is not worse than B0.
4. Add query type breakdown.
5. Calibrate answerability thresholds and document selected policy.
6. Decide defaults:
   - multimodal dense on
   - Chroma dense on for text queries
   - BM25 auxiliary
   - page visual only image-heavy
   - graph off by default
7. Build `small_gold_v1` candidates from spot-check/model-reviewed data.
8. Do not mark `human_verified=true` unless human review fields are actually filled.
9. Run final ablation on available dataset:
   - true Gold if imported
   - otherwise model-reviewed provisional plus explicit blocked Gold note
10. Update docs and capability matrix.

## Non-goals

- No new database or vector store beyond existing Chroma/SQLite.
- No Neo4j/Qdrant/Milvus.
- No new multimodal model.
- No agent/planner/frontend expansion.
- No fake human Gold.
