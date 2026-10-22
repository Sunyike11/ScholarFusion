# ScholarFusion Plan

## Goal

Build a minimal but real vertical loop:

PDF parse -> figure dedup/linking -> multimodal embedding/index -> Any-to-Any retrieval -> rerank -> evidence verifier -> cited QA -> graph/query -> provisional evaluation.

## Implementation Order

1. Parser quality
   - Add figure metadata enrichment, hash/perceptual hash, decorative filtering and duplicate marking.
   - Add parsing quality CLI.

2. Evidence verifier
   - Add deterministic answerability verification.
   - Add claim-level citation support checks.
   - Wire verifier into `ResearchService.qa()`.

3. Multimodal index and Any-to-Any
   - Add SQLite-backed vector index with Chroma-style collection names.
   - Index paragraphs, figures, figure-text pairs, pages and paper summaries.
   - Support fake provider by default and real provider by explicit flag.
   - Add Weighted RRF and query router.

4. Reranking
   - Add deterministic multimodal reranker and optional VL listwise boundary.
   - Do not claim dedicated Qwen3-VL-Reranker unless such a provider is configured and real-tested.

5. OCR/Layout fallback
   - Add `PaperParser`, `NativePDFParser`, and `OCRLayoutFallbackParser` boundary.
   - Implement a simple page-render/VL-or-text fallback trigger path for low-quality pages and test it with a fixture.

6. Graph
   - Add SQLite graph nodes/edges/evidence.
   - Build graph from paper cards and figures.
   - Support graph query and related-work summary based on graph paths and citations.

7. Async ingestion
   - Add BackgroundTasks ingestion endpoint and task status store.
   - Use local task store with Redis-compatible boundary.

8. Evaluation and docs
   - Generate ScholarFusion Lite candidates and review queue.
   - Run provisional benchmark.
   - Write result docs and capability matrix.

## Non-goals

- No Neo4j, Milvus, Qdrant, Celery or UI rewrite.
- No hidden real-provider calls in default tests.
- No fabricated Gold metrics.
