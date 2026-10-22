# ScholarFusion Phase 2 Analysis

## Legacy Chroma/BM25/Reranker Entry

- Legacy ingestion is implemented in `core/ingestion.py`.
- ChromaDB is created through `chromadb.PersistentClient(AppSettings.CHROMA_PERSIST_DIR)` and collection `quickstart`.
- LlamaIndex `VectorStoreIndex` is built from pipeline nodes and persisted through Redis docstore/index store plus Chroma vector store.
- BM25 is initialized in `core/workflow.py::RAGWorkflow._initialize_bm25()`.
- Text reranker is `SentenceTransformerRerank(model=AppSettings.RERANK_MODEL_PATH)`.
- Query fusion is `QueryFusionRetriever(mode="reciprocal_rerank")`.
- The old path is active for normal chat, but ResearchService did not use it before this phase.

## ScholarFusion SQLite Index Entry

- `core/scholarfusion_index.py` stores vectors in SQLite.
- `core/scholarfusion_indexer.py` writes text, multimodal, page and paper-level collections.
- `core/scholarfusion_retrieval.py` performs Any-to-Any retrieval over the SQLite index plus lexical fallback.

## Merge Strategy

- Add `HybridResearchRetriever` as the single ResearchService retrieval entry.
- It attempts legacy production retrieval when a legacy app/workflow is available.
- It always adds ScholarFusion multimodal/page/paper/graph recall and fuses results with Weighted RRF.
- SQLite vector index remains test/fallback and multimodal store; it is no longer presented as the only production retrieval path.

## Citation Metrics

- Previous citation precision was computed by matching cited element IDs against provisional relevant IDs.
- This mixed ID validity with actual claim support and is unreliable because provisional labels are incomplete.
- Phase 2 separates Citation ID Validity, Page Accuracy, Evidence Precision/Recall, and Claim Support.

## Figure Link Rules

- Current parser links captions by nearest same-page caption and paragraphs by mentions/neighbors.
- `figures_with_linked_paragraph_rate=1.0` is caused by broad same-page neighbor fallback.
- Phase 2 grades links: `explicit_reference`, `caption_neighbor`, `same_page_neighbor`, `same_section_fallback`.

## Page Visual Status

- Previous page visual index used page text proxy vectors.
- Phase 2 renders PDF pages to PNG and embeds page images with `qwen3-vl-embedding`.

## Reranker Status

- Previous reranker was deterministic fallback.
- Official DashScope docs expose `qwen3-vl-rerank` through `/api/v1/services/rerank/text-rerank/text-rerank`.
- Phase 2 adds real provider and keeps deterministic fallback for tests.

## Provisional Eval Generation

- Existing candidates are generated from elements, figures and paper pairs.
- `relevant_element_ids` are auto-populated from source element/link IDs and are not human Gold.
- Metrics are comparable only within the same generated dataset and should remain `not_gold=true`.

## Comparable Results

- Text/image/pair Any-to-Any results are comparable within ScholarFusion smoke DB.
- B0-B4 legacy Chroma/BM25 numbers are only comparable when the same research papers have been ingested into the legacy pipeline.
- Current provisional metrics cannot be used as final resume numbers.

## Files Planned

- `core/hybrid_research_retriever.py`
- `core/page_visual.py`
- `core/phase2_citations.py`
- `core/paper_card_llm.py`
- `scripts/import_human_review.py`
- `scripts/check_research_index_consistency.py`
- `scripts/run_phase2_ablation.py`
- `scripts/build_phase2_review_queue.py`
- docs and tests.

## Rollback

- New Phase 2 functionality is additive.
- ResearchService can fall back to the existing SQLite/lexical path if legacy retrieval or real providers are unavailable.
