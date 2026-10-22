# ScholarFusion Analysis

## Current Real Data Flow

- Upload/API path: `app/routers/research.py` calls `ResearchService.upload_pdf_bytes()`.
- Parser: `ResearchPDFParser` uses PyMuPDF (`fitz`) when available and stores `PaperElement` rows in SQLite through `ResearchStore`.
- Store: `ResearchStore` persists papers, elements, cards and ingestion runs in `file/research/research_papers.sqlite3`.
- Retrieval: `ResearchService` currently uses `ResearchRetriever`, a structured lexical baseline over SQLite elements.
- QA: `ResearchService.qa()` retrieves text hits, builds an `EvidenceBundle`, creates evidence-snippet answers and validates citation IDs.
- Multimodal embedding: `DashScopeQwen3VLEmbeddingProvider` can call `qwen3-vl-embedding`; text/image/pair real smoke has passed in this repo.

## Figure Source And Duplicate Cause

- Native PyMuPDF returns image blocks from page dictionaries.
- Research parser currently converts every image block into a `figure` element and links same-page nearest caption.
- Some papers contain many small image blocks or subfigure fragments, so a single logical Figure can become many `figure` elements.
- Caption reuse causes many image blocks to point to the same caption, inflating `figure_count`.
- There is no image hash, perceptual hash, area filter, bbox overlap merge, duplicate flag, or decorative filter yet.

## Paper Card

- `ResearchService.generate_card()` is deterministic and keyword/rule based.
- It does not yet call DashScope text generation.
- Evidence refs are attached when a rule finds a relevant element.

## Chroma/BM25/Reranker Usage

- Legacy RAG code has Chroma/BM25/reranker concepts, but `ResearchService` does not call them.
- Research retrieval is currently SQLite lexical scoring only.
- This round will add a local multimodal vector index with Chroma-compatible collection names and metadata, without replacing the old RAG path.

## Redis

- Current research path does not use Redis.
- Existing project uses Redis for LlamaIndex stores/cache in the legacy path.
- This round uses a local SQLite-backed task status fallback and optional Redis adapter boundary, avoiding a hard dependency in tests.

## Multimodal Provider

- `core/research_multimodal_embedding.py` supports fake and real providers.
- Real provider uses `qwen3-vl-embedding`.
- Fusion uses `enable_fusion=true`.
- Validated real dimension: `2560`.

## Real PDF And Figure Data

- Real papers are under `data/research_papers/`.
- Extracted images are under `file/research/assets/{paper_id}/`.
- Local evaluation artifacts are under `evaluation_results/` and ignored by git.

## Citations, Refusal, Evaluation

- `CitationValidator` checks element existence and paper ID match.
- Current QA does not verify answerability beyond “some hits exist,” which caused an unanswerable real query to be answered.
- Evaluation runner supports provisional metrics and marks non-gold data as `not_gold=true`.

## Files To Modify/Add

- `core/research_pdf_parser.py`
- `core/research_store.py`
- `core/research_context.py`
- `core/research_service.py`
- `core/research_retrieval.py`
- `app/routers/research.py`
- `app/schemas_research.py`
- `config/settings.py`
- New ScholarFusion modules under `core/`
- New scripts under `scripts/`
- New docs and tests.

## API Compatibility

- Existing Research API endpoints remain available.
- New endpoints are additive under `/api/research/...`.
- Existing response fields such as `sources` remain; new structured fields are additive.

## Resume Capability Evidence

- Implemented code and test evidence will be mapped in `RESUME_CAPABILITY_MATRIX.md`.
- Each capability will be marked honestly as real-tested, fake-only, partial, or not implemented.
