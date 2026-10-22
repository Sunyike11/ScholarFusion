# Research RAG MVP Architecture

## Scope

This MVP upgrades the project from generic document QA toward a structured research-paper reading backend. It keeps the existing LlamaIndex text RAG path intact and adds a parallel structured-paper layer.

## Data Flow

```text
POST /api/papers/upload
  -> ResearchService.upload_pdf_bytes
  -> ResearchPDFParser
      -> native PyMuPDF extraction if fitz is installed
      -> text fallback when native parser is unavailable
      -> parse quality status: native / ocr_required / parse_failed
  -> ResearchStore SQLite persistence
      -> papers
      -> elements
      -> cards
      -> ingestion_runs
  -> heuristic Paper Card generation with evidence refs
```

Search and QA:

```text
query
  -> ResearchRetriever
      -> structured Text-to-Text lexical baseline
      -> Text-to-Figure baseline over captions and linked paragraphs
  -> EvidenceBundleBuilder
      -> dedupe
      -> per-paper cap
      -> character budget
      -> linked figure/caption/neighbor expansion
  -> CitationValidator
  -> structured answer + citations
```

## Storage

- ChromaDB remains the vector store for the original RAG chain.
- Redis remains the LlamaIndex docstore, index store and ingestion cache.
- SQLite stores research paper metadata, structured elements and paper cards.
- Figure files are stored under `Settings.RESEARCH_ASSET_DIR` and exposed through `/api/research/assets/{paper_id}/{filename}`.

Redis is not used as a business task-state store in this MVP.

## Compatibility

Existing endpoints remain:

- `/api/docs/*`
- `/api/chat/*`
- `/users/*`

New research endpoints are mounted under:

- `/api/papers/*`
- `/api/research/*`

The old `sources` field remains available in the generic chat API. New research QA returns structured citations.

## Multimodal Extension Point

The MVP reserves `representation_ids` on paper elements and exposes figure ids and image URLs. It does not generate image embeddings or use a multimodal reranker.
