# ScholarFusion Architecture

ScholarFusion is an incremental research-paper RAG layer on top of the existing FastAPI project.

Core flow:

```text
PDF upload
-> native PyMuPDF parse
-> OCR/Layout fallback boundary
-> structured PaperElement storage
-> Figure enrichment/dedup/linking
-> Paper Card
-> multimodal embeddings
-> SQLite vector index collections
-> Any-to-Any retrieval
-> rerank
-> evidence verifier
-> cited QA / graph / related work
```

The current vector store is SQLite-backed for portability. Collection names match the planned Chroma collections:

- `research_text_elements`
- `research_multimodal_elements`
- `research_page_visual`
- `research_paper_level`

This keeps the demo runnable without adding Milvus, Qdrant or Neo4j.
