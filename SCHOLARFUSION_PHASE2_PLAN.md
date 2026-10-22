# ScholarFusion Phase 2 Plan

1. Add `HybridResearchRetriever` and route ResearchService search/QA through it.
2. Render page images and index them with real image embeddings.
3. Re-grade Figure links and restrict high-confidence evidence to explicit/caption-neighbor links.
4. Split citations into primary and context citations with claim support metadata.
5. Add `qwen3-vl-rerank` provider and provider selection.
6. Add LLM Paper Card builder with schema/evidence validation and cache.
7. Add human review queue/import and phase2 ablation runner.
8. Add index consistency checks.
9. Run default tests and small real smoke over current PDFs.
10. Document honest capability status and remaining limits.
