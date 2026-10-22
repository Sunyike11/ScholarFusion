# ScholarFusion Result

## Implemented In This Round

- Figure enrichment and deduplication metadata.
- Figure classification by deterministic rules.
- Deterministic Figure-Caption-Paragraph link evidence.
- Parsing quality CLI.
- Deterministic answerability verifier and claim-level citation support checks.
- SQLite-backed multimodal vector index with Chroma-compatible collection names.
- Text/Image/Text-Image Pair query support through unified 2560-d vectors.
- Weighted RRF hybrid retrieval over lexical, text dense, multimodal dense, page proxy and paper-level vectors.
- Deterministic multimodal reranker path with trace output.
- Evidence-aware QA refusal path.
- OCR/Layout fallback boundary and triggerable fallback elements.
- Lightweight SQLite knowledge graph.
- FastAPI endpoints for Any-to-Any, image/pair search, graph, related work and async task status.
- ScholarFusion Lite provisional evaluation dataset and runner.

## Real Commands And Results

```bash
conda run -n rag python -m pytest -q
```

Result: `20 passed, 1 skipped`.

```bash
conda run -n rag python scripts/run_scholarfusion_smoke.py --pdf-dir data/research_papers --max-papers 3 --max-requests 45 --output-dir evaluation_results/scholarfusion --force
```

Result:

- `paper_count=3`
- `index_counts.text=45`
- `index_counts.multimodal=40`
- `index_counts.page=27`
- `index_counts.paper=3`
- Text/Image/Pair Any-to-Any all returned ranked results.
- `qa_unanswerable_answerable=false`
- `graph_query_node_count=2`
- `related_work_citation_count=2`

```bash
conda run -n rag python scripts/run_scholarfusion_smoke.py --pdf-dir data/research_papers --max-papers 1 --max-requests 12 --output-dir evaluation_results/scholarfusion_real_provider --real-provider --force
```

Result:

- `provider=dashscope`
- `request_count=12`
- `index_counts.text=5`
- `index_counts.multimodal=4`
- `index_counts.page=2`
- `index_counts.paper=1`
- Image-to-any and Pair-to-any returned results.

```bash
conda run -n rag python scripts/evaluate_paper_parsing.py --db-path evaluation_results/scholarfusion/scholarfusion.sqlite3 --output evaluation_results/scholarfusion/parsing_quality.json
```

Result:

- `paper_count=3`
- `figure_count=394`
- `unique_figure_count=65`
- `duplicate_figure_rate=0.835`
- `figures_with_caption_rate=0.3385`
- `figures_with_linked_paragraph_rate=1.0`
- `elements_with_page_rate=1.0`
- `elements_with_bbox_rate=1.0`
- `parse_warning_count=1`

```bash
conda run -n rag python scripts/build_scholarfusion_lite_eval.py --db-path evaluation_results/scholarfusion/scholarfusion.sqlite3 --output-dir evaluation_data/scholarfusion_lite_v1 --target-count 60 --force
```

Result: `candidate_count=60`, `review_queue_count=40`, `dataset_status=provisional`.

```bash
conda run -n rag python scripts/run_scholarfusion_lite_evaluation.py --db-path evaluation_results/scholarfusion/scholarfusion.sqlite3 --index-db-path evaluation_results/scholarfusion/scholarfusion_index.sqlite3 --dataset evaluation_data/scholarfusion_lite_v1/candidates.jsonl --output evaluation_results/scholarfusion/evaluation_metrics.json
```

Provisional result:

- `not_gold=true`
- `item_count=60`
- `recall@5=0.2924`
- `recall@10=0.2966`
- `mrr=0.5179`
- `citation_precision=0.0904`
- `citation_recall=0.0678`
- `page_accuracy=1.0`
- `element_id_validity_rate=1.0`
- `answerable_accuracy=0.2167`
- `p50_retrieval_latency_ms=58.016`
- `p95_retrieval_latency_ms=61.074`

## Known Failures And Limits

- Supported real-paper QA is currently conservative; one smoke supported question still refused. This is safer than hallucination but needs better verifier calibration.
- Page visual retrieval currently uses page text proxy vectors, not rendered page-image vectors.
- Multimodal reranker is deterministic/listwise fallback, not a dedicated Qwen3-VL-Reranker weight.
- OCR/Layout fallback is triggerable and schema-compatible, but not a production OCR benchmark implementation.
- Paper Card remains deterministic/rule based.
- Evaluation data is provisional and not human Gold.
- ResearchService still does not directly reuse the legacy Chroma/BM25/reranker pipeline; it now has its own SQLite vector index and lexical baseline.

## Git Safety

No commit or push was performed. `.env`, PDFs, evaluation data and generated result files are ignored.
