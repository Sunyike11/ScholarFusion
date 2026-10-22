# Real Paper Validation

Phase 1.5 validates the research-paper MVP against real PDFs without pretending fixtures are real data.

## Input Directory

Default:

```bash
data/research_papers/
```

Run:

```bash
PYTHONPATH=. python scripts/smoke_research_mvp_real.py \
  --pdf-dir data/research_papers \
  --max-papers 5 \
  --output-dir evaluation_results/research_real_smoke \
  --json
```

If no PDF files exist, the script returns:

```json
{
  "status": "no_pdf_files",
  "paper_count": 0
}
```

It does not create synthetic papers.

## Summary Fields

Per paper, the smoke writes:

- `paper_id`
- `file_name`
- `page_count`
- `element_count`
- `paragraph_count`
- `heading_count`
- `figure_count`
- `caption_count`
- `table_count`
- `equation_count`
- `elements_with_page_count`
- `elements_with_bbox_count`
- `figures_with_caption_count`
- `figures_with_linked_paragraph_count`
- `parse_status`
- `parse_warnings`
- `ingestion_latency_ms`

The report stores IDs and aggregate counts only, not full paper text.

## Page Definition

Current page numbers are PDF physical pages starting from 1. Printed page labels in the paper are not inferred.

## Failure Policy

Parser warnings should remain explicit:

- `fitz_unavailable_text_fallback`
- `ocr_required_pages`
- `missing_page_numbers`
- `figures_without_caption`

These warnings should not be hidden by synthetic fallbacks.
