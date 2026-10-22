# Page Visual Retrieval

Phase2 replaces the previous page text proxy with rendered page images for PDF-backed papers.

## Pipeline

```text
PDF
-> PageVisualRenderer
-> stable PNG: {asset_root}/{paper_id}/pages/page_0001.png
-> qwen3-vl-embedding image vector
-> research_page_visual collection
```

## Metadata

Each page vector stores:

- `paper_id`
- `page_number`
- `page_image_path`
- `page_hash`
- `input_modality=page_image`
- `model`
- `dimension`
- `embedding_version=scholarfusion-page-visual-v2`

## Validation

Phase2 smoke indexed:

```text
research_page_visual=27
input_modality=page_image
page_image_path present
stale_page_image_count=0
```

## Fallback

If the source PDF is unavailable, the indexer returns a clearly marked `page_proxy:{paper_id}:{page}` fallback so unit tests can still run. This fallback must not be reported as real page visual retrieval.
