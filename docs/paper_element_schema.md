# Paper Element Schema

## PaperDocument

```json
{
  "paper_id": "paper_<sha16>",
  "metadata": {},
  "elements": [],
  "markdown_debug": "",
  "created_at": "...",
  "updated_at": "..."
}
```

`paper_id` is generated from PDF bytes, so the same file uploaded from a different temporary path keeps the same id.

## PaperMetadata

Fields:

- `title`
- `authors`
- `year`
- `venue`
- `source_file`
- `file_sha256`
- `parser_status`: `native`, `ocr_required`, or `parse_failed`
- `parser_name`
- `parse_quality`

## PaperElement

Required fields:

- `element_id`
- `paper_id`
- `element_type`
- `page_number`
- `section_path`
- `text`
- `bbox`
- `reading_order`
- `parent_id`
- `linked_element_ids`
- `source_file`
- `parser_name`
- `parse_confidence`
- `metadata`
- `representation_ids`

Supported `element_type` values:

- `title`
- `abstract`
- `heading`
- `paragraph`
- `figure`
- `figure_caption`
- `table`
- `table_caption`
- `equation`
- `reference`
- `page`

## Figure Metadata

Figure elements use `element_type="figure"` and include:

- `image_path`
- `image_url`
- `caption`
- `caption_element_id`
- `mentioned_by_element_ids`
- `neighbor_element_ids`
- `link_status`
- `figure_label`

When no native image block is available but a caption is detected, the parser creates a low-confidence synthetic figure with `synthetic_from_caption=true`.

## EvidenceRef

```json
{
  "paper_id": "...",
  "element_id": "...",
  "page_number": 5,
  "section_path": ["3 Method"],
  "element_type": "paragraph",
  "evidence_text": "...",
  "figure_id": null
}
```

Paper card fields and QA citations use this schema.
