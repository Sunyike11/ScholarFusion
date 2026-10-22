# API

## Existing APIs

Unchanged:

- `POST /api/docs/upload`
- `GET /api/docs/list`
- `POST /api/docs/reset`
- `POST /api/chat/chat`
- `POST /api/chat/chat/stream`
- `GET /api/chat/history`
- `POST /api/chat/clear`
- `/users/*`

## Research Paper APIs

### Upload paper

```http
POST /api/papers/upload
```

Multipart field:

- `file`: PDF

Returns:

- `paper_id`
- `title`
- `parser_status`
- `parse_quality`
- `element_count`

### List papers

```http
GET /api/papers
```

### Get paper and elements

```http
GET /api/papers/{paper_id}
```

### Get or update paper card

```http
GET /api/papers/{paper_id}/card
PUT /api/papers/{paper_id}/card
```

`PUT` body:

```json
{
  "fields": {
    "representation": "3D Gaussian Splatting"
  }
}
```

Manual fields are stored separately from auto-generated cards.

### List figures

```http
GET /api/papers/{paper_id}/figures
```

### Text search

```http
POST /api/research/search/text
```

```json
{
  "query": "SMPL Gaussian initialization",
  "top_k": 10,
  "filters": {
    "paper_ids": ["paper_x"],
    "years": [2025],
    "section_contains": "Method",
    "element_types": ["paragraph"],
    "body_only": true
  }
}
```

### Figure search baseline

```http
POST /api/research/search/figures
```

Uses caption and linked paragraph text only. It does not use image embeddings.

### Similar papers

```http
POST /api/research/search/similar-papers
```

### Compare papers

```http
POST /api/research/compare
```

### Research QA

```http
POST /api/research/qa
```

Returns:

- `answer`
- `answerable`
- `citations`
- `sources`
- `context_manifest`

### Research assets

```http
GET /api/research/assets/{paper_id}/{filename}
```

The route checks path traversal and never exposes server absolute paths.

### Tasks

```http
GET /api/research/tasks/{task_id}
```

Asynchronous task execution is not implemented in this MVP. The endpoint returns `not_found`.
