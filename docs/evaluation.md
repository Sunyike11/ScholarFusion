# Evaluation

## Dataset Item Format

```json
{
  "query_id": "qa_001",
  "query_type": "text_fact",
  "input_modality": "text",
  "target_modality": ["text"],
  "question": "...",
  "reference_answer": "...",
  "relevant_element_ids": ["..."],
  "answerable": true,
  "difficulty": "easy",
  "review_status": "draft"
}
```

Supported query types:

- `text_fact`
- `cross_section`
- `text_to_figure`
- `cross_paper_comparison`
- `unanswerable`

Do not mark automatically generated questions as `human_verified`.

## Implemented Metrics

Retrieval:

- Recall@5
- Recall@10
- Precision@K
- MRR
- nDCG@K

Citation:

- Citation Precision
- Citation Recall
- Page Accuracy
- Element ID Validity Rate

Generation basics:

- Answerable/Unanswerable Accuracy

System metadata:

- timestamp
- git commit when available
- caller-provided model, embedding, reranker, top-k and chunk configuration metadata

## Experiment Groups

The code records metric-compatible outputs for:

- `B0`: vector only
- `B1`: BM25 only
- `B2`: vector + BM25/RRF
- `B3`: vector + BM25/RRF + reranker
- `B4`: B3 + structured elements + evidence bundle

Only the structured-element MVP runner is implemented in code. The other groups are documented experiment labels for future runs using the existing RAG pipeline.
