# qwen3-vl-embedding Integration

Phase 1.5 adds a minimal smoke adapter for DashScope `qwen3-vl-embedding`. It does not replace the production text RAG index.

## Configuration

```bash
DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL=qwen3-vl-embedding
DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION=256
DASHSCOPE_REQUEST_TIMEOUT=60
```

Use the international endpoint only when the current DashScope region/account requires it.

## Provider Classes

- `MultimodalEmbeddingProvider`
- `FakeMultimodalEmbeddingProvider`
- `DashScopeQwen3VLEmbeddingProvider`

Supported methods:

- `encode_text(text)`
- `encode_image(image)`
- `encode_text_image_pair(text, image)`

## Fusion Rule

Text-image pair embeddings call the provider with `enable_fusion=true`. The implementation does not add or average text and image vectors manually.

## Smoke Command

Offline fake provider:

```bash
PYTHONPATH=. python scripts/smoke_qwen3_vl_embedding.py \
  --output-dir evaluation_results/qwen3_vl_embedding_smoke
```

Real provider:

```bash
PYTHONPATH=. python scripts/smoke_qwen3_vl_embedding.py \
  --real-provider \
  --output-dir evaluation_results/qwen3_vl_embedding_smoke
```

Real provider calls require `DASHSCOPE_API_KEY` and a valid multimodal embedding endpoint. The script records vector dimension, latency, request count, success count, failure count, and ID-only sample rankings.

## Security

- API keys are read from environment variables only.
- API key values are never printed.
- Image Base64 payloads are never logged.
- Unit tests use the fake provider by default.

## Current Status

In this workspace, the fake smoke produced `no_vectors` because no real papers or figures were ingested yet. The real DashScope provider has not been executed because the repository root did not contain a readable `.env` file at validation time.

## Reference

Alibaba Cloud Model Studio documents that multimodal fused vectors use DashScope's multimodal embedding API, and `qwen3-vl-embedding` generates fused vectors by setting `enable_fusion=true`.
