# ScholarFusion

面向科研论文的检索增强问答系统，当前聚焦论文文档解析、混合检索、证据约束回答与 FastAPI 服务化；后续将扩展为支持文本、图像和图文对查询的 Any-to-Any 多模态科研助手。

ScholarFusion 的目标场景是 3D Gaussian Splatting、人体重建与计算机视觉论文阅读：帮助研究者围绕论文正文、方法描述、架构图、实验表格和跨论文对比进行可追溯问答。

## Current Capabilities

- **论文与文档摄取**：支持 PDF、TXT、DOCX、Markdown 上传，完成解析、切分、向量化与索引构建。
- **混合检索**：结合向量检索与 BM25 稀疏检索，并通过 SentenceTransformer reranker 对候选片段重排序。
- **证据约束问答**：在非流式 RAG 链路中执行 deterministic evidence verification，对缺少证据支持的回答进行保守改写或拒答。
- **引用返回**：回答与检索来源绑定，返回文件名、页码、chunk id 和内容片段，便于人工复核。
- **流式问答 API**：基于 FastAPI 提供非流式与 SSE 流式问答接口。
- **知识库管理**：支持文档上传、文档列表、索引重置、会话历史和用户鉴权。
- **工程化存储**：使用 ChromaDB 存储向量索引，Redis 存储文档、索引和缓存状态。

## Roadmap

Coming soon:

- **论文元素统一建模**：构建 `paper/page/section/element/bbox` 模型，原生提取正文、标题层级、图片、图注、表格、公式与页面坐标。
- **版面感知 PDF 解析**：引入 OCR/Layout 降级策略，建立 Figure-Caption、正文引用和相邻段落之间的稳定证据映射。
- **Any-to-Any 跨模态检索**：将 Text、Image、Text-Image Pair 编码到统一多模态向量空间，支持文本查图、图像查正文、图文对查方法。
- **页面级视觉检索**：结合页面视觉多向量检索、多模态 reranker、RRF 融合和元数据过滤。
- **Evidence Bundle 上下文管理**：将命中元素、所属章节、图片说明、关联正文、相邻段落及论文元数据统一组织进入生成上下文。
- **科研知识图谱**：抽取 Paper、Method、Module、Dataset、Metric、Figure 等实体及关系，支持方法对比、实验结果查询和相关工作总结。
- **科研评估集**：覆盖文本事实、图表理解、跨论文比较、多跳推理和不可回答问题，评估 Recall@K、MRR、nDCG、RAGAS、Citation Precision/Recall 与 P95 延迟。

## Architecture

```text
Client
  |
  v
FastAPI
  |
  +--> Auth / Session
  |
  +--> Document Upload
  |      |
  |      v
  |   Ingestion Pipeline
  |      |
  |      +--> PDF / Markdown / Text Parsing
  |      +--> Chunking
  |      +--> Embedding
  |      +--> ChromaDB + Redis
  |
  +--> Chat Query
         |
         v
      RAG Workflow
         |
         +--> Vector Retrieval
         +--> BM25 Retrieval
         +--> Rerank
         +--> Answer Generation
         +--> Evidence Verification
         |
         v
      Answer + Citations
```

## Repository Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI application entrypoint
│   ├── routers/                # Chat, document and user APIs
│   ├── schemas.py              # Pydantic request/response models
│   └── services/               # Service layer for RAG application
├── config/
│   └── settings.py             # Model, retrieval and storage settings
├── core/
│   ├── application.py          # Session-aware RAG application facade
│   ├── ingestion.py            # Document ingestion pipeline
│   ├── workflow.py             # Retrieval, reranking and generation workflow
│   ├── evidence_verifier.py    # Deterministic evidence grounding checks
│   ├── pdf_parser.py           # PDF-to-Markdown parser
│   └── documentManager.py      # Knowledge base document listing/reset helpers
├── tests/
│   └── test_evidence_verifier.py
├── requirements.txt
└── README.md
```

## Tech Stack

- **API**: FastAPI, Pydantic, Uvicorn
- **RAG framework**: LlamaIndex
- **LLM**: DashScope-compatible chat model
- **Embedding**: HuggingFace embedding model
- **Retrieval**: Vector retrieval, BM25, SentenceTransformer reranker
- **Storage**: ChromaDB, Redis
- **Document processing**: PyPDF, python-docx, Unstructured, PyMuPDF-based PDF parsing
- **Testing**: pytest

## Evidence-Grounded Answering

The current evidence verifier is deterministic by design. It does not use an LLM judge.

For each generated answer, the verifier checks:

- whether retrieved chunks exist;
- whether the top retrieval score passes the configured threshold;
- whether each answer sentence is covered by retrieved chunk text through exact or token-overlap matching;
- whether unsupported claims should be removed or trigger refusal.

Expected output fields:

```python
{
    "grounded": bool,
    "answerable": bool,
    "cited_chunk_ids": list[str],
    "unsupported_claims": list[str],
    "suggested_revision": str,
}
```

Citation format:

```text
[source: paper_or_file_name, page page_number, chunk_id]
```

## API Overview

### Health

```http
GET /api/health
```

### Documents

```http
POST /api/docs/upload
GET  /api/docs/list
POST /api/docs/reset
```

### Chat

```http
POST /api/chat/chat
POST /api/chat/chat/stream
GET  /api/chat/history
POST /api/chat/clear
```

### Users

```http
POST /users/register
POST /users/token
GET  /users/me
PUT  /users/me
POST /users/logout
```

## Quick Start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create `.env`:

```bash
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=your_dashscope_base_url
```

### 3. Start Redis

The application expects Redis on `127.0.0.1:6379`.

```bash
redis-server
```

### 4. Start API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000/docs
```

## Testing

Run unit tests:

```bash
PYTHONPATH=. pytest tests -q
```

The current test suite focuses on deterministic evidence verification:

- supported answer with citation;
- unsupported answer;
- empty retrieval result;
- low-score retrieval refusal;
- partially unsupported answer revision.

## Configuration

Core settings live in `config/settings.py`:

- `MODEL`, `TEMPERATURE`
- `EMBEDDING_MODEL_PATH`
- `RERANK_MODEL_PATH`
- `SIMILARITY_TOP_K`
- `RERANK_TOP_K`
- `SIMILARITY_CUTOFF`
- ChromaDB, Redis and document storage paths

For local development, update model paths to match your machine.

## Current Limitations

- The current verifier is lexical and deterministic; it is not an NLI model and does not fully understand paraphrases or complex multi-hop reasoning.
- Streaming answers currently return sources before generation; strict post-generation evidence rewriting is applied to the non-streaming RAG path.
- Multimodal retrieval, page-level visual embeddings, knowledge graph extraction and research benchmark metrics are planned work.

## License

MIT License
