import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置类"""

    # OpenAI配置
    API_KEY: Optional[str] = os.getenv("DASHSCOPE_API_KEY")
    API_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
    MODEL: str = "qwen-plus-2025-07-14"
    DASHSCOPE_TEXT_MODEL: str = os.getenv("DASHSCOPE_TEXT_MODEL", MODEL)
    DASHSCOPE_VL_MODEL: str = os.getenv("DASHSCOPE_VL_MODEL", "qwen3-vl-plus")
    DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL: str = os.getenv("DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL", "qwen3-vl-embedding")
    DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT: str = os.getenv(
        "DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
    )
    _DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION = os.getenv("DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION")
    DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION: Optional[int] = (
        int(_DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION)
        if _DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION
        else None
    )
    DASHSCOPE_REQUEST_TIMEOUT: float = float(os.getenv("DASHSCOPE_REQUEST_TIMEOUT", "60"))
    DASHSCOPE_MULTIMODAL_RERANK_MODEL: str = os.getenv("DASHSCOPE_MULTIMODAL_RERANK_MODEL", "qwen3-vl-rerank")
    DASHSCOPE_MULTIMODAL_RERANK_ENDPOINT: str = os.getenv(
        "DASHSCOPE_MULTIMODAL_RERANK_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    SCHOLARFUSION_ENABLE_REAL_RERANKER: bool = os.getenv("SCHOLARFUSION_ENABLE_REAL_RERANKER", "false").lower() in {"1", "true", "yes"}
    TEMPERATURE: float = 0.1
    EMBEDDING_MODEL_PATH: str = r"D:\llm\Local_model\BAAI\bge-small-zh-v1___5"

    # 文档处理配置
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TITLE_EXTRACTOR_NODES: int = 5

    # 检索配置
    SIMILARITY_TOP_K: int = 5
    RERANK_TOP_K: int = 3
    SIMILARITY_CUTOFF: float = 0.5
    RERANK_MODEL_PATH: str = r"D:\llm\Local_model\BAAI\bge-reranker-large"

    # 存储配置
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # 项目根目录
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "file/chroma_db")
    BM25_PERSIST_DIR: str = str(PROJECT_ROOT / "file/storage_bm25")
    DOCUMENTS_DIR: str = str(PROJECT_ROOT / "file/documents")
    DEFAULT_PERSIST_DIR: str = str(PROJECT_ROOT / "file/storage")
    PDF_IMAGE_DIR: str = str(PROJECT_ROOT / "file/image")
    LOG_DIR: str = str(PROJECT_ROOT / "file")
    RESEARCH_STORAGE_DIR: str = str(PROJECT_ROOT / "file/research")
    RESEARCH_DB_PATH: str = str(PROJECT_ROOT / "file/research/research_papers.sqlite3")
    RESEARCH_ASSET_DIR: str = str(PROJECT_ROOT / "file/research/assets")
    RESEARCH_REAL_PAPER_DIR: str = str(PROJECT_ROOT / "data/research_papers")
    RESEARCH_EVAL_DATA_DIR: str = str(PROJECT_ROOT / "evaluation_data/research_v1")
    RESEARCH_EVAL_RESULTS_DIR: str = str(PROJECT_ROOT / "evaluation_results")
    RESEARCH_STATIC_URL_PREFIX: str = "/api/research/assets"
    RESEARCH_CONTEXT_MAX_CHARS: int = 6000
    RESEARCH_MAX_EVIDENCE_PER_PAPER: int = 6
    RESEARCH_COMPARE_MAX_PAPERS: int = 5
    SCHOLARFUSION_INDEX_DB_PATH: str = str(PROJECT_ROOT / "file/research/scholarfusion_index.sqlite3")
    SCHOLARFUSION_TASK_DB_PATH: str = str(PROJECT_ROOT / "file/research/scholarfusion_tasks.sqlite3")
    SCHOLARFUSION_TEXT_COLLECTION: str = os.getenv("SCHOLARFUSION_TEXT_COLLECTION", "research_text_elements")
    SCHOLARFUSION_MULTIMODAL_COLLECTION: str = os.getenv("SCHOLARFUSION_MULTIMODAL_COLLECTION", "research_multimodal_elements")
    SCHOLARFUSION_PAGE_COLLECTION: str = os.getenv("SCHOLARFUSION_PAGE_COLLECTION", "research_page_visual")
    SCHOLARFUSION_PAPER_COLLECTION: str = os.getenv("SCHOLARFUSION_PAPER_COLLECTION", "research_paper_level")
    SCHOLARFUSION_TEXT_RECALL_TOP_K: int = int(os.getenv("SCHOLARFUSION_TEXT_RECALL_TOP_K", "20"))
    SCHOLARFUSION_MULTIMODAL_RECALL_TOP_K: int = int(os.getenv("SCHOLARFUSION_MULTIMODAL_RECALL_TOP_K", "20"))
    SCHOLARFUSION_PAGE_RECALL_TOP_K: int = int(os.getenv("SCHOLARFUSION_PAGE_RECALL_TOP_K", "10"))
    SCHOLARFUSION_FUSION_TOP_K: int = int(os.getenv("SCHOLARFUSION_FUSION_TOP_K", "20"))
    SCHOLARFUSION_RERANK_TOP_N: int = int(os.getenv("SCHOLARFUSION_RERANK_TOP_N", "10"))
    SCHOLARFUSION_FINAL_TOP_K: int = int(os.getenv("SCHOLARFUSION_FINAL_TOP_K", "5"))
    SCHOLARFUSION_MAX_FIGURES_PER_CONTEXT: int = int(os.getenv("SCHOLARFUSION_MAX_FIGURES_PER_CONTEXT", "3"))
    SCHOLARFUSION_MAX_PAGES_PER_CONTEXT: int = int(os.getenv("SCHOLARFUSION_MAX_PAGES_PER_CONTEXT", "2"))
    SCHOLARFUSION_MAX_REQUESTS: int = int(os.getenv("SCHOLARFUSION_MAX_REQUESTS", "100"))
    SCHOLARFUSION_RRF_K: int = int(os.getenv("SCHOLARFUSION_RRF_K", "60"))
    SCHOLARFUSION_WEIGHTS: dict = {
        "bm25": 1.0,
        "text_dense": 1.0,
        "multimodal_dense": 1.2,
        "page_visual": 0.8,
        "paper_level": 0.5,
    }
    SCHOLARFUSION_MIN_ANSWERABILITY_CONFIDENCE: float = float(os.getenv("SCHOLARFUSION_MIN_ANSWERABILITY_CONFIDENCE", "0.34"))
    RESEARCH_SIMILARITY_WEIGHTS: dict = {
        "task": 1.2,
        "representation": 1.0,
        "body_model": 1.0,
        "initialization": 1.0,
        "non_rigid_deformation": 1.0,
        "gaussian_binding": 1.0,
        "density_control": 0.8,
        "datasets": 0.8,
    }

    # 支持的文件类型
    SUPPORTED_FILE_TYPES: list = [".txt", ".pdf", ".docx", ".md"]

    @classmethod
    def validate_api_key(cls) -> bool:
        """验证API密钥是否存在"""
        return cls.API_KEY is not None and len(cls.API_KEY.strip()) > 0
