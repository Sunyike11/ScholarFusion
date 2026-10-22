from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchUploadResponse(BaseModel):
    paper_id: str
    title: str
    parser_status: str
    parse_quality: Dict[str, Any]
    element_count: int


class PaperSummaryResponse(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    venue: str = "unknown"
    parser_status: str = "native"


class PaperDetailResponse(BaseModel):
    paper_id: str
    metadata: Dict[str, Any]
    elements: List[Dict[str, Any]]


class PaperCardResponse(BaseModel):
    paper_id: str
    card: Dict[str, Any]


class PaperCardUpdateRequest(BaseModel):
    fields: Dict[str, Any] = Field(default_factory=dict)


class FigureListResponse(BaseModel):
    paper_id: str
    figures: List[Dict[str, Any]]


class ResearchFiltersRequest(BaseModel):
    paper_ids: Optional[List[str]] = None
    years: Optional[List[int]] = None
    section_contains: Optional[str] = None
    element_types: Optional[List[str]] = None
    body_only: bool = False


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[ResearchFiltersRequest] = None


class FigureSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[ResearchFiltersRequest] = None


class SimilarPapersRequest(BaseModel):
    paper_id: str
    top_k: int = 5


class ComparePapersRequest(BaseModel):
    base_paper_id: str
    compared_paper_ids: List[str]


class ResearchQARequest(BaseModel):
    question: str
    top_k: int = 5
    filters: Optional[ResearchFiltersRequest] = None


class AnyToAnySearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_element_id: Optional[str] = None
    target_modalities: Optional[List[str]] = None
    top_k: int = 10
    filters: Optional[ResearchFiltersRequest] = None


class PairSearchRequest(BaseModel):
    query_text: str
    target_modalities: Optional[List[str]] = None
    top_k: int = 10
    filters: Optional[ResearchFiltersRequest] = None


class GraphQueryRequest(BaseModel):
    query: str
    top_k: int = 10


class RelatedWorkRequest(BaseModel):
    query: str
    top_k: int = 5


class ResearchTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    progress: float = 0.0
    paper_id: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
