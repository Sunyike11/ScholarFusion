import hashlib
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.schemas_research import (
    ComparePapersRequest,
    FigureListResponse,
    FigureSearchRequest,
    AnyToAnySearchRequest,
    GraphQueryRequest,
    PairSearchRequest,
    PaperCardResponse,
    PaperCardUpdateRequest,
    PaperDetailResponse,
    PaperSummaryResponse,
    ResearchQARequest,
    RelatedWorkRequest,
    ResearchTaskResponse,
    ResearchUploadResponse,
    SimilarPapersRequest,
    TextSearchRequest,
)
from config.settings import Settings as AppSettings
from core.research_retrieval import ResearchSearchFilters
from core.research_service import ResearchService
from core.scholarfusion_tasks import TaskStatusStore

router = APIRouter()
_research_service = ResearchService()
_task_store = TaskStatusStore(AppSettings.SCHOLARFUSION_TASK_DB_PATH)


def get_research_service() -> ResearchService:
    return _research_service


def _filters_from_request(filters) -> ResearchSearchFilters:
    if not filters:
        return ResearchSearchFilters()
    return ResearchSearchFilters(
        paper_ids=filters.paper_ids,
        years=filters.years,
        section_contains=filters.section_contains,
        element_types=filters.element_types,
        body_only=filters.body_only,
    )


@router.post("/papers/upload", response_model=ResearchUploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF upload is supported for research papers")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    paper = get_research_service().upload_pdf_bytes(content, file.filename)
    return ResearchUploadResponse(
        paper_id=paper.paper_id,
        title=paper.metadata.title,
        parser_status=paper.metadata.parser_status,
        parse_quality=paper.metadata.parse_quality,
        element_count=len(paper.elements),
    )


@router.post("/papers/upload-async")
async def upload_paper_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF upload is supported for research papers")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    task_id = hashlib.sha1(content + file.filename.encode("utf-8")).hexdigest()[:16]
    _task_store.set_status(task_id, "pending", 0.0, "Queued for ingestion")
    background_tasks.add_task(_run_ingestion_task, task_id, content, file.filename)
    return {"task_id": task_id, "status": "pending"}


def _run_ingestion_task(task_id: str, content: bytes, filename: str) -> None:
    try:
        _task_store.set_status(task_id, "parsing", 0.15, "Parsing PDF")
        paper = get_research_service().upload_pdf_bytes(content, filename)
        _task_store.set_status(task_id, "embedding", 0.55, "Building multimodal index", paper_id=paper.paper_id)
        index_summary = get_research_service().build_multimodal_index(paper.paper_id, force=True, max_requests=AppSettings.SCHOLARFUSION_MAX_REQUESTS)
        _task_store.set_status(task_id, "graph_building", 0.85, "Building graph", paper_id=paper.paper_id, summary=index_summary)
        graph = get_research_service().graph_for_paper(paper.paper_id)
        _task_store.set_status(task_id, "completed", 1.0, "Completed", paper_id=paper.paper_id, summary={
            "paper_id": paper.paper_id,
            "element_count": len(paper.elements),
            "index": index_summary,
            "graph_node_count": len(graph.get("nodes", [])),
            "graph_edge_count": len(graph.get("edges", [])),
        })
    except Exception as exc:
        _task_store.set_status(task_id, "failed", 1.0, str(exc)[:300])


@router.get("/papers", response_model=List[PaperSummaryResponse])
def list_papers():
    return [
        PaperSummaryResponse(
            paper_id=paper.paper_id,
            title=paper.metadata.title,
            authors=paper.metadata.authors,
            year=paper.metadata.year,
            venue=paper.metadata.venue,
            parser_status=paper.metadata.parser_status,
        )
        for paper in get_research_service().list_papers()
    ]


@router.get("/papers/{paper_id}", response_model=PaperDetailResponse)
def get_paper(paper_id: str):
    paper = get_research_service().get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperDetailResponse(
        paper_id=paper.paper_id,
        metadata=paper.metadata.to_dict(),
        elements=[element.to_dict() for element in paper.elements],
    )


@router.get("/papers/{paper_id}/card", response_model=PaperCardResponse)
def get_card(paper_id: str):
    card = get_research_service().get_card(paper_id)
    if not card:
        raise HTTPException(status_code=404, detail="Paper card not found")
    return PaperCardResponse(paper_id=paper_id, card=card.to_dict())


@router.put("/papers/{paper_id}/card", response_model=PaperCardResponse)
def update_card(paper_id: str, req: PaperCardUpdateRequest):
    if not get_research_service().get_paper(paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    card = get_research_service().update_card(paper_id, req.fields)
    return PaperCardResponse(paper_id=paper_id, card=card.to_dict())


@router.get("/papers/{paper_id}/figures", response_model=FigureListResponse)
def list_figures(paper_id: str):
    if not get_research_service().get_paper(paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    return FigureListResponse(
        paper_id=paper_id,
        figures=[figure.to_dict() for figure in get_research_service().list_figures(paper_id)],
    )


@router.post("/research/search/text")
def search_text(req: TextSearchRequest):
    filters = _filters_from_request(req.filters)
    return {
        "query": req.query,
        "results": get_research_service().search_text(req.query, top_k=req.top_k, filters=filters),
    }


@router.post("/research/search/figures")
def search_figures(req: FigureSearchRequest):
    filters = _filters_from_request(req.filters)
    return {
        "query": req.query,
        "results": get_research_service().search_figures(req.query, top_k=req.top_k, filters=filters),
    }


@router.post("/research/search/any-to-any")
def search_any_to_any(req: AnyToAnySearchRequest):
    if not req.query_text and not req.query_element_id:
        raise HTTPException(status_code=400, detail="query_text or query_element_id is required")
    filters = _filters_from_request(req.filters)
    return {
        "query_text": req.query_text,
        "query_element_id": req.query_element_id,
        "results": get_research_service().search_any_to_any(
            query_text=req.query_text,
            query_element_id=req.query_element_id,
            target_modalities=req.target_modalities,
            filters=filters,
            top_k=req.top_k,
        ),
    }


@router.post("/research/search/image")
async def search_image(
        file: UploadFile = File(...),
        target_modalities: str = Form("text,figure,page,pair"),
        top_k: int = Form(10),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image upload is empty")
    modalities = [item.strip() for item in target_modalities.split(",") if item.strip()]
    return {
        "results": get_research_service().search_any_to_any(
            query_image=content,
            target_modalities=modalities,
            top_k=top_k,
        )
    }


@router.post("/research/search/pair")
async def search_pair(
        req_text: str = Form(...),
        file: UploadFile = File(...),
        target_modalities: str = Form("text,figure,page,pair"),
        top_k: int = Form(10),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image upload is empty")
    modalities = [item.strip() for item in target_modalities.split(",") if item.strip()]
    return {
        "results": get_research_service().search_any_to_any(
            query_text=req_text,
            query_image=content,
            target_modalities=modalities,
            top_k=top_k,
        )
    }


@router.post("/research/search/similar-papers")
def similar_papers(req: SimilarPapersRequest):
    if not get_research_service().get_paper(req.paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    return {
        "paper_id": req.paper_id,
        "results": get_research_service().similar_papers(req.paper_id, top_k=req.top_k),
    }


@router.post("/research/compare")
def compare(req: ComparePapersRequest):
    if not get_research_service().get_paper(req.base_paper_id):
        raise HTTPException(status_code=404, detail="Base paper not found")
    report = get_research_service().compare(req.base_paper_id, req.compared_paper_ids)
    return report.to_dict()


@router.post("/research/qa")
def research_qa(req: ResearchQARequest):
    filters = _filters_from_request(req.filters)
    return get_research_service().qa(req.question, top_k=req.top_k, filters=filters)


@router.get("/research/tasks/{task_id}", response_model=ResearchTaskResponse)
def get_task(task_id: str):
    data = _task_store.get(task_id)
    return ResearchTaskResponse(
        task_id=task_id,
        status=data.get("status", "not_found"),
        message=data.get("message", ""),
        progress=float(data.get("progress", 0.0) or 0.0),
        paper_id=data.get("paper_id"),
        summary=data.get("summary") or {},
    )


@router.get("/research/tasks/{task_id}/stream")
def stream_task(task_id: str):
    def event_stream():
        data = _task_store.get(task_id)
        yield f"event: task_status\ndata: {data}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/papers/{paper_id}/graph")
def paper_graph(paper_id: str):
    if not get_research_service().get_paper(paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    return get_research_service().graph_for_paper(paper_id)


@router.post("/research/graph/query")
def graph_query(req: GraphQueryRequest):
    return get_research_service().graph_query(req.query)


@router.post("/research/related-work")
def related_work(req: RelatedWorkRequest):
    return get_research_service().related_work(req.query, top_k=req.top_k)


@router.get("/research/assets/{paper_id}/{filename}")
def get_research_asset(paper_id: str, filename: str):
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid asset filename")
    root = Path(AppSettings.RESEARCH_ASSET_DIR).resolve()
    path = (root / paper_id / filename).resolve()
    if root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)
