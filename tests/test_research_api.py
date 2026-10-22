from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers import research
from core.research_pdf_parser import ResearchPDFParser
from core.research_service import ResearchService
from core.research_store import ResearchStore


def test_research_upload_and_qa_api(tmp_path):
    service = ResearchService(
        store=ResearchStore(str(tmp_path / "research.sqlite3")),
        parser=ResearchPDFParser(str(tmp_path / "assets")),
    )
    original = research._research_service
    research._research_service = service
    try:
        test_app = FastAPI()
        test_app.include_router(research.router, prefix="/api")
        client = TestClient(test_app)
        content = b"""API Paper

Abstract
This paper studies Gaussian human reconstruction.

Method
The method uses SMPL anchored Gaussian initialization.
Fig. 1: Method overview.
"""
        upload = client.post(
            "/api/papers/upload",
            files={"file": ("api_paper.pdf", content, "application/pdf")},
        )
        assert upload.status_code == 200
        paper_id = upload.json()["paper_id"]

        papers = client.get("/api/papers")
        assert papers.status_code == 200
        assert papers.json()[0]["paper_id"] == paper_id

        qa = client.post("/api/research/qa", json={"question": "What initialization is used?", "top_k": 3})
        assert qa.status_code == 200
        body = qa.json()
        assert body["answerable"] is True
        assert body["citations"][0]["paper_id"] == paper_id
    finally:
        research._research_service = original
