from pathlib import Path

from core.research_citations import CitationValidator
from core.research_context import EvidenceBundleBuilder
from core.research_evaluation import EvaluationItem, ResearchEvaluationRunner
from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_pdf_parser import ResearchPDFParser
from core.research_retrieval import ResearchSearchFilters, ResearchSearchResult
from core.research_service import ResearchService
from core.research_store import ResearchStore


def _service(tmp_path: Path) -> ResearchService:
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    parser = ResearchPDFParser(str(tmp_path / "assets"))
    return ResearchService(store=store, parser=parser)


def _fake_pdf_text(title: str = "Gaussian Human Reconstruction") -> bytes:
    return f"""{title}

Alice Researcher, Bob Scientist, 2025

Abstract
We propose a 3D Gaussian human reconstruction method for animatable avatars.

1 Introduction
Human reconstruction benefits from 3D Gaussian Splatting and SMPL body priors.
Figure 1 shows the overall architecture of our method.

1 Method
The method uses SMPL surface anchored Gaussian initialization and non-rigid deformation.
Fig. 1: Overview of the Gaussian avatar architecture.

2 Experiments
We evaluate on ZJU-MoCap using PSNR, SSIM, and LPIPS metrics.
""".encode("utf-8")


def test_stable_paper_id_and_page_metadata(tmp_path):
    parser = ResearchPDFParser(str(tmp_path / "assets"))
    first = tmp_path / "a.pdf"
    second = tmp_path / "nested" / "b.pdf"
    second.parent.mkdir()
    first.write_bytes(_fake_pdf_text())
    second.write_bytes(_fake_pdf_text())

    paper_a = parser.parse(str(first))
    paper_b = parser.parse(str(second))

    assert paper_a.paper_id == paper_b.paper_id
    assert paper_a.metadata.parser_status == "native"
    assert all(element.page_number == 1 for element in paper_a.elements)
    assert any(element.section_path == ["Abstract"] for element in paper_a.elements)
    assert any("Method" in " ".join(element.section_path) for element in paper_a.elements)


def test_figure_caption_and_paragraph_linking(tmp_path):
    parser = ResearchPDFParser(str(tmp_path / "assets"))
    raw_blocks = [
        {"page_number": 1, "text": "Method", "bbox": [0, 0, 100, 20], "font_size": 16},
        {"page_number": 1, "text": "Figure 1 shows the overview.", "bbox": [0, 30, 100, 50], "font_size": 10},
        {"page_number": 1, "text": "Fig. 1: Overview architecture.", "bbox": [0, 200, 100, 220], "font_size": 10},
    ]
    images = [{
        "page_number": 1,
        "bbox": [0, 120, 100, 180],
        "image_path": str(tmp_path / "assets" / "fig.png"),
        "image_url": "/api/research/assets/paper_x/fig.png",
    }]

    elements, _ = parser._blocks_to_elements(raw_blocks, images, "paper_x", "paper.pdf")
    figures = [element for element in elements if element.element_type == "figure"]
    captions = [element for element in elements if element.element_type == "figure_caption"]

    assert len(figures) == 1
    assert len(captions) == 1
    figure = figures[0]
    assert figure.metadata["caption_element_id"] == captions[0].element_id
    assert figure.metadata["mentioned_by_element_ids"]
    assert figure.metadata["neighbor_element_ids"]
    assert figure.metadata["link_status"] == "caption_and_context"


def test_upload_card_search_compare_qa_and_evaluation(tmp_path):
    svc = _service(tmp_path)
    paper = svc.upload_pdf_bytes(_fake_pdf_text(), "paper.pdf")
    second = svc.upload_pdf_bytes(_fake_pdf_text("SMPL Gaussian Avatar"), "paper2.pdf")

    card = svc.get_card(paper.paper_id)
    assert card is not None
    assert card.fields["title"].value == "Gaussian Human Reconstruction"
    assert card.fields["title"].evidence_refs[0].paper_id == paper.paper_id

    manual = svc.update_card(paper.paper_id, {"representation": "3D Gaussian Splatting"})
    assert manual.is_manual is True
    assert svc.get_card(paper.paper_id).fields["representation"].value == "3D Gaussian Splatting"

    text_results = svc.search_text("SMPL Gaussian initialization", top_k=5)
    assert text_results
    assert text_results[0]["element_id"]
    assert text_results[0]["page_number"] == 1

    figure_results = svc.search_figures("overview architecture", top_k=5)
    assert figure_results
    assert figure_results[0]["figure_id"]
    assert figure_results[0]["hit_reason"] in {"caption match", "linked paragraph match", "both"}

    similar = svc.similar_papers(paper.paper_id, top_k=3)
    assert similar
    assert "matched_dimensions" in similar[0]

    report = svc.compare(paper.paper_id, [second.paper_id])
    assert "representation" in report.rows
    assert report.markdown_report.startswith("| Dimension |")

    qa = svc.qa("How does the method initialize Gaussians?", top_k=3)
    assert qa["answerable"] is True
    assert qa["citations"]
    assert CitationValidator(svc.store).validate(qa["citations"])[0] is True

    no_answer = svc.qa("What is the exact training GPU temperature?", top_k=3)
    assert no_answer["answerable"] is False

    relevant_id = text_results[0]["element_id"]
    item = EvaluationItem(
        query_id="q1",
        query_type="text_fact",
        input_modality="text",
        target_modality=["text"],
        question="SMPL Gaussian initialization",
        reference_answer="uses SMPL anchored Gaussian initialization",
        relevant_element_ids=[relevant_id],
        answerable=True,
        difficulty="easy",
        review_status="draft",
    )
    runner = ResearchEvaluationRunner()
    retrieval_metrics = runner.evaluate_retrieval([item], {"q1": [relevant_id]}, k_values=(5, 10))
    citation_metrics = runner.evaluate_citations([item], {"q1": qa["citations"]})
    assert retrieval_metrics["recall@5"] == 1.0
    assert citation_metrics["element_id_validity_rate"] == 1.0


def test_context_budget_and_no_duplicate_elements(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    paper = PaperDocument(
        paper_id="paper_budget",
        metadata=PaperMetadata(title="Budget Paper", source_file="budget.pdf"),
        elements=[
            PaperElement(
                element_id=f"e{i}",
                paper_id="paper_budget",
                element_type="paragraph",
                page_number=1,
                section_path=["Method"],
                text="Gaussian reconstruction evidence " + str(i),
                reading_order=i,
            )
            for i in range(5)
        ],
    )
    store.upsert_paper(paper)
    hits = [
        ResearchSearchResult(
            paper_id="paper_budget",
            element_id="e0",
            element_type="paragraph",
            page_number=1,
            section_path=["Method"],
            text="Gaussian reconstruction evidence 0",
            raw_score=1,
            fusion_score=1,
            rerank_score=1,
        ),
        ResearchSearchResult(
            paper_id="paper_budget",
            element_id="e0",
            element_type="paragraph",
            page_number=1,
            section_path=["Method"],
            text="Gaussian reconstruction evidence 0",
            raw_score=1,
            fusion_score=1,
            rerank_score=1,
        ),
    ]
    bundle = EvidenceBundleBuilder(store, max_chars=80, max_elements_per_paper=2).build(hits)
    assert bundle.manifest["element_ids"] == ["e0"]
    assert bundle.manifest["total_chars"] <= 80
