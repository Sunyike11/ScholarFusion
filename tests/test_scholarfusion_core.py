from pathlib import Path

from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_multimodal_embedding import FakeMultimodalEmbeddingProvider
from core.research_store import ResearchStore
from core.scholarfusion_figures import enrich_and_deduplicate_figures, unique_non_decorative_figures
from core.scholarfusion_graph import ScholarFusionGraphStore, build_graph_from_store
from core.scholarfusion_index import IndexedVector, ScholarFusionVectorIndex
from core.scholarfusion_indexer import ScholarFusionIndexer
from core.scholarfusion_ocr import OCRLayoutFallbackParser, PageQualityGate
from core.scholarfusion_retrieval import AnyToAnyQuery, ScholarFusionRetriever, WeightedRRF
from core.scholarfusion_reranker import FakeMultimodalReranker
from core.scholarfusion_tasks import TaskStatusStore
from core.scholarfusion_verifier import AnswerabilityVerifier


def test_figure_enrichment_duplicate_and_decorative_filter(tmp_path):
    image = tmp_path / "fig.png"
    image.write_bytes(b"same-image")
    elements = [
        PaperElement("f1", "p1", "figure", 1, ["Method"], "Fig. 1: Framework overview.", [0, 0, 300, 200], metadata={"image_path": str(image), "caption": "Fig. 1: Framework overview.", "figure_label": "1"}),
        PaperElement("f2", "p1", "figure", 1, ["Method"], "Fig. 1: Framework overview.", [1, 1, 301, 201], metadata={"image_path": str(image), "caption": "Fig. 1: Framework overview.", "figure_label": "1"}),
        PaperElement("logo", "p1", "figure", 1, [], "", [0, 0, 20, 20], metadata={}),
    ]

    enrich_and_deduplicate_figures(elements)

    assert elements[0].metadata["image_hash"]
    assert elements[1].metadata["is_duplicate"] is True
    assert elements[2].metadata["is_decorative"] is True
    assert [figure.element_id for figure in unique_non_decorative_figures(elements)] == ["f1"]


def test_answerability_verifier_refuses_private_unsupported_query():
    verifier = AnswerabilityVerifier()
    evidence = [PaperElement("e1", "p1", "paragraph", 1, ["Method"], "The method uses SMPL and Gaussian initialization.", reading_order=1)]

    result = verifier.verify("What private deployment latency is reported?", evidence)

    assert result.status == "insufficient_evidence"
    assert result.reason_code == "unsupported_private_or_undisclosed"
    claims = verifier.verify_claims("The method uses SMPL.", evidence)
    assert claims[0].support_status == "supported"


def test_ocr_fallback_trigger():
    paper = PaperDocument(
        paper_id="p_scan",
        metadata=PaperMetadata(title="Scan", parse_quality={"pages": [{"page_number": 1, "effective_chars": 0, "image_count": 1}], "ocr_required_pages": [1]}),
        elements=[],
    )
    pages = PageQualityGate().low_quality_pages(paper)
    fallback = OCRLayoutFallbackParser().parse_pages("scan.pdf", paper.paper_id, pages)

    assert pages == [1]
    assert fallback[0].metadata["fallback_status"] == "triggered"


def test_vector_index_any_to_any_and_rerank(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    index = ScholarFusionVectorIndex(str(tmp_path / "index.sqlite3"))
    provider = FakeMultimodalEmbeddingProvider(dimension=8)
    paper = PaperDocument(
        paper_id="p1",
        metadata=PaperMetadata(title="Gaussian Avatar"),
        elements=[
            PaperElement("t1", "p1", "paragraph", 1, ["Method"], "Gaussian avatar method uses SMPL initialization.", reading_order=1),
            PaperElement("cap1", "p1", "figure_caption", 1, ["Method"], "Fig. 1: Framework architecture.", reading_order=2, metadata={"figure_label": "1"}),
            PaperElement("fig1", "p1", "figure", 1, ["Method"], "Fig. 1: Framework architecture.", [0, 0, 200, 200], reading_order=3, linked_element_ids=["cap1"], metadata={"caption": "Fig. 1: Framework architecture.", "caption_element_id": "cap1"}),
        ],
    )
    enrich_and_deduplicate_figures(paper.elements)
    store.upsert_paper(paper)
    summary = ScholarFusionIndexer(store, index, provider).index_paper("p1", force=True, max_requests=20)

    retriever = ScholarFusionRetriever(store, index, provider)
    results = retriever.search_any_to_any(AnyToAnyQuery(query_text="framework architecture", target_modalities=["text", "figure", "pair", "page"], top_k=5))
    reranked = FakeMultimodalReranker().rerank("framework architecture", results, top_n=3)

    assert summary["text"] >= 1
    assert summary["figure"] >= 1
    assert summary["pair"] >= 1
    assert results
    assert reranked[0].metadata["rerank_trace"]["post_rerank_rank"] == 1
    assert provider.last_request_metadata["enable_fusion"] in {False, True}


def test_graph_and_task_store(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    graph = ScholarFusionGraphStore(str(tmp_path / "research.sqlite3"))
    paper = PaperDocument(
        paper_id="p_graph",
        metadata=PaperMetadata(title="Graph Paper"),
        elements=[PaperElement("e1", "p_graph", "paragraph", 1, ["Abstract"], "This paper uses SMPL.", reading_order=1)],
    )
    store.upsert_paper(paper)
    counts = build_graph_from_store(store, graph)
    query = graph.query("Paper")
    tasks = TaskStatusStore(str(tmp_path / "tasks.sqlite3"))
    tasks.set_status("task1", "parsing", 0.2, "Parsing")

    assert counts["node_count"] >= 1
    assert query["nodes"]
    assert tasks.get("task1")["status"] == "parsing"
