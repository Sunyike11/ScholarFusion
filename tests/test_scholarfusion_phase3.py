import csv

from core.phase3_production_retrieval import Phase3ProductionRetriever, ProtectedWeightedRRF
from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_store import ResearchStore
from core.scholarfusion_index import IndexedVector, ScholarFusionVectorIndex
from core.scholarfusion_retrieval import AnyToAnyResult
from scripts.build_small_gold_v1 import approved_from_csv


def _result(object_id: str, source: str, score: float = 1.0) -> AnyToAnyResult:
    return AnyToAnyResult(
        object_id=object_id,
        paper_id="p1",
        element_id=object_id,
        matched_modality="text",
        fusion_score=score,
        raw_scores={source: score},
        retrieval_sources=[source],
    )


def test_protected_rrf_keeps_primary_dense_topk():
    rrf = ProtectedWeightedRRF()
    fused = rrf.fuse_with_primary(
        {
            "multimodal_dense": [_result(f"d{i}", "multimodal_dense") for i in range(5)],
            "bm25": [_result(f"b{i}", "bm25") for i in range(20)],
            "chroma_dense": [_result(f"c{i}", "chroma_dense") for i in range(20)],
        },
        top_k=5,
        primary_source="multimodal_dense",
        protect_k=5,
    )

    assert [item.object_id for item in fused] == ["d0", "d1", "d2", "d3", "d4"]


def test_phase3_retriever_prepares_real_chroma_and_bm25(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    paper = PaperDocument(
        paper_id="p1",
        metadata=PaperMetadata(title="Phase3 Paper"),
        elements=[
            PaperElement("e1", "p1", "paragraph", 1, ["Method"], "Gaussian initialization uses SMPL anchors."),
            PaperElement("e2", "p1", "paragraph", 2, ["Experiments"], "The method reports PSNR and SSIM."),
        ],
    )
    store.upsert_paper(paper)
    index = ScholarFusionVectorIndex(str(tmp_path / "index.sqlite3"))
    index.upsert(
        IndexedVector(
            collection="research_text_elements",
            object_id="e1",
            paper_id="p1",
            element_id="e1",
            modality="text",
            vector=[0.1] * 2560,
            text="Gaussian initialization uses SMPL anchors.",
            metadata={"paper_id": "p1", "element_id": "e1"},
        )
    )

    retriever = Phase3ProductionRetriever(
        store,
        index,
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="phase3_test_text_dense",
    )
    summary = retriever.prepare(force=True)

    assert summary["chroma_count"] == 1
    assert summary["bm25_enabled"] is True
    assert summary["bm25_node_count"] == 2
    assert retriever._bm25_results("SMPL anchors", 3)


def test_small_gold_requires_explicit_human_approval(tmp_path):
    csv_path = tmp_path / "spot_check.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "spot_check_status"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "spot_check_status": ""})
        writer.writerow({"query_id": "q2", "spot_check_status": "approved"})

    assert approved_from_csv(csv_path) == {"q2"}
