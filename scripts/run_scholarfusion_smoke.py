from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_multimodal_embedding import DashScopeQwen3VLEmbeddingProvider, FakeMultimodalEmbeddingProvider
from core.research_pdf_parser import ResearchPDFParser
from core.research_service import ResearchService
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScholarFusion vertical smoke.")
    parser.add_argument("--pdf-dir", default=AppSettings.RESEARCH_REAL_PAPER_DIR)
    parser.add_argument("--max-papers", type=int, default=3)
    parser.add_argument("--output-dir", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion"))
    parser.add_argument("--real-provider", action="store_true")
    parser.add_argument("--max-requests", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))[: args.max_papers] if Path(args.pdf_dir).exists() else []
    if not pdfs:
        result = {"status": "no_pdf_files", "pdf_dir": args.pdf_dir}
        write_json(output_dir / "scholarfusion_smoke.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    db_path = output_dir / "scholarfusion.sqlite3"
    index_path = output_dir / "scholarfusion_index.sqlite3"
    service = ResearchService(
        store=ResearchStore(str(db_path)),
        parser=ResearchPDFParser(str(output_dir / "assets")),
    )
    service.set_vector_index(ScholarFusionVectorIndex(str(index_path)))
    provider = DashScopeQwen3VLEmbeddingProvider() if args.real_provider else FakeMultimodalEmbeddingProvider(
        model=AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL,
        dimension=2560,
    )
    papers = []
    total_embedding_requests = 0
    start_all = time.perf_counter()
    for pdf in pdfs:
        start = time.perf_counter()
        paper = service.upload_pdf_bytes(pdf.read_bytes(), pdf.name)
        # Use explicit provider for this smoke.
        from core.scholarfusion_indexer import ScholarFusionIndexer

        index_summary = ScholarFusionIndexer(service.store, service.vector_index, provider).index_paper(
            paper.paper_id,
            force=args.force,
            max_requests=args.max_requests,
        )
        total_embedding_requests += index_summary.get("request_count", 0)
        graph = service.graph_for_paper(paper.paper_id)
        papers.append({
            "paper_id": paper.paper_id,
            "file_name": pdf.name,
            "parse_status": paper.metadata.parser_status,
            "element_count": len(paper.elements),
            "figure_count": len([e for e in paper.elements if e.element_type == "figure"]),
            "unique_figure_count": len([e for e in paper.elements if e.element_type == "figure" and not e.metadata.get("is_duplicate") and not e.metadata.get("is_decorative")]),
            "index_summary": index_summary,
            "graph_nodes": len(graph.get("nodes", [])),
            "graph_edges": len(graph.get("edges", [])),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        })

    text_results = service.search_any_to_any(query_text="Gaussian avatar initialization", target_modalities=["text", "figure", "pair", "page"], top_k=5)
    image_results: List[Dict[str, Any]] = []
    first_figure = next((e for e in service.store.list_elements(element_types=["figure"]) if e.metadata.get("image_path") and not e.metadata.get("is_duplicate") and not e.metadata.get("is_decorative")), None)
    if not first_figure:
        first_figure = next((e for e in service.store.list_elements(element_types=["figure"]) if e.metadata.get("image_path")), None)
    if first_figure:
        image_results = service.search_any_to_any(query_image=first_figure.metadata["image_path"], target_modalities=["text", "figure", "pair", "page"], top_k=5)
    pair_results: List[Dict[str, Any]] = []
    if first_figure:
        pair_results = service.search_any_to_any(query_text="framework architecture", query_image=first_figure.metadata["image_path"], target_modalities=["text", "figure", "pair", "page"], top_k=5)
    qa_supported = service.qa("What initialization is used?", top_k=5)
    qa_unanswerable = service.qa("What private deployment latency on an undisclosed production dataset is reported?", top_k=5)
    graph_query = service.graph_query("SMPL")
    related_work = service.related_work("SMPL", top_k=3)
    result = {
        "status": "ok",
        "provider": "dashscope" if args.real_provider else "fake",
        "paper_count": len(papers),
        "papers": papers,
        "embedding_request_count": total_embedding_requests,
        "index_counts": {
            "text": service.vector_index.count(AppSettings.SCHOLARFUSION_TEXT_COLLECTION),
            "multimodal": service.vector_index.count(AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION),
            "page": service.vector_index.count(AppSettings.SCHOLARFUSION_PAGE_COLLECTION),
            "paper": service.vector_index.count(AppSettings.SCHOLARFUSION_PAPER_COLLECTION),
        },
        "text_any_to_any_top": [item["object_id"] for item in text_results[:5]],
        "image_any_to_any_top": [item["object_id"] for item in image_results[:5]],
        "pair_any_to_any_top": [item["object_id"] for item in pair_results[:5]],
        "qa_supported_answerable": qa_supported["answerable"],
        "qa_unanswerable_answerable": qa_unanswerable["answerable"],
        "graph_query_node_count": len(graph_query.get("nodes", [])),
        "related_work_citation_count": len(related_work.get("citations", [])),
        "total_latency_ms": round((time.perf_counter() - start_all) * 1000, 2),
    }
    write_json(output_dir / "scholarfusion_smoke.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
