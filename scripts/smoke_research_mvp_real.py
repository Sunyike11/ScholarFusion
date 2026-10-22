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
from core.research_citations import CitationValidator
from core.research_pdf_parser import ResearchPDFParser
from core.research_service import ResearchService
from core.research_store import ResearchStore


SMOKE_TEXT_QUERIES = [
    "core innovation method",
    "initialization",
    "non-rigid deformation",
    "density control",
    "datasets metrics",
]

SMOKE_FIGURE_QUERIES = [
    "overall framework",
    "deformation module",
    "ablation experiment",
    "Gaussian initialization",
]

SMOKE_QA_QUERIES = [
    "What is the core method?",
    "What datasets or metrics are reported?",
    "What figure describes the framework?",
    "What does the paper say about an unsupported private training set?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summary-only real-paper Research RAG MVP smoke.")
    parser.add_argument("--pdf-dir", default=AppSettings.RESEARCH_REAL_PAPER_DIR)
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--max-papers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "research_real_smoke"))
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        result = {
            "status": "no_pdf_files",
            "pdf_dir": str(pdf_dir),
            "paper_count": 0,
            "papers": [],
        }
        write_json(output_dir / "summary.json", result)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("no_pdf_files")
        return 0

    selected = pdf_files[: max(args.max_papers, 0)]
    store = ResearchStore(str(output_dir / "research_real_smoke.sqlite3"))
    parser = ResearchPDFParser(str(output_dir / "assets"))
    service = ResearchService(store=store, parser=parser)
    paper_summaries: List[Dict[str, Any]] = []

    for pdf_path in selected:
        start = time.perf_counter()
        paper = service.upload_pdf_bytes(pdf_path.read_bytes(), pdf_path.name)
        latency_ms = (time.perf_counter() - start) * 1000
        if args.paper_id and paper.paper_id != args.paper_id:
            continue
        paper_summaries.append(summarize_paper(service, paper.paper_id, pdf_path.name, latency_ms))

    result = {
        "status": "ok",
        "pdf_dir": str(pdf_dir),
        "paper_count": len(paper_summaries),
        "papers": paper_summaries,
        "text_queries": run_text_queries(service),
        "figure_queries": run_figure_queries(service),
        "qa_queries": run_qa_queries(service),
    }
    write_json(output_dir / "summary.json", result)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"processed {len(paper_summaries)} paper(s)")
    return 0


def summarize_paper(service: ResearchService, paper_id: str, file_name: str, latency_ms: float) -> Dict[str, Any]:
    paper = service.get_paper(paper_id)
    if not paper:
        return {"paper_id": paper_id, "file_name": file_name, "parse_status": "missing_after_ingest"}
    elements = paper.elements
    figures = [element for element in elements if element.element_type == "figure"]
    captions = [element for element in elements if element.element_type == "figure_caption"]
    return {
        "paper_id": paper.paper_id,
        "file_name": file_name,
        "page_count": len({element.page_number for element in elements if element.page_number is not None}),
        "element_count": len(elements),
        "paragraph_count": count_type(elements, "paragraph"),
        "heading_count": count_type(elements, "heading"),
        "figure_count": len(figures),
        "caption_count": len(captions),
        "table_count": count_type(elements, "table"),
        "equation_count": count_type(elements, "equation"),
        "elements_with_page_count": len([element for element in elements if element.page_number is not None]),
        "elements_with_bbox_count": len([element for element in elements if element.bbox is not None]),
        "figures_with_caption_count": len([figure for figure in figures if figure.metadata.get("caption_element_id")]),
        "figures_with_linked_paragraph_count": len([
            figure for figure in figures
            if any((service.store.get_element(element_id) or figure).element_type in {"paragraph", "abstract"} for element_id in figure.linked_element_ids)
        ]),
        "parse_status": paper.metadata.parser_status,
        "parse_warnings": parse_warnings(paper.metadata.parse_quality),
        "ingestion_latency_ms": round(latency_ms, 2),
    }


def run_text_queries(service: ResearchService) -> List[Dict[str, Any]]:
    results = []
    for query in SMOKE_TEXT_QUERIES:
        hits = service.search_text(query, top_k=5)
        results.append({
            "query": query,
            "hit_count": len(hits),
            "top_element_ids": [hit["element_id"] for hit in hits[:3]],
        })
    return results


def run_figure_queries(service: ResearchService) -> List[Dict[str, Any]]:
    results = []
    for query in SMOKE_FIGURE_QUERIES:
        hits = service.search_figures(query, top_k=5)
        results.append({
            "query": query,
            "hit_count": len(hits),
            "top_figure_ids": [hit["figure_id"] for hit in hits[:3]],
            "hit_reasons": [hit["hit_reason"] for hit in hits[:3]],
        })
    return results


def run_qa_queries(service: ResearchService) -> List[Dict[str, Any]]:
    validator = CitationValidator(service.store)
    results = []
    for query in SMOKE_QA_QUERIES:
        response = service.qa(query, top_k=5)
        valid, errors = validator.validate(response.get("citations", []))
        results.append({
            "query": query,
            "answerable": bool(response.get("answerable")),
            "citation_count": len(response.get("citations", [])),
            "citations_valid": valid,
            "citation_errors": errors,
            "source_count": len(response.get("sources", [])),
        })
    return results


def count_type(elements, element_type: str) -> int:
    return len([element for element in elements if element.element_type == element_type])


def parse_warnings(parse_quality: Dict[str, Any]) -> List[str]:
    warnings = []
    if parse_quality.get("ocr_required_pages"):
        warnings.append("ocr_required_pages")
    if parse_quality.get("missing_page_numbers"):
        warnings.append("missing_page_numbers")
    if parse_quality.get("figures_without_caption"):
        warnings.append("figures_without_caption")
    if parse_quality.get("parser_dependency") == "fitz_unavailable_text_fallback":
        warnings.append("fitz_unavailable_text_fallback")
    return warnings


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
