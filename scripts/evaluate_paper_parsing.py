from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScholarFusion parsing quality.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--output", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion/parsing_quality.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ResearchStore(args.db_path)
    result = evaluate(store)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def evaluate(store: ResearchStore) -> Dict[str, Any]:
    papers = store.list_papers()
    elements = store.list_elements()
    figures = [element for element in elements if element.element_type == "figure"]
    unique_figures = [figure for figure in figures if not figure.metadata.get("is_duplicate") and not figure.metadata.get("is_decorative")]
    figures_with_caption = [figure for figure in unique_figures if figure.metadata.get("caption_element_id")]
    figures_with_linked = [figure for figure in unique_figures if figure.metadata.get("mentioned_by_element_ids") or figure.metadata.get("neighbor_element_ids")]
    duplicate_filtered_count = len([figure for figure in figures if figure.metadata.get("is_duplicate")])
    decorative_filtered_count = len([figure for figure in figures if figure.metadata.get("is_decorative")])
    indexed_figures = [
        figure for figure in unique_figures
        if figure.metadata.get("image_path") and not figure.metadata.get("synthetic_from_caption")
    ]
    explicit_figures = []
    high_confidence_caption = []
    fallback_linked = []
    for figure in unique_figures:
        links = figure.metadata.get("link_evidence") or []
        if any(link.get("link_type") == "explicit_reference" and float(link.get("confidence", 0.0)) >= 0.8 for link in links):
            explicit_figures.append(figure)
        if any(link.get("link_type") == "caption_neighbor" and float(link.get("confidence", 0.0)) >= 0.8 for link in links):
            high_confidence_caption.append(figure)
        if any(link.get("link_type") in {"same_page_neighbor", "same_section_fallback"} for link in links):
            fallback_linked.append(figure)
    warnings = []
    for paper in papers:
        quality = paper.metadata.parse_quality
        warnings.extend(quality.get("ocr_required_pages") or [])
        if quality.get("figures_without_caption"):
            warnings.append("figures_without_caption")
    return {
        "paper_count": len(papers),
        "figure_count": len(figures),
        "unique_figure_count": len(unique_figures),
        "duplicate_figure_rate": round((len(figures) - len(unique_figures)) / len(figures), 4) if figures else 0.0,
        "figures_with_caption_rate": round(len(figures_with_caption) / len(unique_figures), 4) if unique_figures else 0.0,
        "figures_with_linked_paragraph_rate": round(len(figures_with_linked) / len(unique_figures), 4) if unique_figures else 0.0,
        "high_confidence_caption_rate": round(len(high_confidence_caption) / len(unique_figures), 4) if unique_figures else 0.0,
        "explicit_reference_rate": round(len(explicit_figures) / len(unique_figures), 4) if unique_figures else 0.0,
        "fallback_link_rate": round(len(fallback_linked) / len(unique_figures), 4) if unique_figures else 0.0,
        "decorative_filtered_count": decorative_filtered_count,
        "duplicate_filtered_count": duplicate_filtered_count,
        "indexed_figure_count": len(indexed_figures),
        "elements_with_page_rate": round(len([element for element in elements if element.page_number is not None]) / len(elements), 4) if elements else 0.0,
        "elements_with_bbox_rate": round(len([element for element in elements if element.bbox is not None]) / len(elements), 4) if elements else 0.0,
        "parse_warning_count": len(warnings),
    }


if __name__ == "__main__":
    raise SystemExit(main())
