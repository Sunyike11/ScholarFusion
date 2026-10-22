from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_store import ResearchStore


TYPES = [
    "text_fact",
    "method_detail",
    "text_to_figure",
    "image_to_text",
    "image_to_image",
    "pair_to_any",
    "cross_paper_comparison",
    "unanswerable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ScholarFusion Lite provisional eval candidates.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--output-dir", default=str(Path("evaluation_data") / "scholarfusion_lite_v1"))
    parser.add_argument("--target-count", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    candidates_path = output_dir / "candidates.jsonl"
    review_path = output_dir / "review_queue.csv"
    if candidates_path.exists() and not args.force:
        print(json.dumps({"status": "exists_without_force", "path": str(candidates_path)}, ensure_ascii=False))
        return 0
    store = ResearchStore(args.db_path)
    records = build_records(store, args.target_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""), encoding="utf-8")
    write_review_queue(review_path, records[:40])
    print(json.dumps({"status": "ok", "candidate_count": len(records), "review_queue_count": min(len(records), 40), "dataset_status": "provisional", "not_gold": True}, ensure_ascii=False))
    return 0


def build_records(store: ResearchStore, target_count: int) -> List[Dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    records: List[Dict[str, Any]] = []
    papers = store.list_papers()
    seen = set()
    for paper in papers:
        elements = store.list_elements(paper_id=paper.paper_id)
        text_elements = [e for e in elements if e.element_type in {"abstract", "paragraph", "figure_caption"} and e.text.strip()]
        text_elements.sort(key=lambda e: (
            0 if e.element_type == "abstract" else 1,
            0 if any(word in e.text.lower() for word in ["method", "gaussian", "deformation", "smpl", "dataset", "metric"]) else 1,
            e.page_number or 999,
            e.reading_order,
        ))
        figures = [e for e in elements if e.element_type == "figure" and not e.metadata.get("is_duplicate") and not e.metadata.get("is_decorative")]
        for element in text_elements[:8]:
            add(records, seen, record(
                paper.paper_id,
                "text_fact",
                f"What fact is stated by evidence {element.element_id} on page {element.page_number}?",
                element.text[:240],
                [element.element_id],
                [element.page_number],
                True,
                timestamp,
            ))
        for element in text_elements:
            if any(word in element.text.lower() for word in ["method", "gaussian", "deformation", "smpl", "initial"]):
                add(records, seen, record(paper.paper_id, "method_detail", f"What method detail does this paper report on page {element.page_number}?", element.text[:240], [element.element_id], [element.page_number], True, timestamp))
                if len([r for r in records if r["query_type"] == "method_detail" and paper.paper_id in r["paper_ids"]]) >= 3:
                    break
        for figure in figures[:4]:
            caption = figure.metadata.get("caption") or figure.text
            linked = [figure.element_id] + list(figure.linked_element_ids[:3])
            add(records, seen, record(paper.paper_id, "text_to_figure", f"Find the figure about {caption[:80]}", caption[:240], linked, [figure.page_number], True, timestamp, query_image_element_id=None))
            add(records, seen, record(paper.paper_id, "image_to_text", f"Using the figure image, retrieve its caption or related text.", caption[:240], linked, [figure.page_number], True, timestamp, query_image_element_id=figure.element_id))
            add(records, seen, record(paper.paper_id, "image_to_image", f"Find visually or semantically similar figures.", caption[:240], [figure.element_id], [figure.page_number], True, timestamp, query_image_element_id=figure.element_id))
            add(records, seen, record(paper.paper_id, "pair_to_any", f"Use this figure and caption to retrieve related evidence.", caption[:240], linked, [figure.page_number], True, timestamp, query_image_element_id=figure.element_id))
        add(records, seen, record(
            paper.paper_id,
            "unanswerable",
            f"What private deployment latency does {paper.metadata.title or paper.paper_id} report on an undisclosed production dataset?",
            "The paper does not provide evidence for this private deployment latency.",
            [],
            [],
            False,
            timestamp,
        ))
    if len(papers) >= 2:
        ids = [paper.paper_id for paper in papers[:2]]
        evidence = []
        pages = []
        for paper_id in ids:
            elems = [
                elem for elem in store.list_elements(paper_id=paper_id, element_types=["abstract", "paragraph"])
                if any(word in elem.text.lower() for word in ["method", "gaussian", "avatar", "deformation", "smpl"])
            ] or store.list_elements(paper_id=paper_id, element_types=["abstract", "paragraph"])
            if elems:
                evidence.append(elems[0].element_id)
                pages.append(elems[0].page_number)
        evidence_texts = []
        for element_id in evidence:
            element = store.get_element(element_id)
            if element:
                evidence_texts.append(f"{element.paper_id}: {element.text[:220]}")
        for index in range(8):
            add(records, seen, {
                "query_id": f"sf_cross_paper_{index:03d}",
                "paper_ids": ids,
                "query_type": "cross_paper_comparison",
                "question": f"Compare the technical evidence provided by these two papers for method route dimension {index + 1}.",
                "reference_answer_candidate": " | ".join(evidence_texts)[:520],
                "relevant_element_ids": evidence,
                "relevant_page_numbers": pages,
                "answerable": bool(evidence),
                "difficulty": "hard",
                "generation_model": "scholarfusion_lite_builder_v1",
                "generation_timestamp": timestamp,
                "review_status": "pending_review",
            })
    return records[:target_count]


def record(paper_id: str, query_type: str, question: str, answer: str, evidence_ids: List[str], pages: List[Any], answerable: bool, timestamp: str, query_image_element_id: Any = None) -> Dict[str, Any]:
    return {
        "query_id": f"sf_{paper_id}_{query_type}_{abs(hash(question + str(evidence_ids))) % 100000:05d}",
        "paper_ids": [paper_id],
        "query_type": query_type,
        "question": question,
        "query_image_element_id": query_image_element_id,
        "reference_answer_candidate": answer,
        "relevant_element_ids": list(dict.fromkeys([item for item in evidence_ids if item])),
        "relevant_page_numbers": list(dict.fromkeys([page for page in pages if page is not None])),
        "answerable": answerable,
        "difficulty": "medium" if query_type not in {"cross_paper_comparison", "unanswerable"} else "hard",
        "generation_model": "scholarfusion_lite_builder_v1",
        "generation_timestamp": timestamp,
        "review_status": "pending_review",
    }


def add(records: List[Dict[str, Any]], seen: set[str], item: Dict[str, Any]) -> None:
    key = f"{item['query_type']}:{item['question']}:{item.get('query_image_element_id')}"
    if key in seen:
        return
    seen.add(key)
    records.append(item)


def write_review_queue(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = ["query_id", "paper_ids", "query_type", "question", "query_image_element_id", "reference_answer_candidate", "relevant_element_ids", "relevant_page_numbers", "answerable", "review_status", "human_verified_answer", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = {field: item.get(field, "") for field in fields}
            row["paper_ids"] = ";".join(item.get("paper_ids", []))
            row["relevant_element_ids"] = ";".join(item.get("relevant_element_ids", []))
            row["relevant_page_numbers"] = ";".join(str(page) for page in item.get("relevant_page_numbers", []))
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
