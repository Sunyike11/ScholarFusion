from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_models import PaperElement
from core.research_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pending-review Research RAG evaluation candidates.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--output-dir", default=AppSettings.RESEARCH_EVAL_DATA_DIR)
    parser.add_argument("--max-per-paper", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    candidates_path = output_dir / "candidates.jsonl"
    review_path = output_dir / "review_template.csv"
    if candidates_path.exists() and not args.force:
        print(f"exists_without_force: {candidates_path}")
        return 0

    store = ResearchStore(args.db_path)
    records = build_candidates(store, max_per_paper=args.max_per_paper)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    write_review_template(review_path, records)
    print(json.dumps({"status": "ok", "candidate_count": len(records), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


def build_candidates(store: ResearchStore, max_per_paper: int = 12) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_questions: Set[str] = set()
    timestamp = datetime.now(timezone.utc).isoformat()
    for paper in store.list_papers():
        paper_full = store.get_paper(paper.paper_id)
        if not paper_full:
            continue
        elements = [
            element for element in paper_full.elements
            if element.element_type in {"title", "abstract", "paragraph", "figure_caption", "heading"}
            and element.text.strip()
        ]
        per_paper: List[Dict[str, Any]] = []
        per_paper.extend(_text_fact_candidates(paper.paper_id, elements, timestamp))
        per_paper.extend(_figure_candidates(paper.paper_id, paper_full.elements, timestamp))
        per_paper.append(_unanswerable_candidate(paper.paper_id, timestamp))
        for record in per_paper:
            key = normalize_question(record["question"])
            if key in seen_questions:
                continue
            if validate_candidate(store, record):
                seen_questions.add(key)
                records.append(record)
            if len([item for item in records if paper.paper_id in item["paper_ids"]]) >= max_per_paper:
                break
    records.extend(_cross_paper_candidates(store, timestamp, seen_questions))
    return records


def _text_fact_candidates(paper_id: str, elements: List[PaperElement], timestamp: str) -> Iterable[Dict[str, Any]]:
    priority = [element for element in elements if element.element_type in {"abstract", "paragraph", "heading"}][:8]
    for index, element in enumerate(priority):
        query_type = "method_detail" if any(word in element.text.lower() for word in ["method", "deformation", "gaussian", "initial"]) else "text_fact"
        yield candidate_record(
            query_id=f"{paper_id}_{query_type}_{index:03d}",
            paper_ids=[paper_id],
            query_type=query_type,
            question=question_for_element(element, query_type),
            reference_answer_candidate=short_text(element.text, 260),
            relevant_element_ids=[element.element_id],
            relevant_page_numbers=[element.page_number] if element.page_number is not None else [],
            answerable=True,
            difficulty="easy",
            timestamp=timestamp,
        )


def _figure_candidates(paper_id: str, elements: List[PaperElement], timestamp: str) -> Iterable[Dict[str, Any]]:
    figures = [element for element in elements if element.element_type == "figure"][:4]
    for index, figure in enumerate(figures):
        linked = [element_id for element_id in figure.linked_element_ids if element_id]
        evidence_ids = [figure.element_id] + linked[:2]
        yield candidate_record(
            query_id=f"{paper_id}_text_to_figure_{index:03d}",
            paper_ids=[paper_id],
            query_type="text_to_figure",
            question=f"Which figure is relevant to: {short_text(figure.metadata.get('caption') or figure.text, 90)}?",
            reference_answer_candidate=short_text(figure.metadata.get("caption") or figure.text, 260),
            relevant_element_ids=evidence_ids,
            relevant_page_numbers=[figure.page_number] if figure.page_number is not None else [],
            answerable=bool(evidence_ids),
            difficulty="medium",
            timestamp=timestamp,
        )


def _unanswerable_candidate(paper_id: str, timestamp: str) -> Dict[str, Any]:
    return candidate_record(
        query_id=f"{paper_id}_unanswerable_000",
        paper_ids=[paper_id],
        query_type="unanswerable",
        question="What private deployment latency on an undisclosed production dataset does this paper report?",
        reference_answer_candidate="The paper does not provide evidence for this private deployment latency.",
        relevant_element_ids=[],
        relevant_page_numbers=[],
        answerable=False,
        difficulty="medium",
        timestamp=timestamp,
    )


def _cross_paper_candidates(store: ResearchStore, timestamp: str, seen_questions: Set[str]) -> List[Dict[str, Any]]:
    papers = store.list_papers()[:3]
    if len(papers) < 2:
        return []
    records: List[Dict[str, Any]] = []
    paper_ids = [paper.paper_id for paper in papers[:2]]
    evidence_ids = []
    pages = []
    for paper_id in paper_ids:
        elements = store.list_elements(paper_id=paper_id, element_types=["abstract", "paragraph"])
        if elements:
            evidence_ids.append(elements[0].element_id)
            if elements[0].page_number is not None:
                pages.append(elements[0].page_number)
    question = "How do these papers differ in their main method or task?"
    if normalize_question(question) not in seen_questions and evidence_ids:
        records.append(candidate_record(
            query_id="cross_paper_comparison_000",
            paper_ids=paper_ids,
            query_type="cross_paper_comparison",
            question=question,
            reference_answer_candidate="Pending human review: compare method/task evidence from the listed papers.",
            relevant_element_ids=evidence_ids,
            relevant_page_numbers=pages,
            answerable=True,
            difficulty="hard",
            timestamp=timestamp,
        ))
    return records


def candidate_record(
        query_id: str,
        paper_ids: List[str],
        query_type: str,
        question: str,
        reference_answer_candidate: str,
        relevant_element_ids: List[str],
        relevant_page_numbers: List[int],
        answerable: bool,
        difficulty: str,
        timestamp: str,
) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "paper_ids": paper_ids,
        "query_type": query_type,
        "question": question,
        "reference_answer_candidate": reference_answer_candidate,
        "relevant_element_ids": list(dict.fromkeys(relevant_element_ids)),
        "relevant_page_numbers": list(dict.fromkeys(relevant_page_numbers)),
        "answerable": answerable,
        "difficulty": difficulty,
        "generation_model": "deterministic_candidate_builder_v1",
        "generation_timestamp": timestamp,
        "review_status": "pending_review",
    }


def validate_candidate(store: ResearchStore, record: Dict[str, Any]) -> bool:
    if record["review_status"] != "pending_review":
        return False
    if not record["question"].strip() or normalize_question(record["question"]) in normalize_question(record["reference_answer_candidate"]):
        return False
    if record["answerable"] and not record["relevant_element_ids"]:
        return False
    if not record["answerable"] and record["relevant_element_ids"]:
        return False
    paper_ids = set(record["paper_ids"])
    for element_id in record["relevant_element_ids"]:
        element = store.get_element(element_id)
        if not element or element.paper_id not in paper_ids:
            return False
        if element.page_number is not None and element.page_number not in record["relevant_page_numbers"]:
            return False
    return True


def write_review_template(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "query_id", "paper_ids", "query_type", "question", "reference_answer_candidate",
        "relevant_element_ids", "relevant_page_numbers", "answerable", "difficulty",
        "review_status", "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["paper_ids"] = ";".join(record.get("paper_ids", []))
            row["relevant_element_ids"] = ";".join(record.get("relevant_element_ids", []))
            row["relevant_page_numbers"] = ";".join(str(page) for page in record.get("relevant_page_numbers", []))
            row["human_notes"] = ""
            writer.writerow(row)


def question_for_element(element: PaperElement, query_type: str) -> str:
    if query_type == "method_detail":
        return f"What method detail is supported by the evidence on page {element.page_number or 'unknown'}?"
    if element.element_type == "heading":
        return f"What section topic appears on page {element.page_number or 'unknown'}?"
    return f"What fact is stated on page {element.page_number or 'unknown'}?"


def normalize_question(text: str) -> str:
    return " ".join((text or "").lower().split())


def short_text(text: Any, max_len: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
