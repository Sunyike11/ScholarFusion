from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_store import ResearchStore
from core.scholarfusion_model_review import (
    DashScopeEvaluationItemReviewer,
    EvaluationItemReviewer,
    ModelReviewedDatasetValidator,
    apply_review,
    build_review_payload,
    deduplicate_records,
)


TARGETS = {
    "text_fact": 6,
    "method_detail": 5,
    "text_to_figure": 6,
    "image_to_text": 5,
    "image_to_image": 4,
    "pair_to_any": 4,
    "cross_paper_comparison": 6,
    "unanswerable": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review ScholarFusion eval candidates with an independent model.")
    parser.add_argument("--db-path", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion_phase2/scholarfusion.sqlite3"))
    parser.add_argument("--candidates", default="evaluation_data/scholarfusion_lite_v1/candidates.jsonl")
    parser.add_argument("--output-dir", default="evaluation_data/scholarfusion_model_reviewed_v1")
    parser.add_argument("--real-reviewer", action="store_true")
    parser.add_argument("--max-items", type=int, default=48)
    parser.add_argument("--max-requests", type=int, default=48)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "review_report.json"
    if report_path.exists() and not args.force:
        print(json.dumps({"status": "exists_without_force", "path": str(report_path)}, ensure_ascii=False))
        return 0

    store = ResearchStore(args.db_path)
    raw_records = load_jsonl(Path(args.candidates))
    selected = select_balanced(raw_records, args.max_items)
    write_jsonl(output_dir / "raw_candidates.jsonl", selected)

    reviewer = DashScopeEvaluationItemReviewer() if args.real_reviewer else EvaluationItemReviewer()
    validator = ModelReviewedDatasetValidator(store)
    review_rows = []
    accepted_candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    request_count = 0

    for record in selected:
        payload = build_review_payload(record, store)
        if args.real_reviewer and request_count >= args.max_requests:
            rejected.append(record | {"reject_reason": "review_request_limit"})
            continue
        try:
            review = reviewer.review(payload)
            request_count += 1 if args.real_reviewer else 0
        except Exception as exc:
            rejected.append(record | {"reject_reason": "reviewer_error", "safe_error": str(exc)[:240]})
            continue
        reviewed = apply_review(record, review)
        validation = validator.validate(reviewed)
        reviewed.update(validation)
        keep = (
            review.decision in {"approve", "revise"}
            and review.confidence >= args.confidence_threshold
            and validation["validation_status"] == "valid"
            and not reviewed.get("human_verified")
        )
        review_rows.append({
            "query_id": record.get("query_id"),
            "query_type": record.get("query_type"),
            "decision": review.decision,
            "confidence": review.confidence,
            "validation_status": validation["validation_status"],
            "reason_codes": review.reason_codes,
            "reviewer_model": review.reviewer_model,
        })
        if keep:
            accepted_candidates.append(reviewed)
        else:
            rejected.append(reviewed | {"reject_reason": "low_confidence_or_validation"})

    accepted, duplicate_rejects = deduplicate_records(accepted_candidates)
    rejected.extend(duplicate_rejects)
    write_jsonl(output_dir / "review_results.jsonl", review_rows)
    write_jsonl(output_dir / "provisional.jsonl", accepted)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_spot_check(output_dir / "spot_check.csv", accepted, store)

    report = {
        "status": "ok",
        "dataset_version": "scholarfusion_model_reviewed_v1",
        "dataset_status": "provisional",
        "review_status": "model_reviewed",
        "not_gold": True,
        "human_verified": False,
        "source_candidate_count": len(raw_records),
        "selected_candidate_count": len(selected),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "reviewer_model": reviewer.provider_name if not args.real_reviewer else getattr(reviewer, "model", reviewer.provider_name),
        "real_reviewer": bool(args.real_reviewer),
        "confidence_threshold": args.confidence_threshold,
        "request_count": request_count,
        "decision_counts": Counter(row["decision"] for row in review_rows),
        "accepted_type_counts": Counter(row.get("query_type") for row in accepted),
        "rejected_type_counts": Counter(row.get("query_type") for row in rejected),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    serializable_report = json.loads(json.dumps(report, ensure_ascii=False, default=dict))
    (output_dir / "review_report.json").write_text(json.dumps(serializable_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "real_reviewer": bool(args.real_reviewer),
        "request_count": request_count,
    }, ensure_ascii=False))
    return 0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def select_balanced(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        qtype = record.get("query_type")
        if counts[qtype] >= TARGETS.get(qtype, 0):
            continue
        selected.append(record)
        counts[qtype] += 1
        if len(selected) >= limit:
            break
    return selected


def write_spot_check(path: Path, rows: List[Dict[str, Any]], store: ResearchStore) -> None:
    wanted = {
        "text_fact": 1,
        "text_to_figure": 1,
        "image_to_text": 1,
        "image_to_image": 1,
        "pair_to_any": 1,
        "cross_paper_comparison": 1,
        "unanswerable": 2,
    }
    selected = []
    counts: Counter[str] = Counter()
    for row in rows:
        qtype = row.get("query_type")
        if counts[qtype] >= wanted.get(qtype, 0):
            continue
        selected.append(row)
        counts[qtype] += 1
    fields = [
        "query_id", "query_type", "question", "reference_answer", "answerable",
        "primary_evidence_preview", "alternative_evidence_preview", "context_evidence_preview",
        "paper_ids", "page_numbers", "model_reviewer_confidence", "spot_check_status", "spot_check_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "query_id": row.get("query_id"),
                "query_type": row.get("query_type"),
                "question": row.get("question"),
                "reference_answer": row.get("reference_answer"),
                "answerable": row.get("answerable"),
                "primary_evidence_preview": preview(row.get("gold_primary_element_ids", []), store),
                "alternative_evidence_preview": preview(row.get("acceptable_alternative_element_ids", []), store),
                "context_evidence_preview": preview(row.get("gold_context_element_ids", []), store),
                "paper_ids": ";".join(row.get("paper_ids", [])),
                "page_numbers": ";".join(str(page) for page in row.get("relevant_page_numbers", [])),
                "model_reviewer_confidence": row.get("model_reviewer_confidence"),
                "spot_check_status": "",
                "spot_check_notes": "",
            })


def preview(element_ids: List[str], store: ResearchStore) -> str:
    chunks = []
    for element_id in element_ids[:2]:
        element = store.get_element(element_id)
        if element:
            chunks.append(f"{element.element_id}@p{element.page_number}:{(element.text or '')[:120]}")
    return " | ".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
