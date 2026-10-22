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
from core.research_store import ResearchStore
from core.research_evaluation import ResearchEvaluationRunner, EvaluationItem
from core.research_service import ResearchService
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.phase2_citations import citation_quality_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ScholarFusion Lite provisional evaluation.")
    parser.add_argument("--db-path", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion/scholarfusion.sqlite3"))
    parser.add_argument("--index-db-path", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion/scholarfusion_index.sqlite3"))
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_lite_v1/candidates.jsonl")
    parser.add_argument("--output", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion/evaluation_metrics.json"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records(Path(args.dataset))
    service = ResearchService(store=ResearchStore(args.db_path))
    service.set_vector_index(ScholarFusionVectorIndex(args.index_db_path))
    retrieved: Dict[str, List[str]] = {}
    citations: Dict[str, List[Dict[str, Any]]] = {}
    primary_citations: Dict[str, List[Dict[str, Any]]] = {}
    predictions: Dict[str, bool] = {}
    citation_records: List[Dict[str, Any]] = []
    retrieval_latencies: List[float] = []
    qa_latencies: List[float] = []
    for record in records:
        start = time.perf_counter()
        if record.get("query_image_element_id"):
            element = service.store.get_element(record["query_image_element_id"])
            image = element.metadata.get("image_path") if element else None
            results = service.search_any_to_any(
                query_text=record["question"] if record["query_type"] == "pair_to_any" else None,
                query_image=image,
                target_modalities=["text", "figure", "pair", "page"],
                top_k=args.top_k,
            ) if image else []
        else:
            results = service.search_any_to_any(query_text=record["question"], target_modalities=["text", "figure", "pair", "page"], top_k=args.top_k)
        retrieval_latencies.append((time.perf_counter() - start) * 1000)
        retrieved[record["query_id"]] = [item.get("element_id") or item.get("object_id") for item in results]
        qa_start = time.perf_counter()
        qa = service.qa(record["question"], top_k=5)
        qa_latencies.append((time.perf_counter() - qa_start) * 1000)
        citations[record["query_id"]] = qa.get("citations", [])
        primary_citations[record["query_id"]] = qa.get("primary_citations", [])
        predictions[record["query_id"]] = bool(qa.get("answerable"))
        citation_records.append({
            "query_id": record["query_id"],
            "relevant_element_ids": record.get("relevant_element_ids", []),
            "primary_citation_ids": [item.get("element_id") for item in qa.get("primary_citations", []) if item.get("element_id")],
            "context_citation_ids": [item.get("element_id") for item in qa.get("context_citations", []) if item.get("element_id")],
            "claim_support": qa.get("context_manifest", {}).get("claim_evidence", []),
            "page_valid": all(item.get("page_number") is not None for item in qa.get("citations", [])),
        })
    items = [
        EvaluationItem(
            query_id=record["query_id"],
            query_type=record["query_type"],
            input_modality="text",
            target_modality=["any"],
            question=record["question"],
            reference_answer=record.get("reference_answer_candidate", ""),
            relevant_element_ids=record.get("relevant_element_ids", []),
            answerable=bool(record.get("answerable")),
            difficulty=record.get("difficulty", "medium"),
            review_status=record.get("review_status", "pending_review"),
        )
        for record in records
    ]
    runner = ResearchEvaluationRunner()
    metrics = {}
    metrics.update(runner.evaluate_retrieval(items, retrieved, k_values=(5, 10)))
    metrics.update(runner.evaluate_citations(items, citations))
    primary_metrics = runner.evaluate_citations(items, primary_citations)
    metrics["primary_citation_precision"] = primary_metrics["citation_precision"]
    metrics["primary_citation_recall"] = primary_metrics["citation_recall"]
    metrics.update(citation_quality_metrics(citation_records))
    metrics.update(runner.evaluate_answerability(items, predictions))
    metrics["not_gold"] = any(item.review_status != "human_verified" for item in items)
    metrics["dataset_status"] = "provisional" if metrics["not_gold"] else "gold"
    metrics["item_count"] = len(items)
    metrics["p50_retrieval_latency_ms"] = percentile(retrieval_latencies, 50)
    metrics["p95_retrieval_latency_ms"] = percentile(retrieval_latencies, 95)
    metrics["p50_qa_latency_ms"] = percentile(qa_latencies, 50)
    metrics["p95_qa_latency_ms"] = percentile(qa_latencies, 95)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    weight = index - lo
    return round(values[lo] * (1 - weight) + values[hi] * weight, 3)


if __name__ == "__main__":
    raise SystemExit(main())
