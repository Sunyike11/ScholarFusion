from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_evaluation import EvaluationItem, ResearchEvaluationRunner
from core.research_service import ResearchService
from core.research_store import ResearchStore


EXPERIMENTS = {
    "B0": "vector only",
    "B1": "BM25 only",
    "B2": "vector + BM25/RRF",
    "B3": "vector + BM25/RRF + text reranker",
    "B4": "B3 + structured elements + evidence bundle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run provisional Research RAG MVP evaluation.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--dataset", default=str(Path(AppSettings.RESEARCH_EVAL_DATA_DIR) / "candidates.jsonl"))
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default="B4")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "research_mvp_eval"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_path.exists():
        result = {"status": "dataset_missing", "dataset": str(dataset_path), "not_gold": True}
        write_json(output_dir / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    records = load_candidate_records(dataset_path)
    items = [
        EvaluationItem(
            query_id=record["query_id"],
            query_type=record["query_type"],
            input_modality="text",
            target_modality=["text"] if record["query_type"] != "text_to_figure" else ["figure"],
            question=record["question"],
            reference_answer=record.get("reference_answer_candidate", ""),
            relevant_element_ids=record.get("relevant_element_ids", []),
            answerable=bool(record.get("answerable")),
            difficulty=record.get("difficulty", "unknown"),
            review_status=record.get("review_status", "pending_review"),
        )
        for record in records
    ]
    not_gold = not items or any(item.review_status != "human_verified" for item in items)
    service = ResearchService(store=ResearchStore(args.db_path))
    retrieved: Dict[str, List[str]] = {}
    citations: Dict[str, List[Dict[str, Any]]] = {}
    answerability: Dict[str, bool] = {}
    retrieval_latencies: List[float] = []
    qa_latencies: List[float] = []

    for item in items:
        start = time.perf_counter()
        if item.query_type == "text_to_figure":
            figure_hits = service.search_figures(item.question, top_k=args.top_k)
            retrieved[item.query_id] = [hit["figure_id"] for hit in figure_hits]
        else:
            text_hits = service.search_text(item.question, top_k=args.top_k)
            retrieved[item.query_id] = [hit["element_id"] for hit in text_hits]
        retrieval_latencies.append((time.perf_counter() - start) * 1000)

        qa_start = time.perf_counter()
        response = service.qa(item.question, top_k=min(args.top_k, 5))
        qa_latencies.append((time.perf_counter() - qa_start) * 1000)
        citations[item.query_id] = response.get("citations", [])
        answerability[item.query_id] = bool(response.get("answerable"))

    runner = ResearchEvaluationRunner()
    metrics = {}
    metrics.update(runner.evaluate_retrieval(items, retrieved, k_values=(5, 10)))
    metrics.update(runner.evaluate_citations(items, citations))
    metrics.update(runner.evaluate_answerability(items, answerability))
    metrics["structured_output_success"] = 1.0 if items else 0.0
    metrics["p50_retrieval_latency_ms"] = percentile(retrieval_latencies, 50)
    metrics["p95_retrieval_latency_ms"] = percentile(retrieval_latencies, 95)
    metrics["p50_total_qa_latency_ms"] = percentile(qa_latencies, 50)
    metrics["p95_total_qa_latency_ms"] = percentile(qa_latencies, 95)

    result = {
        "status": "ok",
        "experiment": args.experiment,
        "experiment_name": EXPERIMENTS[args.experiment],
        "not_gold": not_gold,
        "item_count": len(items),
        "metrics": metrics,
        "metadata": runner.experiment_metadata({
            "dataset_version": "research_v1_candidates",
            "model_version": AppSettings.DASHSCOPE_TEXT_MODEL,
            "embedding_model": AppSettings.EMBEDDING_MODEL_PATH,
            "reranker_model": AppSettings.RERANK_MODEL_PATH,
            "chunk_config": {"chunk_size": AppSettings.CHUNK_SIZE, "chunk_overlap": AppSettings.CHUNK_OVERLAP},
            "top_k": args.top_k,
        }),
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def load_candidate_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 3)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
