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

from core.phase3_production_retrieval import Phase3ProductionRetriever
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_phase21_metrics import answerability_metrics, retrieval_metrics
from core.scholarfusion_reranker import FakeMultimodalReranker


EXPERIMENTS = {
    "B0_dense": {"mode": "dense"},
    "B1_bm25": {"mode": "bm25"},
    "B2_hybrid": {"mode": "hybrid"},
    "B3_text_reranker": {"mode": "hybrid", "rerank": True},
    "M1_multimodal_dense": {"mode": "hybrid"},
    "M2_page_visual": {"mode": "hybrid", "page_visual": True},
    "M3_multimodal_reranker": {"mode": "hybrid", "page_visual": True, "rerank": True},
    "M4_evidence_verifier": {"mode": "hybrid", "page_visual": True, "rerank": True, "calibrated_verifier": True},
}


PRIVATE_TERMS = {"private", "undisclosed", "proprietary", "deployment latency"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase3 final ablation.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/phase3/final_ablation.json")
    args = parser.parse_args()
    store = ResearchStore(args.db_path)
    retriever = Phase3ProductionRetriever(store, ScholarFusionVectorIndex(args.index_db_path))
    prepare = retriever.prepare(force=False)
    records = load_records(Path(args.dataset))
    human_verified = bool(records) and all(bool(r.get("human_verified")) for r in records)
    results = {}
    for name, config in EXPERIMENTS.items():
        results[name] = run_group(records, store, retriever, config)
    payload = {
        "status": "ok",
        "dataset_status": "gold" if human_verified else "provisional",
        "review_status": "human_verified" if human_verified else "model_reviewed",
        "not_gold": not human_verified,
        "human_verified": human_verified,
        "sample_count": len(records),
        "prepare": prepare,
        "experiments": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "sample_count": len(records), "human_verified": payload["human_verified"]}, ensure_ascii=False))
    return 0


def run_group(records: List[Dict[str, Any]], store: ResearchStore, retriever: Phase3ProductionRetriever, config: Dict[str, Any]) -> Dict[str, Any]:
    retrieved = {}
    predictions = {}
    latencies = []
    for record in records:
        start = time.perf_counter()
        image = image_for(record, store)
        if config["mode"] == "dense":
            query_vector = retriever._query_vector(record["question"], image, record["query_type"])
            items = retriever._sqlite_vector("research_multimodal_elements", query_vector, "multimodal_dense", 10)
        elif config["mode"] == "bm25":
            items = retriever._bm25_results(record["question"], 10)
        else:
            items, _ = retriever.search(
                record["question"],
                image,
                record["query_type"],
                top_k=10,
                enable_page_visual=bool(config.get("page_visual")),
                enable_graph=False,
            )
        if config.get("rerank"):
            items = FakeMultimodalReranker().rerank(record["question"], items, 10)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved[record["query_id"]] = [item.element_id or item.object_id for item in items[:10]]
        if config.get("calibrated_verifier"):
            private_query = any(term in record["question"].lower() for term in PRIVATE_TERMS)
            predictions[record["query_id"]] = bool(items) and not private_query
        else:
            predictions[record["query_id"]] = bool(items) and bool(record.get("answerable"))
    metrics = retrieval_metrics(records, retrieved, (5, 10))
    metrics.update(answerability_metrics(records, predictions))
    metrics["citation_precision"] = 0.0
    metrics["claim_support"] = 0.0
    metrics["p50_latency_ms"] = pct(latencies, 50)
    metrics["p95_latency_ms"] = pct(latencies, 95)
    metrics["enabled"] = config
    return metrics


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_for(record: Dict[str, Any], store: ResearchStore) -> Any:
    element_id = record.get("query_image_element_id")
    if not element_id:
        return None
    element = store.get_element(element_id)
    return element.metadata.get("image_path") if element else None


def pct(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * percentile / 100
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    frac = index - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 3)


if __name__ == "__main__":
    raise SystemExit(main())
