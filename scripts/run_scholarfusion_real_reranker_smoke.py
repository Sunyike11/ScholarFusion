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

from core.phase2_reranker_provider import DeterministicFallbackReranker, Qwen3VLRerankerProvider
from core.research_service import ResearchService
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_retrieval import AnyToAnyResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small real qwen3-vl-rerank smoke.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/scholarfusion_phase21/real_reranker_smoke.json")
    parser.add_argument("--recall-top-n", type=int, default=8)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = ResearchService(store=ResearchStore(args.db_path))
    service.set_vector_index(ScholarFusionVectorIndex(args.index_db_path))
    records = load_records(Path(args.dataset))
    selected = select_records(records)
    traces = []
    provider_name = "qwen3-vl-rerank"
    fallback_count = 0
    request_count = 0
    for record in selected:
        image = image_for(record, service)
        pre = service.hybrid_retriever.search(
            query_text=record["question"] if record.get("query_type") != "image_to_image" else None,
            query_image=image,
            target_modalities=["text", "figure", "page", "pair"],
            top_k=args.recall_top_n,
            include_graph=True,
        )
        candidates = [
            AnyToAnyResult(
                object_id=item.result_id,
                paper_id=item.paper_id,
                element_id=item.element_id,
                matched_modality=item.modality,
                fusion_score=item.fusion_score,
                retrieval_sources=item.retrieval_sources,
                raw_scores=item.raw_scores,
                rank_positions=item.rank_positions,
                metadata=item.metadata,
                text=item.text,
            )
            for item in pre
        ]
        start = time.perf_counter()
        actual_provider = provider_name
        try:
            reranker = Qwen3VLRerankerProvider()
            post = reranker.rerank(record["question"], candidates, args.rerank_top_n)
            request_count += 1
        except Exception as exc:
            actual_provider = "deterministic_fallback"
            fallback_count += 1
            post = DeterministicFallbackReranker().rerank(record["question"], candidates, args.rerank_top_n)
            safe_error = str(exc)[:220]
        else:
            safe_error = ""
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        before = [item.result_id for item in pre[: args.rerank_top_n]]
        after = [item.object_id for item in post[: args.rerank_top_n]]
        traces.append({
            "query_id": record["query_id"],
            "query_type": record.get("query_type"),
            "candidate_modalities": sorted({item.modality for item in pre}),
            "actual_provider": actual_provider,
            "before_rerank": before,
            "after_rerank": after,
            "rank_changed": before != after,
            "latency_ms": latency_ms,
            "safe_error": safe_error,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "dataset_status": "provisional",
        "review_status": "model_reviewed",
        "not_gold": True,
        "requested_provider": provider_name,
        "request_count": request_count,
        "fallback_count": fallback_count,
        "rank_change_count": len([trace for trace in traces if trace["rank_changed"]]),
        "traces": traces,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "request_count", "fallback_count", "rank_change_count"]}, ensure_ascii=False))
    return 0


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = ["text_fact", "image_to_text", "image_to_image", "pair_to_any"]
    return [next(record for record in records if record.get("query_type") == qtype) for qtype in wanted if any(record.get("query_type") == qtype for record in records)]


def image_for(record: Dict[str, Any], service: ResearchService) -> Any:
    element_id = record.get("query_image_element_id")
    if not element_id:
        return None
    element = service.store.get_element(element_id)
    return element.metadata.get("image_path") if element else None


if __name__ == "__main__":
    raise SystemExit(main())
