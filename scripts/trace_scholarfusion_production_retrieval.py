from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_service import ResearchService
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace ScholarFusion production retrieval path safely.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/scholarfusion_phase21/production_retrieval_trace.json")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = ResearchService(store=ResearchStore(args.db_path))
    service.set_vector_index(ScholarFusionVectorIndex(args.index_db_path))
    records = load_records(Path(args.dataset))
    selected = select_trace_records(records)
    traces = []
    for record in selected:
        start = time.perf_counter()
        image = image_for(record, service)
        results = service.search_any_to_any(
            query_text=record["question"] if record.get("query_type") != "image_to_image" else None,
            query_image=image,
            target_modalities=["text", "figure", "page", "pair"],
            top_k=args.top_k,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        top_sources = sorted({source for item in results for source in item.get("retrieval_sources", [])})
        configured_sources = configured_for(record)
        traces.append({
            "query_id": record["query_id"],
            "query_type": record.get("query_type"),
            "query_modality": modality(record),
            "retrievers_invoked": configured_sources,
            "retrievers_in_top_results": top_sources,
            "collections_used": collections_from_sources(configured_sources),
            "provider_used": "fake_embedding_provider_default",
            "bm25_enabled": "bm25" in configured_sources,
            "chroma_text_dense_enabled": False,
            "chroma_text_dense_status": "legacy_adapter_not_initialized",
            "multimodal_dense_enabled": "multimodal_dense" in configured_sources,
            "page_visual_enabled": "page_visual" in configured_sources,
            "paper_level_enabled": "paper_level" in configured_sources,
            "graph_enabled": "graph" in configured_sources,
            "fusion_method": "WeightedRRF",
            "reranker_requested": True,
            "reranker_actual_provider": getattr(service.reranker, "provider_name", type(service.reranker).__name__),
            "pre_fusion_ranks": [],
            "post_fusion_ranks": [
                {
                    "rank": index,
                    "object_id": item.get("object_id"),
                    "element_id": item.get("element_id"),
                    "sources": item.get("retrieval_sources", []),
                    "score": item.get("fusion_score"),
                }
                for index, item in enumerate(results[: args.top_k], start=1)
            ],
            "post_rerank_ranks": [
                {
                    "rank": index,
                    "object_id": item.get("object_id"),
                    "reranker_provider": item.get("metadata", {}).get("reranker_provider") or getattr(service.reranker, "provider_name", "unknown"),
                    "rerank_trace": item.get("metadata", {}).get("rerank_trace", {}),
                }
                for index, item in enumerate(results[: args.top_k], start=1)
            ],
            "latency_by_stage": {"total_ms": latency_ms},
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_status": "provisional",
        "review_status": "model_reviewed",
        "not_gold": True,
        "trace_count": len(traces),
        "traces": traces,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "trace_count": len(traces), "output": str(output)}, ensure_ascii=False))
    return 0


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_trace_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = ["text_fact", "image_to_text", "pair_to_any", "text_to_figure", "cross_paper_comparison"]
    selected = []
    for qtype in wanted:
        item = next((record for record in records if record.get("query_type") == qtype), None)
        if item:
            selected.append(item)
    return selected


def image_for(record: Dict[str, Any], service: ResearchService) -> Optional[Any]:
    element_id = record.get("query_image_element_id")
    if not element_id:
        return None
    element = service.store.get_element(element_id)
    return element.metadata.get("image_path") if element else None


def modality(record: Dict[str, Any]) -> str:
    qtype = record.get("query_type")
    if qtype == "pair_to_any":
        return "text_image_pair"
    if qtype in {"image_to_text", "image_to_image"}:
        return "image"
    return "text"


def configured_for(record: Dict[str, Any]) -> List[str]:
    qmod = modality(record)
    sources = []
    if qmod in {"text", "text_image_pair"}:
        sources.extend(["bm25", "text_dense"])
    sources.extend(["multimodal_dense", "page_visual", "paper_level"])
    if qmod in {"text", "text_image_pair"}:
        sources.append("graph")
    return sources


def collections_from_sources(sources: List[str]) -> List[str]:
    mapping = {
        "text_dense": AppSettings.SCHOLARFUSION_TEXT_COLLECTION,
        "multimodal_dense": AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION,
        "page_visual": AppSettings.SCHOLARFUSION_PAGE_COLLECTION,
        "paper_level": AppSettings.SCHOLARFUSION_PAPER_COLLECTION,
    }
    return [mapping[source] for source in sources if source in mapping]


if __name__ == "__main__":
    raise SystemExit(main())
