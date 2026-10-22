from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_production_retrieval import Phase3ProductionRetriever
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace Phase3 production retrieval path.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/phase3/production_trace.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    store = ResearchStore(args.db_path)
    retriever = Phase3ProductionRetriever(store, ScholarFusionVectorIndex(args.index_db_path))
    prepare = retriever.prepare(force=args.force)
    records = load_records(Path(args.dataset))
    selected = select(records)
    traces = []
    for record in selected:
        image = image_for(record, store)
        results, trace = retriever.search(
            query=record["question"],
            query_image=image,
            query_type=record["query_type"],
            top_k=10,
        )
        traces.append({
            "query_id": record["query_id"],
            "query_type": record["query_type"],
            "retrievers_invoked": trace.retrievers_invoked,
            "collections_used": trace.collections_used,
            "provider_used": "fake_embedding_provider_default",
            "bm25_enabled": "bm25" in trace.retrievers_invoked,
            "chroma_dense_enabled": "chroma_dense" in trace.retrievers_invoked,
            "multimodal_dense_enabled": "multimodal_dense" in trace.retrievers_invoked,
            "page_visual_enabled": "page_visual" in trace.retrievers_invoked,
            "rrf_enabled": True,
            "reranker_enabled": False,
            "verifier_enabled": False,
            "pre_fusion_rank": trace.pre_fusion_rank,
            "post_fusion_rank": trace.post_fusion_rank,
            "post_rerank_rank": trace.post_rerank_rank,
            "latency_by_stage": trace.latency_by_stage,
        })
    payload = {
        "status": "ok",
        "prepare": prepare,
        "trace_count": len(traces),
        "production_path": "phase3_chroma_bm25_multimodal_rrf",
        "traces": traces,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "trace_count": len(traces), "prepare": prepare}, ensure_ascii=False))
    return 0


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = ["text_fact", "text_to_figure", "image_to_text", "pair_to_any", "cross_paper_comparison"]
    out = []
    for qtype in wanted:
        item = next((record for record in records if record.get("query_type") == qtype), None)
        if item:
            out.append(item)
    return out


def image_for(record: Dict[str, Any], store: ResearchStore) -> Any:
    element_id = record.get("query_image_element_id")
    if not element_id:
        return None
    element = store.get_element(element_id)
    return element.metadata.get("image_path") if element else None


if __name__ == "__main__":
    raise SystemExit(main())
