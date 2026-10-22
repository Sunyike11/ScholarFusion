from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.phase3_production_retrieval import Phase3ProductionRetriever, ProtectedWeightedRRF
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_phase21_metrics import retrieval_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Phase3 RRF fusion.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/phase3/rrf_debug.json")
    args = parser.parse_args()
    store = ResearchStore(args.db_path)
    retriever = Phase3ProductionRetriever(store, ScholarFusionVectorIndex(args.index_db_path))
    retriever.prepare(force=False)
    records = load_records(Path(args.dataset))
    dense_retrieved = {}
    hybrid_retrieved = {}
    debug_rows = []
    for record in records:
        image = image_for(record, store)
        dense_results = retriever._sqlite_vector(
            "research_multimodal_elements",
            retriever._query_vector(record["question"], image, record["query_type"]),
            "multimodal_dense",
            10,
        )
        hybrid_results, trace = retriever.search(record["question"], image, record["query_type"], top_k=10)
        dense_ids = [item.element_id or item.object_id for item in dense_results]
        hybrid_ids = [item.element_id or item.object_id for item in hybrid_results]
        dense_retrieved[record["query_id"]] = dense_ids
        hybrid_retrieved[record["query_id"]] = hybrid_ids
        sources = trace.pre_fusion_rank
        all_candidates = [candidate for values in sources.values() for candidate in values]
        duplicate_ratio = 1 - len(set(all_candidates)) / max(len(all_candidates), 1)
        debug_rows.append({
            "query_id": record["query_id"],
            "query_type": record["query_type"],
            "dense_topk": dense_ids,
            "bm25_topk": sources.get("bm25", []),
            "chroma_topk": sources.get("chroma_dense", []),
            "hybrid_topk": hybrid_ids,
            "candidate_overlap_dense_hybrid": len(set(dense_ids) & set(hybrid_ids)),
            "duplicate_ratio": duplicate_ratio,
            "weight_contribution": retriever.rrf.weights,
            "pre_fusion_rank": sources,
            "post_fusion_rank": trace.post_fusion_rank,
            "score_normalization": "RRF rank-only; raw score not normalized",
            "top_k_truncation": 10,
        })
    dense_metrics = retrieval_metrics(records, dense_retrieved, (10,))
    hybrid_metrics = retrieval_metrics(records, hybrid_retrieved, (10,))
    payload = {
        "status": "ok",
        "dense_recall@10": dense_metrics["recall@10"],
        "hybrid_recall@10": hybrid_metrics["recall@10"],
        "hybrid_not_worse_than_dense": hybrid_metrics["recall@10"] >= dense_metrics["recall@10"],
        "rows": debug_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["status", "dense_recall@10", "hybrid_recall@10", "hybrid_not_worse_than_dense"]}, ensure_ascii=False))
    return 0


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_for(record: Dict[str, Any], store: ResearchStore) -> Any:
    element_id = record.get("query_image_element_id")
    if not element_id:
        return None
    element = store.get_element(element_id)
    return element.metadata.get("image_path") if element else None


if __name__ == "__main__":
    raise SystemExit(main())
