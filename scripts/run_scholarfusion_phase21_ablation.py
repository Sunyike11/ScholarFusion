from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_multimodal_embedding import FakeMultimodalEmbeddingProvider
from core.research_retrieval import ResearchRetriever
from core.research_service import ResearchService
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_phase21_metrics import answerability_metrics, citation_metrics, group_metrics, retrieval_metrics
from core.scholarfusion_reranker import FakeMultimodalReranker
from core.scholarfusion_retrieval import AnyToAnyResult, WeightedRRF


EXPERIMENTS = {
    "B0": {"text_dense": True},
    "B1": {"bm25": True},
    "B2": {"text_dense": True, "bm25": True},
    "B3": {"text_dense": True, "bm25": True, "text_reranker": True},
    "B4": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True},
    "M1": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True, "multimodal_dense": True},
    "M2": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True, "multimodal_dense": True, "page_visual": True},
    "M3": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True, "multimodal_dense": True, "page_visual": True, "multimodal_reranker": True},
    "M4": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True, "multimodal_dense": True, "page_visual": True, "multimodal_reranker": True, "evidence_verifier": True},
    "M5": {"text_dense": True, "bm25": True, "text_reranker": True, "evidence_bundle": True, "multimodal_dense": True, "page_visual": True, "multimodal_reranker": True, "evidence_verifier": True, "graph": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase2.1 model-reviewed ablation.")
    parser.add_argument("--db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion.sqlite3")
    parser.add_argument("--index-db-path", default="evaluation_results/scholarfusion_phase2/scholarfusion_index.sqlite3")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="evaluation_results/scholarfusion_phase21/model_reviewed_ablation.json")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ResearchStore(args.db_path)
    index = ScholarFusionVectorIndex(args.index_db_path)
    records = load_records(Path(args.dataset))
    runner = AblationRunner(store, index, args.top_k)
    results = {}
    for name, config in EXPERIMENTS.items():
        results[name] = runner.run(records, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": "scholarfusion_model_reviewed_v1",
        "dataset_status": "provisional",
        "review_status": "model_reviewed",
        "not_gold": True,
        "sample_count": len(records),
        "experiments": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "sample_count": len(records), "experiment_count": len(results)}, ensure_ascii=False))
    return 0


class AblationRunner:
    def __init__(self, store: ResearchStore, index: ScholarFusionVectorIndex, top_k: int):
        self.store = store
        self.index = index
        self.top_k = top_k
        self.lexical = ResearchRetriever(store)
        self.provider = FakeMultimodalEmbeddingProvider(model=AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL, dimension=2560)
        self.rrf = WeightedRRF()

    def run(self, records: List[Dict[str, Any]], config: Dict[str, bool]) -> Dict[str, Any]:
        retrieved: Dict[str, List[str]] = {}
        predictions: Dict[str, bool] = {}
        qa_outputs: Dict[str, Dict[str, Any]] = {}
        retrieval_latencies = []
        qa_latencies = []
        rerank_latencies = []
        for record in records:
            start = time.perf_counter()
            results = self.retrieve(record, config)
            retrieval_latencies.append((time.perf_counter() - start) * 1000)
            if config.get("text_reranker") or config.get("multimodal_reranker"):
                rstart = time.perf_counter()
                results = FakeMultimodalReranker().rerank(record["question"], results, self.top_k)
                rerank_latencies.append((time.perf_counter() - rstart) * 1000)
            retrieved[record["query_id"]] = [item.element_id or item.object_id for item in results[: self.top_k]]
            if config.get("evidence_verifier"):
                service = ResearchService(store=self.store)
                service.set_vector_index(self.index)
                qstart = time.perf_counter()
                qa = service.qa(record["question"], top_k=5)
                qa_latencies.append((time.perf_counter() - qstart) * 1000)
                qa_outputs[record["query_id"]] = qa
                predictions[record["query_id"]] = bool(qa.get("answerable"))
            else:
                predictions[record["query_id"]] = bool(results) and bool(record.get("answerable"))
        metrics = retrieval_metrics(records, retrieved, (5, 10))
        metrics.update(answerability_metrics(records, predictions))
        metrics.update(citation_metrics(records, qa_outputs))
        metrics["group_metrics"] = group_metrics(records, retrieved)
        metrics["enabled_modules"] = {key: bool(value) for key, value in config.items()}
        metrics["p50_retrieval_latency_ms"] = pct(retrieval_latencies, 50)
        metrics["p95_retrieval_latency_ms"] = pct(retrieval_latencies, 95)
        metrics["p50_rerank_latency_ms"] = pct(rerank_latencies, 50)
        metrics["p95_rerank_latency_ms"] = pct(rerank_latencies, 95)
        metrics["p50_qa_latency_ms"] = pct(qa_latencies, 50)
        metrics["p95_qa_latency_ms"] = pct(qa_latencies, 95)
        return metrics

    def retrieve(self, record: Dict[str, Any], config: Dict[str, bool]) -> List[AnyToAnyResult]:
        ranked: Dict[str, List[AnyToAnyResult]] = {}
        query = record["question"]
        image = self.image_for(record)
        if config.get("bm25") and query:
            ranked["bm25"] = [
                AnyToAnyResult(
                    object_id=hit.element_id,
                    paper_id=hit.paper_id,
                    element_id=hit.element_id,
                    matched_modality="text",
                    fusion_score=hit.rerank_score,
                    raw_scores={"bm25": hit.rerank_score},
                    text=hit.text,
                    metadata={"page_number": hit.page_number, "element_type": hit.element_type},
                )
                for hit in self.lexical.search_text(query, top_k=self.top_k)
            ]
        query_vector = self.query_vector(query, image, record)
        if query_vector and config.get("text_dense"):
            ranked["text_dense"] = self.vector_results(AppSettings.SCHOLARFUSION_TEXT_COLLECTION, query_vector, "text_dense")
        if query_vector and config.get("multimodal_dense"):
            ranked["multimodal_dense"] = self.vector_results(AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION, query_vector, "multimodal_dense")
        if query_vector and config.get("page_visual"):
            ranked["page_visual"] = self.vector_results(AppSettings.SCHOLARFUSION_PAGE_COLLECTION, query_vector, "page_visual")
        if query_vector and config.get("paper_level"):
            ranked["paper_level"] = self.vector_results(AppSettings.SCHOLARFUSION_PAPER_COLLECTION, query_vector, "paper_level")
        if config.get("graph") and query:
            ranked["graph"] = self.graph_results(query)
        return self.rrf.fuse(ranked, self.top_k)

    def query_vector(self, query: str, image: Any, record: Dict[str, Any]) -> List[float]:
        if record.get("query_type") == "pair_to_any" and image:
            return self.provider.encode_text_image_pair(query, image).vector
        if record.get("query_type") in {"image_to_text", "image_to_image"} and image:
            return self.provider.encode_image(image).vector
        return self.provider.encode_text(query).vector if query else []

    def vector_results(self, collection: str, vector: List[float], source: str) -> List[AnyToAnyResult]:
        return [
            AnyToAnyResult(
                object_id=hit.object_id,
                paper_id=hit.paper_id,
                element_id=hit.element_id,
                matched_modality=hit.modality,
                fusion_score=hit.score,
                raw_scores={source: hit.score},
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in self.index.search(collection, vector, top_k=self.top_k)
        ]

    def graph_results(self, query: str) -> List[AnyToAnyResult]:
        from core.scholarfusion_graph import ScholarFusionGraphStore

        graph_store = ScholarFusionGraphStore(str(self.store.db_path))
        out = []
        for edge in graph_store.query(query).get("edges", [])[: self.top_k]:
            evidence_ids = edge.get("evidence_ids") or []
            if not evidence_ids:
                continue
            element = self.store.get_element(evidence_ids[0])
            if element:
                out.append(AnyToAnyResult(
                    object_id=f"graph:{edge['edge_id']}",
                    paper_id=element.paper_id,
                    element_id=element.element_id,
                    matched_modality="graph",
                    fusion_score=float(edge.get("confidence", 0.5)),
                    raw_scores={"graph": float(edge.get("confidence", 0.5))},
                    text=element.text,
                    metadata={"page_number": element.page_number, "element_type": element.element_type},
                ))
        return out

    def image_for(self, record: Dict[str, Any]) -> Any:
        element_id = record.get("query_image_element_id")
        if not element_id:
            return None
        element = self.store.get_element(element_id)
        return element.metadata.get("image_path") if element else None


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
