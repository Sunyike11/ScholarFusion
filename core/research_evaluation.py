from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class EvaluationItem:
    query_id: str
    query_type: str
    input_modality: str
    target_modality: List[str]
    question: str
    reference_answer: str
    relevant_element_ids: List[str]
    answerable: bool
    difficulty: str
    review_status: str


def load_evaluation_items(path: str) -> List[EvaluationItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvaluationItem(**item) for item in data]


class ResearchEvaluationRunner:
    """Computes deterministic retrieval and citation metrics."""

    def evaluate_retrieval(self, items: Sequence[EvaluationItem], retrieved: Dict[str, List[str]], k_values: Sequence[int] = (5, 10)) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for k in k_values:
            recalls = []
            precisions = []
            ndcgs = []
            for item in items:
                relevant = set(item.relevant_element_ids)
                top = retrieved.get(item.query_id, [])[:k]
                if not relevant:
                    continue
                hits = [element_id for element_id in top if element_id in relevant]
                recalls.append(len(set(hits)) / len(relevant))
                precisions.append(len(hits) / max(k, 1))
                ndcgs.append(self._ndcg(top, relevant, k))
            metrics[f"recall@{k}"] = self._mean(recalls)
            metrics[f"precision@{k}"] = self._mean(precisions)
            metrics[f"ndcg@{k}"] = self._mean(ndcgs)
        metrics["mrr"] = self._mrr(items, retrieved)
        return metrics

    def evaluate_citations(self, items: Sequence[EvaluationItem], citations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        precisions = []
        recalls = []
        page_hits = []
        valid_rates = []
        for item in items:
            relevant = set(item.relevant_element_ids)
            cited = citations.get(item.query_id, [])
            cited_ids = [str(cite.get("element_id")) for cite in cited if cite.get("element_id")]
            valid_rates.append(len(cited_ids) / max(len(cited), 1) if cited else 1.0)
            if relevant:
                hit_count = len(set(cited_ids) & relevant)
                precisions.append(hit_count / max(len(cited_ids), 1))
                recalls.append(hit_count / len(relevant))
            for cite in cited:
                if cite.get("page_number") is not None:
                    page_hits.append(1.0)
                else:
                    page_hits.append(0.0)
        return {
            "citation_precision": self._mean(precisions),
            "citation_recall": self._mean(recalls),
            "page_accuracy": self._mean(page_hits),
            "element_id_validity_rate": self._mean(valid_rates),
        }

    def evaluate_answerability(self, items: Sequence[EvaluationItem], predictions: Dict[str, bool]) -> Dict[str, float]:
        if not items:
            return {"answerable_accuracy": 0.0}
        correct = sum(1 for item in items if predictions.get(item.query_id) == item.answerable)
        return {"answerable_accuracy": correct / len(items)}

    def experiment_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(config)
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            metadata["git_commit"] = "unknown"
        return metadata

    def _mrr(self, items: Sequence[EvaluationItem], retrieved: Dict[str, List[str]]) -> float:
        reciprocal_ranks = []
        for item in items:
            relevant = set(item.relevant_element_ids)
            rank = 0
            for index, element_id in enumerate(retrieved.get(item.query_id, []), start=1):
                if element_id in relevant:
                    rank = index
                    break
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        return self._mean(reciprocal_ranks)

    def _ndcg(self, ranking: List[str], relevant: set[str], k: int) -> float:
        dcg = 0.0
        for index, element_id in enumerate(ranking[:k], start=1):
            rel = 1.0 if element_id in relevant else 0.0
            dcg += rel / math.log2(index + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0

    def _mean(self, values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0
