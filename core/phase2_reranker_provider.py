from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import Settings as AppSettings
from core.scholarfusion_retrieval import AnyToAnyResult
from core.scholarfusion_reranker import FakeMultimodalReranker


class RerankerError(RuntimeError):
    pass


@dataclass
class RerankerSelection:
    provider_name: str
    real_provider: bool
    reason: str


class MultimodalRerankerProvider:
    provider_name = "base"

    def rerank(self, query: str, candidates: List[AnyToAnyResult], top_n: int) -> List[AnyToAnyResult]:
        raise NotImplementedError


class DeterministicFallbackReranker(MultimodalRerankerProvider):
    provider_name = "deterministic_fallback"

    def __init__(self):
        self.impl = FakeMultimodalReranker()

    def rerank(self, query: str, candidates: List[AnyToAnyResult], top_n: int) -> List[AnyToAnyResult]:
        results = self.impl.rerank(query, candidates, top_n)
        for item in results:
            item.metadata["reranker_provider"] = self.provider_name
        return results


class Qwen3VLRerankerProvider(MultimodalRerankerProvider):
    provider_name = "qwen3-vl-rerank"

    def __init__(
            self,
            api_key: Optional[str] = None,
            endpoint: str = AppSettings.DASHSCOPE_MULTIMODAL_RERANK_ENDPOINT,
            model: str = AppSettings.DASHSCOPE_MULTIMODAL_RERANK_MODEL,
            timeout: float = AppSettings.DASHSCOPE_REQUEST_TIMEOUT,
    ):
        self.api_key = AppSettings.API_KEY if api_key is None else api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        if not self.api_key:
            raise RerankerError("DASHSCOPE_API_KEY is missing")
        if not self.endpoint:
            raise RerankerError("DASHSCOPE_MULTIMODAL_RERANK_ENDPOINT is missing")

    def rerank(self, query: str, candidates: List[AnyToAnyResult], top_n: int) -> List[AnyToAnyResult]:
        documents = []
        for item in candidates:
            doc: Dict[str, Any] = {"text": item.text or str(item.metadata)}
            image_path = item.metadata.get("image_path") or item.metadata.get("page_image_path")
            if image_path and not str(image_path).startswith("page_proxy:"):
                doc["image"] = str(image_path)
            documents.append(doc)
        payload = {
            "model": self.model,
            "input": {
                "query": {"text": query},
                "documents": documents,
            },
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        raw = self._request(payload)
        scores = self._extract_scores(raw, len(candidates))
        for index, (item, score) in enumerate(zip(candidates, scores), start=1):
            item.fusion_score = score
            item.metadata["reranker_provider"] = self.provider_name
            item.metadata["rerank_trace"] = {
                "candidate_id": item.object_id,
                "pre_rerank_rank": index,
                "rerank_score": score,
                "reason_code": "qwen3_vl_rerank_score",
            }
        ranked = sorted(candidates, key=lambda item: item.fusion_score, reverse=True)[:top_n]
        for index, item in enumerate(ranked, start=1):
            item.metadata["rerank_trace"]["post_rerank_rank"] = index
        return ranked

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise RerankerError(f"DashScope reranker HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RerankerError(f"DashScope reranker request failed: {exc.reason}") from exc

    def _extract_scores(self, data: Dict[str, Any], expected: int) -> List[float]:
        results = data.get("output", {}).get("results") or data.get("results") or []
        if not results:
            raise RerankerError("Reranker response did not contain scores")
        scores = [0.0] * expected
        sequential_scores = []
        for position, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            sequential_scores.append(score)
            index = item.get("index")
            if index is None:
                index = item.get("document_index")
            if index is not None:
                try:
                    idx = int(index)
                    if 0 <= idx < expected:
                        scores[idx] = score
                except (TypeError, ValueError):
                    pass
        if not any(scores):
            scores = sequential_scores
        if len(scores) < expected:
            scores.extend([0.0] * (expected - len(scores)))
        return scores[:expected]


def select_reranker(real_provider: bool = False) -> MultimodalRerankerProvider:
    if real_provider:
        return Qwen3VLRerankerProvider()
    return DeterministicFallbackReranker()
