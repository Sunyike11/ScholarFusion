from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings as AppSettings
from core.research_multimodal_embedding import FakeMultimodalEmbeddingProvider, ImageInput, MultimodalEmbeddingProvider
from core.research_retrieval import ResearchRetriever, ResearchSearchFilters
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex, VectorSearchHit


@dataclass
class AnyToAnyQuery:
    query_text: Optional[str] = None
    query_image: Optional[ImageInput] = None
    query_element_id: Optional[str] = None
    target_modalities: Optional[List[str]] = None
    filters: Optional[ResearchSearchFilters] = None
    top_k: int = 10


@dataclass
class AnyToAnyResult:
    object_id: str
    paper_id: str
    element_id: Optional[str]
    matched_modality: str
    fusion_score: float
    retrieval_sources: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    rank_positions: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryRouter:
    def route(self, query: AnyToAnyQuery) -> Dict[str, Any]:
        if query.query_text and query.query_image is not None:
            query_type = "text_image_pair"
        elif query.query_image is not None:
            query_type = "image"
        elif query.query_element_id:
            query_type = "stored_element"
        else:
            query_type = "text"
        text = (query.query_text or "").lower()
        intent = "general"
        if any(word in text for word in ["figure", "fig", "architecture", "framework", "diagram"]):
            intent = "figure"
        elif any(word in text for word in ["dataset", "psnr", "ssim", "lpips", "metric"]):
            intent = "metric"
        elif any(word in text for word in ["compare", "difference", "versus", "vs"]):
            intent = "comparison"
        return {"query_type": query_type, "intent": intent, "target_modalities": query.target_modalities or ["text", "figure", "page", "pair"]}


class WeightedRRF:
    def __init__(self, k: int = AppSettings.SCHOLARFUSION_RRF_K, weights: Optional[Dict[str, float]] = None):
        self.k = k
        self.weights = weights or AppSettings.SCHOLARFUSION_WEIGHTS

    def fuse(self, ranked_lists: Dict[str, List[AnyToAnyResult]], top_k: int) -> List[AnyToAnyResult]:
        merged: Dict[str, AnyToAnyResult] = {}
        for source, results in ranked_lists.items():
            weight = float(self.weights.get(source, 1.0))
            for rank, result in enumerate(results, start=1):
                score = weight / (self.k + rank)
                existing = merged.get(result.object_id)
                if not existing:
                    result = AnyToAnyResult(**result.to_dict())
                    result.fusion_score = 0.0
                    merged[result.object_id] = result
                    existing = result
                existing.fusion_score += score
                existing.retrieval_sources.append(source)
                existing.raw_scores[source] = result.raw_scores.get(source, result.fusion_score)
                existing.rank_positions[source] = rank
        fused = list(merged.values())
        fused.sort(key=lambda item: item.fusion_score, reverse=True)
        return fused[:top_k]


class ScholarFusionRetriever:
    def __init__(
            self,
            store: ResearchStore,
            index: ScholarFusionVectorIndex,
            embedding_provider: Optional[MultimodalEmbeddingProvider] = None,
    ):
        self.store = store
        self.index = index
        self.embedding_provider = embedding_provider or FakeMultimodalEmbeddingProvider(
            model=AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL,
            dimension=2560,
        )
        self.lexical = ResearchRetriever(store)
        self.router = QueryRouter()
        self.rrf = WeightedRRF()

    def search_any_to_any(self, query: AnyToAnyQuery) -> List[AnyToAnyResult]:
        route = self.router.route(query)
        target_modalities = route["target_modalities"]
        filters = query.filters or ResearchSearchFilters()
        ranked: Dict[str, List[AnyToAnyResult]] = {}

        if route["query_type"] in {"text", "text_image_pair", "stored_element"} and query.query_text:
            ranked["bm25"] = self._lexical_results(query.query_text, filters, AppSettings.SCHOLARFUSION_TEXT_RECALL_TOP_K)
            ranked["text_dense"] = self._vector_results(
                AppSettings.SCHOLARFUSION_TEXT_COLLECTION,
                self.embedding_provider.encode_text(query.query_text).vector,
                target_modalities,
                filters,
                AppSettings.SCHOLARFUSION_TEXT_RECALL_TOP_K,
                "text_dense",
            )
        if route["query_type"] == "image" and query.query_image is not None:
            query_vector = self.embedding_provider.encode_image(query.query_image).vector
        elif route["query_type"] == "text_image_pair" and query.query_text and query.query_image is not None:
            query_vector = self.embedding_provider.encode_text_image_pair(query.query_text, query.query_image).vector
        elif query.query_text:
            query_vector = self.embedding_provider.encode_text(query.query_text).vector
        else:
            query_vector = None

        if query_vector:
            ranked["multimodal_dense"] = self._vector_results(
                AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION,
                query_vector,
                target_modalities,
                filters,
                AppSettings.SCHOLARFUSION_MULTIMODAL_RECALL_TOP_K,
                "multimodal_dense",
            )
            ranked["page_visual"] = self._vector_results(
                AppSettings.SCHOLARFUSION_PAGE_COLLECTION,
                query_vector,
                ["page"],
                filters,
                AppSettings.SCHOLARFUSION_PAGE_RECALL_TOP_K,
                "page_visual",
            )
            ranked["paper_level"] = self._vector_results(
                AppSettings.SCHOLARFUSION_PAPER_COLLECTION,
                query_vector,
                ["paper"],
                filters,
                AppSettings.SCHOLARFUSION_PAGE_RECALL_TOP_K,
                "paper_level",
            )
        return self.rrf.fuse(ranked, top_k=query.top_k or AppSettings.SCHOLARFUSION_FINAL_TOP_K)

    def _lexical_results(self, text: str, filters: ResearchSearchFilters, top_k: int) -> List[AnyToAnyResult]:
        hits = self.lexical.search_text(text, filters=filters, top_k=top_k)
        results = []
        for hit in hits:
            results.append(AnyToAnyResult(
                object_id=hit.element_id,
                paper_id=hit.paper_id,
                element_id=hit.element_id,
                matched_modality="text",
                fusion_score=hit.rerank_score,
                raw_scores={"bm25": hit.rerank_score},
                metadata=hit.metadata | {"page_number": hit.page_number, "section_path": hit.section_path, "element_type": hit.element_type},
                text=hit.text,
            ))
        return results

    def _vector_results(
            self,
            collection: str,
            query_vector: List[float],
            target_modalities: List[str],
            filters: ResearchSearchFilters,
            top_k: int,
            source: str,
    ) -> List[AnyToAnyResult]:
        hits = self.index.search(
            collection,
            query_vector,
            top_k=top_k,
            target_modalities=target_modalities if target_modalities else None,
            paper_ids=filters.paper_ids if filters else None,
        )
        return [
            AnyToAnyResult(
                object_id=hit.object_id,
                paper_id=hit.paper_id,
                element_id=hit.element_id,
                matched_modality=hit.modality,
                fusion_score=hit.score,
                raw_scores={source: hit.score},
                metadata=hit.metadata,
                text=hit.text,
            )
            for hit in hits
        ]
