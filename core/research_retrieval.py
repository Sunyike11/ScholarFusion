from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from core.research_models import PaperCard, PaperElement, SimilarPaperResult, SimilarityDimension
from core.research_store import ResearchStore


BODY_ELEMENT_TYPES = ["abstract", "paragraph"]
TEXT_ELEMENT_TYPES = ["title", "abstract", "heading", "paragraph", "figure_caption", "table", "table_caption"]
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "how",
    "does", "do", "did", "of", "to", "in", "on", "for", "and", "or", "with",
    "exact", "current", "paper", "method",
}


@dataclass
class ResearchSearchFilters:
    paper_ids: Optional[List[str]] = None
    years: Optional[List[int]] = None
    section_contains: Optional[str] = None
    element_types: Optional[List[str]] = None
    body_only: bool = False


@dataclass
class ResearchSearchResult:
    paper_id: str
    element_id: str
    element_type: str
    page_number: Optional[int]
    section_path: List[str]
    text: str
    raw_score: float
    fusion_score: float
    rerank_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureSearchResult:
    figure_id: str
    paper_id: str
    page_number: Optional[int]
    section_path: List[str]
    caption: str
    image_url: str
    hit_reason: str
    caption_score: float
    linked_paragraph_score: float
    score: float
    linked_element_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchRetriever:
    """Structured-element retrieval baseline for the research-paper MVP."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def search_text(
            self,
            query: str,
            filters: Optional[ResearchSearchFilters] = None,
            top_k: int = 10,
    ) -> List[ResearchSearchResult]:
        filters = filters or ResearchSearchFilters()
        element_types = BODY_ELEMENT_TYPES if filters.body_only else (filters.element_types or TEXT_ELEMENT_TYPES)
        candidates = self._filtered_elements(filters, element_types)
        query_tokens = self._tokens(query)
        scored: List[ResearchSearchResult] = []
        for element in candidates:
            score = self._score(query_tokens, element.text)
            if score <= 0:
                continue
            scored.append(ResearchSearchResult(
                paper_id=element.paper_id,
                element_id=element.element_id,
                element_type=element.element_type,
                page_number=element.page_number,
                section_path=element.section_path,
                text=element.text,
                raw_score=score,
                fusion_score=score,
                rerank_score=score * (1.0 + min(element.parse_confidence, 1.0) * 0.05),
                metadata=element.metadata,
            ))
        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        return scored[:top_k]

    def search_figures(
            self,
            query: str,
            filters: Optional[ResearchSearchFilters] = None,
            top_k: int = 10,
    ) -> List[FigureSearchResult]:
        filters = filters or ResearchSearchFilters()
        figures = self._filtered_elements(filters, ["figure"])
        query_tokens = self._tokens(query)
        results: List[FigureSearchResult] = []
        for figure in figures:
            caption = str(figure.metadata.get("caption") or figure.text or "")
            caption_score = self._score(query_tokens, caption)
            linked_text_parts = []
            for linked_id in figure.linked_element_ids:
                linked = self.store.get_element(linked_id)
                if linked and linked.element_type in {"paragraph", "abstract", "figure_caption"}:
                    linked_text_parts.append(linked.text)
            linked_score = self._score(query_tokens, "\n".join(linked_text_parts))
            score = caption_score * 0.65 + linked_score * 0.35
            if score <= 0:
                continue
            if caption_score > 0 and linked_score > 0:
                reason = "both"
            elif caption_score > 0:
                reason = "caption match"
            else:
                reason = "linked paragraph match"
            results.append(FigureSearchResult(
                figure_id=figure.element_id,
                paper_id=figure.paper_id,
                page_number=figure.page_number,
                section_path=figure.section_path,
                caption=caption,
                image_url=str(figure.metadata.get("image_url") or ""),
                hit_reason=reason,
                caption_score=caption_score,
                linked_paragraph_score=linked_score,
                score=score,
                linked_element_ids=figure.linked_element_ids,
            ))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def similar_papers(
            self,
            paper_id: str,
            top_k: int = 5,
            weights: Optional[Dict[str, float]] = None,
    ) -> List[SimilarPaperResult]:
        base_card = self.store.get_card(paper_id, "manual") or self.store.get_card(paper_id, "auto")
        if not base_card:
            return []
        weights = weights or {}
        results: List[SimilarPaperResult] = []
        for paper in self.store.list_papers():
            if paper.paper_id == paper_id:
                continue
            card = self.store.get_card(paper.paper_id, "manual") or self.store.get_card(paper.paper_id, "auto")
            if not card:
                continue
            matched: List[SimilarityDimension] = []
            different: List[SimilarityDimension] = []
            weighted_total = 0.0
            weight_sum = 0.0
            for dimension in [
                "task",
                "representation",
                "body_model",
                "initialization",
                "non_rigid_deformation",
                "gaussian_binding",
                "density_control",
                "datasets",
            ]:
                score = self._field_similarity(base_card, card, dimension)
                weight = float(weights.get(dimension, 1.0))
                weighted_total += score * weight
                weight_sum += weight
                reason = self._dimension_reason(base_card, card, dimension, score)
                item = SimilarityDimension(dimension=dimension, score=score, reason=reason)
                if score >= 0.5:
                    matched.append(item)
                else:
                    different.append(item)
            overall = weighted_total / weight_sum if weight_sum else 0.0
            results.append(SimilarPaperResult(
                paper_id=paper.paper_id,
                overall_score=round(overall, 4),
                matched_dimensions=matched,
                different_dimensions=different,
            ))
        results.sort(key=lambda item: item.overall_score, reverse=True)
        return results[:top_k]

    def _filtered_elements(self, filters: ResearchSearchFilters, element_types: List[str]) -> List[PaperElement]:
        all_elements: List[PaperElement] = []
        paper_ids = filters.paper_ids or [paper.paper_id for paper in self.store.list_papers()]
        years = set(filters.years or [])
        for paper_id in paper_ids:
            paper = self.store.get_paper(paper_id, include_elements=False)
            if not paper:
                continue
            if years and paper.metadata.year not in years:
                continue
            all_elements.extend(self.store.list_elements(
                paper_id=paper_id,
                element_types=element_types,
                section_contains=filters.section_contains,
            ))
        return all_elements

    def _tokens(self, text: str) -> Set[str]:
        text = (text or "").lower()
        words = set()
        for word in re.findall(r"[a-z0-9_]+", text):
            if word in STOPWORDS or len(word) <= 1:
                continue
            words.add(self._stem(word))
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
        return words | bigrams

    def _stem(self, word: str) -> str:
        if word.endswith("ization"):
            return word[:-7]
        if word.endswith("ation"):
            return word[:-5]
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("s") and len(word) > 3:
            return word[:-1]
        return word

    def _score(self, query_tokens: Set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = self._tokens(text)
        if not doc_tokens:
            return 0.0
        overlap = query_tokens & doc_tokens
        precision = len(overlap) / len(doc_tokens)
        recall = len(overlap) / len(query_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _field_similarity(self, left: PaperCard, right: PaperCard, dimension: str) -> float:
        left_value = str(left.fields.get(dimension).value if left.fields.get(dimension) else "unknown")
        right_value = str(right.fields.get(dimension).value if right.fields.get(dimension) else "unknown")
        if left_value == "unknown" or right_value == "unknown":
            return 0.0
        left_tokens = self._tokens(left_value)
        right_tokens = self._tokens(right_value)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _dimension_reason(self, left: PaperCard, right: PaperCard, dimension: str, score: float) -> str:
        left_value = left.fields.get(dimension).value if left.fields.get(dimension) else "unknown"
        right_value = right.fields.get(dimension).value if right.fields.get(dimension) else "unknown"
        if score >= 0.5:
            return f"both mention similar {dimension}: {left_value}"
        if left_value == "unknown" or right_value == "unknown":
            return f"{dimension} is missing in at least one paper"
        return f"different {dimension}: {left_value} vs {right_value}"
