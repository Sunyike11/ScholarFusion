from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from core.research_retrieval import STOPWORDS
from core.scholarfusion_retrieval import AnyToAnyResult


@dataclass
class RerankTrace:
    candidate_id: str
    pre_rerank_rank: int
    post_rerank_rank: int
    rerank_score: float
    reason_code: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FakeMultimodalReranker:
    def rerank(self, query: str, candidates: List[AnyToAnyResult], top_n: int = 5) -> List[AnyToAnyResult]:
        tokens = _tokens(query)
        scored = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_tokens = _tokens(candidate.text or str(candidate.metadata))
            overlap = len(tokens & candidate_tokens)
            modality_bonus = 0.05 if candidate.matched_modality in {"figure", "pair", "page"} else 0.0
            score = candidate.fusion_score + overlap * 0.1 + modality_bonus
            candidate.metadata["rerank_trace"] = RerankTrace(
                candidate_id=candidate.object_id,
                pre_rerank_rank=index,
                post_rerank_rank=0,
                rerank_score=round(score, 4),
                reason_code="token_overlap_modality_bonus",
            ).to_dict()
            candidate.fusion_score = score
            scored.append(candidate)
        scored.sort(key=lambda item: item.fusion_score, reverse=True)
        for index, candidate in enumerate(scored, start=1):
            candidate.metadata["rerank_trace"]["post_rerank_rank"] = index
        return scored[:top_n]


class QwenVLListwiseReranker(FakeMultimodalReranker):
    """Placeholder-compatible listwise reranker boundary.

    This class intentionally reuses deterministic scoring unless a real VL scorer is
    injected in a future phase. It must not be advertised as dedicated Qwen3-VL-Reranker.
    """

    provider_name = "qwen_vl_listwise_fallback"


def _tokens(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z0-9_]+", (text or "").lower()):
        if word in STOPWORDS or len(word) <= 1:
            continue
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]
        words.add(word)
    return words
