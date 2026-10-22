from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import Settings as AppSettings
from core.research_retrieval import ResearchSearchFilters
from core.research_store import ResearchStore
from core.scholarfusion_graph import ScholarFusionGraphStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_retrieval import AnyToAnyQuery, AnyToAnyResult, ScholarFusionRetriever, WeightedRRF


@dataclass
class HybridRetrievalResult:
    result_id: str
    paper_id: str
    element_id: Optional[str]
    page_number: Optional[int]
    element_type: str
    modality: str
    raw_scores: Dict[str, float] = field(default_factory=dict)
    rank_positions: Dict[str, int] = field(default_factory=dict)
    retrieval_sources: List[str] = field(default_factory=list)
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegacyProductionRetrieverAdapter:
    """Adapter for legacy Chroma/BM25/Reranker path.

    It uses the existing LlamaIndex workflow only when an initialized workflow is
    supplied. Tests and cold research runs fall back to empty results rather than
    instantiating Redis/Chroma implicitly.
    """

    def __init__(self, workflow: Any = None):
        self.workflow = workflow

    def available(self) -> bool:
        return bool(self.workflow and getattr(self.workflow, "retriever", None))

    async def aretrieve(self, query: str) -> List[HybridRetrievalResult]:
        if not self.available():
            return []
        nodes = await self.workflow.retriever.aretrieve(query)
        results = []
        for rank, node in enumerate(nodes, start=1):
            metadata = dict(getattr(node.node, "metadata", {}) or {})
            results.append(HybridRetrievalResult(
                result_id=getattr(node.node, "node_id", f"legacy:{rank}"),
                paper_id=str(metadata.get("paper_id") or metadata.get("doc_id") or "legacy"),
                element_id=str(metadata.get("element_id") or getattr(node.node, "node_id", "")),
                page_number=metadata.get("page_number") or metadata.get("page"),
                element_type=str(metadata.get("element_type") or "chunk"),
                modality="text",
                raw_scores={"legacy_chroma_bm25": float(node.score or 0.0)},
                rank_positions={"legacy_chroma_bm25": rank},
                retrieval_sources=["legacy_chroma_bm25"],
                fusion_score=float(node.score or 0.0),
                rerank_score=float(node.score or 0.0),
                metadata=metadata,
                text=getattr(node.node, "text", ""),
            ))
        return results


class HybridResearchRetriever:
    def __init__(
            self,
            store: ResearchStore,
            index: ScholarFusionVectorIndex,
            graph_store: Optional[ScholarFusionGraphStore] = None,
            legacy_adapter: Optional[LegacyProductionRetrieverAdapter] = None,
    ):
        self.store = store
        self.index = index
        self.graph_store = graph_store
        self.legacy_adapter = legacy_adapter or LegacyProductionRetrieverAdapter()
        self.sf_retriever = ScholarFusionRetriever(store, index)
        self.rrf = WeightedRRF()

    async def asearch(
            self,
            query_text: Optional[str] = None,
            query_image: Any = None,
            target_modalities: Optional[List[str]] = None,
            filters: Optional[ResearchSearchFilters] = None,
            top_k: int = 10,
            include_graph: bool = False,
    ) -> List[HybridRetrievalResult]:
        ranked: Dict[str, List[AnyToAnyResult]] = {}
        if query_text:
            legacy = await self.legacy_adapter.aretrieve(query_text)
            if legacy:
                ranked["legacy_chroma_bm25"] = [self._from_hybrid(item) for item in legacy]
        sf_results = self.sf_retriever.search_any_to_any(AnyToAnyQuery(
            query_text=query_text,
            query_image=query_image,
            target_modalities=target_modalities,
            filters=filters,
            top_k=max(top_k, AppSettings.SCHOLARFUSION_FUSION_TOP_K),
        ))
        ranked["scholarfusion_multimodal"] = sf_results
        if include_graph and query_text and self.graph_store:
            graph_results = self._graph_results(query_text)
            if graph_results:
                ranked["graph"] = graph_results
        fused = self.rrf.fuse(ranked, top_k)
        return [self._to_hybrid(item) for item in fused]

    def search(self, **kwargs: Any) -> List[HybridRetrievalResult]:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asearch(**kwargs))
        # FastAPI multipart endpoints are async, but the ScholarFusion SQLite
        # index path is synchronous. In an active loop we skip the optional
        # async legacy adapter and still keep all production research requests
        # behind the same HybridResearchRetriever entry point.
        return self._search_sync_without_legacy(**kwargs)

    def _search_sync_without_legacy(
            self,
            query_text: Optional[str] = None,
            query_image: Any = None,
            target_modalities: Optional[List[str]] = None,
            filters: Optional[ResearchSearchFilters] = None,
            top_k: int = 10,
            include_graph: bool = False,
    ) -> List[HybridRetrievalResult]:
        ranked: Dict[str, List[AnyToAnyResult]] = {}
        sf_results = self.sf_retriever.search_any_to_any(AnyToAnyQuery(
            query_text=query_text,
            query_image=query_image,
            target_modalities=target_modalities,
            filters=filters,
            top_k=max(top_k, AppSettings.SCHOLARFUSION_FUSION_TOP_K),
        ))
        ranked["scholarfusion_multimodal"] = sf_results
        if include_graph and query_text and self.graph_store:
            graph_results = self._graph_results(query_text)
            if graph_results:
                ranked["graph"] = graph_results
        fused = self.rrf.fuse(ranked, top_k)
        return [self._to_hybrid(item) for item in fused]

    def _to_hybrid(self, item: AnyToAnyResult) -> HybridRetrievalResult:
        metadata = item.metadata or {}
        return HybridRetrievalResult(
            result_id=item.object_id,
            paper_id=item.paper_id,
            element_id=item.element_id,
            page_number=metadata.get("page_number"),
            element_type=str(metadata.get("element_type") or item.matched_modality),
            modality=item.matched_modality,
            raw_scores=item.raw_scores,
            rank_positions=item.rank_positions,
            retrieval_sources=item.retrieval_sources,
            fusion_score=item.fusion_score,
            rerank_score=item.fusion_score,
            metadata=metadata,
            text=item.text,
        )

    def _from_hybrid(self, item: HybridRetrievalResult) -> AnyToAnyResult:
        return AnyToAnyResult(
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

    def _graph_results(self, query: str) -> List[AnyToAnyResult]:
        graph = self.graph_store.query(query) if self.graph_store else {"edges": []}
        results = []
        for edge in graph.get("edges", [])[:5]:
            evidence_ids = edge.get("evidence_ids") or []
            if not evidence_ids:
                continue
            element = self.store.get_element(evidence_ids[0])
            if not element:
                continue
            results.append(AnyToAnyResult(
                object_id=f"graph:{edge['edge_id']}",
                paper_id=element.paper_id,
                element_id=element.element_id,
                matched_modality="graph",
                fusion_score=float(edge.get("confidence", 0.5)),
                raw_scores={"graph": float(edge.get("confidence", 0.5))},
                metadata={"relation": edge.get("relation"), "page_number": element.page_number, "element_type": element.element_type},
                text=element.text,
            ))
        return results
