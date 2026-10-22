from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from config.settings import Settings as AppSettings
from core.research_multimodal_embedding import FakeMultimodalEmbeddingProvider
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_retrieval import AnyToAnyResult, WeightedRRF


@dataclass
class Phase3Trace:
    retrievers_invoked: List[str] = field(default_factory=list)
    collections_used: List[str] = field(default_factory=list)
    latency_by_stage: Dict[str, float] = field(default_factory=dict)
    pre_fusion_rank: Dict[str, List[str]] = field(default_factory=dict)
    post_fusion_rank: List[str] = field(default_factory=list)
    post_rerank_rank: List[str] = field(default_factory=list)


class ProtectedWeightedRRF(WeightedRRF):
    """Weighted RRF with primary list protection.

    The first primary list's top candidates are guaranteed to survive final
    truncation, so adding auxiliary retrievers cannot erase a strong primary
    dense baseline from top-k.
    """

    def fuse_with_primary(
            self,
            ranked_lists: Dict[str, List[AnyToAnyResult]],
            top_k: int,
            primary_source: str,
            protect_k: Optional[int] = None,
    ) -> List[AnyToAnyResult]:
        fused = self.fuse(ranked_lists, top_k=max(top_k * 3, top_k))
        protected = ranked_lists.get(primary_source, [])[: protect_k or top_k]
        out: List[AnyToAnyResult] = []
        seen = set()
        for item in protected + fused:
            if item.object_id in seen:
                continue
            seen.add(item.object_id)
            out.append(item)
            if len(out) >= top_k:
                break
        return out


class Phase3ProductionRetriever:
    def __init__(
            self,
            store: ResearchStore,
            vector_index: ScholarFusionVectorIndex,
            chroma_dir: str = AppSettings.CHROMA_PERSIST_DIR,
            chroma_collection: str = "phase3_research_text_dense",
    ):
        self.store = store
        self.vector_index = vector_index
        self.provider = FakeMultimodalEmbeddingProvider(
            model=AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL,
            dimension=2560,
        )
        self.rrf = ProtectedWeightedRRF()
        self.chroma_client = chromadb.PersistentClient(chroma_dir)
        self.chroma_collection_name = chroma_collection
        self.chroma = self.chroma_client.get_or_create_collection(chroma_collection)
        self._bm25: Optional[BM25Retriever] = None

    def prepare(self, force: bool = False) -> Dict[str, Any]:
        if force:
            try:
                self.chroma_client.delete_collection(self.chroma_collection_name)
            except Exception:
                pass
            self.chroma = self.chroma_client.get_or_create_collection(self.chroma_collection_name)
        text_vectors = self.vector_index.list_vectors(AppSettings.SCHOLARFUSION_TEXT_COLLECTION)
        if text_vectors:
            self.chroma.upsert(
                ids=[item.object_id for item in text_vectors],
                embeddings=[item.vector for item in text_vectors],
                documents=[item.text[:2000] for item in text_vectors],
                metadatas=[_flat_metadata(item.metadata | {"paper_id": item.paper_id, "element_id": item.element_id or ""}) for item in text_vectors],
            )
        self._bm25 = self._build_bm25()
        return {
            "chroma_collection": self.chroma_collection_name,
            "chroma_count": self.chroma.count(),
            "bm25_enabled": self._bm25 is not None,
            "bm25_node_count": len(self._bm25_nodes()) if self._bm25 else 0,
        }

    def search(
            self,
            query: str,
            query_image: Any = None,
            query_type: str = "text",
            top_k: int = 10,
            enable_page_visual: Optional[bool] = None,
            enable_graph: bool = False,
    ) -> tuple[List[AnyToAnyResult], Phase3Trace]:
        trace = Phase3Trace()
        ranked: Dict[str, List[AnyToAnyResult]] = {}
        image_heavy = query_type in {"image_to_text", "image_to_image", "pair_to_any", "text_to_figure"}
        use_page = image_heavy if enable_page_visual is None else enable_page_visual

        start = time.perf_counter()
        ranked["multimodal_dense"] = self._sqlite_vector(
            AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION,
            self._query_vector(query, query_image, query_type),
            "multimodal_dense",
            top_k,
        )
        trace.latency_by_stage["multimodal_dense_ms"] = _elapsed(start)

        if query:
            start = time.perf_counter()
            ranked["chroma_dense"] = self._chroma(query, top_k)
            trace.latency_by_stage["chroma_dense_ms"] = _elapsed(start)

            start = time.perf_counter()
            ranked["bm25"] = self._bm25_results(query, top_k)
            trace.latency_by_stage["bm25_ms"] = _elapsed(start)

        if use_page:
            start = time.perf_counter()
            ranked["page_visual"] = self._sqlite_vector(
                AppSettings.SCHOLARFUSION_PAGE_COLLECTION,
                self._query_vector(query, query_image, query_type),
                "page_visual",
                top_k,
            )
            trace.latency_by_stage["page_visual_ms"] = _elapsed(start)

        start = time.perf_counter()
        fused = self.rrf.fuse_with_primary(ranked, top_k, primary_source="multimodal_dense", protect_k=min(5, top_k))
        trace.latency_by_stage["rrf_ms"] = _elapsed(start)
        trace.retrievers_invoked = [key for key, value in ranked.items() if value]
        trace.collections_used = [
            self.chroma_collection_name if key == "chroma_dense" else key
            for key in trace.retrievers_invoked
        ]
        trace.pre_fusion_rank = {key: [item.object_id for item in value[:top_k]] for key, value in ranked.items()}
        trace.post_fusion_rank = [item.object_id for item in fused]
        trace.post_rerank_rank = trace.post_fusion_rank
        return fused, trace

    def _query_vector(self, query: str, image: Any, query_type: str) -> List[float]:
        if query_type == "pair_to_any" and image:
            return self.provider.encode_text_image_pair(query, image).vector
        if query_type in {"image_to_text", "image_to_image"} and image:
            return self.provider.encode_image(image).vector
        return self.provider.encode_text(query or "research paper").vector

    def _sqlite_vector(self, collection: str, vector: List[float], source: str, top_k: int) -> List[AnyToAnyResult]:
        return [
            AnyToAnyResult(
                object_id=hit.object_id,
                paper_id=hit.paper_id,
                element_id=hit.element_id,
                matched_modality=hit.modality,
                fusion_score=hit.score,
                raw_scores={source: hit.score},
                retrieval_sources=[source],
                text=hit.text,
                metadata=hit.metadata,
            )
            for hit in self.vector_index.search(collection, vector, top_k=top_k)
        ]

    def _chroma(self, query: str, top_k: int) -> List[AnyToAnyResult]:
        vector = self.provider.encode_text(query).vector
        result = self.chroma.query(query_embeddings=[vector], n_results=top_k, include=["metadatas", "documents", "distances"])
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out = []
        for object_id, doc, meta, distance in zip(ids, docs, metas, distances):
            score = 1 / (1 + float(distance or 0.0))
            out.append(AnyToAnyResult(
                object_id=object_id,
                paper_id=str(meta.get("paper_id", "")),
                element_id=str(meta.get("element_id") or object_id),
                matched_modality="text",
                fusion_score=score,
                raw_scores={"chroma_dense": score},
                retrieval_sources=["chroma_dense"],
                text=doc or "",
                metadata=dict(meta or {}),
            ))
        return out

    def _build_bm25(self) -> Optional[BM25Retriever]:
        nodes = self._bm25_nodes()
        if not nodes:
            return None
        try:
            top_k = max(1, min(AppSettings.SCHOLARFUSION_TEXT_RECALL_TOP_K, len(nodes)))
            return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
        except Exception:
            return None

    def _bm25_nodes(self) -> List[TextNode]:
        elements = self.store.list_elements(element_types=["title", "abstract", "paragraph", "figure_caption", "table", "table_caption"])
        return [
            TextNode(
                id_=element.element_id,
                text=element.text or "",
                metadata={
                    "paper_id": element.paper_id,
                    "element_id": element.element_id,
                    "page_number": element.page_number,
                    "element_type": element.element_type,
                },
            )
            for element in elements if (element.text or "").strip()
        ]

    def _bm25_results(self, query: str, top_k: int) -> List[AnyToAnyResult]:
        if not self._bm25:
            self._bm25 = self._build_bm25()
        if not self._bm25:
            return []
        hits = self._bm25.retrieve(query)[:top_k]
        out = []
        for hit in hits:
            meta = dict(hit.node.metadata or {})
            score = float(hit.score or 0.0)
            out.append(AnyToAnyResult(
                object_id=hit.node.node_id,
                paper_id=str(meta.get("paper_id", "")),
                element_id=str(meta.get("element_id") or hit.node.node_id),
                matched_modality="text",
                fusion_score=score,
                raw_scores={"bm25": score},
                retrieval_sources=["bm25"],
                text=hit.node.text or "",
                metadata=meta,
            ))
        return out


def _flat_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
