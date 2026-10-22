from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings as AppSettings
from core.research_citations import CitationValidator
from core.research_context import EvidenceBundleBuilder
from core.research_models import (
    ComparisonCell,
    ComparisonReport,
    EvidenceRef,
    PAPER_CARD_FIELDS,
    PaperCard,
    PaperCardField,
    PaperDocument,
    PaperElement,
    utc_now_iso,
)
from core.research_pdf_parser import ResearchPDFParser
from core.research_retrieval import ResearchRetriever, ResearchSearchFilters
from core.research_store import ResearchStore
from core.scholarfusion_graph import ScholarFusionGraphStore, build_graph_from_store
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_indexer import ScholarFusionIndexer
from core.scholarfusion_retrieval import AnyToAnyResult
from core.scholarfusion_verifier import AnswerabilityVerifier
from core.phase2_citations import CitationPlanner
from core.hybrid_research_retriever import HybridResearchRetriever
from core.phase2_reranker_provider import select_reranker


COMPARISON_DIMENSIONS = [
    "task",
    "representation",
    "body_model",
    "initialization",
    "canonical_space",
    "rigid_deformation",
    "non_rigid_deformation",
    "gaussian_binding",
    "density_control",
    "losses",
    "datasets",
    "metrics",
    "main_contributions",
    "limitations",
]


class ResearchService:
    """Application service for the structured research-paper MVP."""

    def __init__(self, store: Optional[ResearchStore] = None, parser: Optional[ResearchPDFParser] = None):
        self.store = store or ResearchStore(AppSettings.RESEARCH_DB_PATH)
        self.parser = parser or ResearchPDFParser(AppSettings.RESEARCH_ASSET_DIR)
        self.retriever = ResearchRetriever(self.store)
        self.citations = CitationValidator(self.store)
        self.vector_index = ScholarFusionVectorIndex(AppSettings.SCHOLARFUSION_INDEX_DB_PATH)
        self.graph_store = ScholarFusionGraphStore(str(self.store.db_path))
        self.answerability = AnswerabilityVerifier()
        self.citation_planner = CitationPlanner()
        self.reranker = select_reranker(real_provider=AppSettings.SCHOLARFUSION_ENABLE_REAL_RERANKER)
        self.hybrid_retriever = HybridResearchRetriever(self.store, self.vector_index, self.graph_store)

    def set_vector_index(self, index: ScholarFusionVectorIndex) -> None:
        self.vector_index = index
        self.hybrid_retriever = HybridResearchRetriever(self.store, self.vector_index, self.graph_store)

    def upload_pdf_bytes(self, content: bytes, filename: str) -> PaperDocument:
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(prefix="research_pdf_", suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            paper = self.parser.parse(tmp_path)
            paper.metadata.source_file = Path(filename).name
            self.store.upsert_paper(paper)
            card = self.generate_card(paper.paper_id)
            self.store.upsert_card(card, "auto")
            build_graph_from_store(self.store, self.graph_store)
            self.store.record_ingestion_run(
                run_id=hashlib.sha1(f"{paper.paper_id}{utc_now_iso()}".encode("utf-8")).hexdigest()[:16],
                paper_id=paper.paper_id,
                status="success",
                parser_status=paper.metadata.parser_status,
                metrics=paper.metadata.parse_quality,
            )
            return self.store.get_paper(paper.paper_id) or paper
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def list_papers(self) -> List[PaperDocument]:
        return self.store.list_papers()

    def get_paper(self, paper_id: str) -> Optional[PaperDocument]:
        return self.store.get_paper(paper_id)

    def list_figures(self, paper_id: str) -> List[PaperElement]:
        return self.store.list_elements(paper_id=paper_id, element_types=["figure"])

    def get_card(self, paper_id: str, prefer_manual: bool = True) -> Optional[PaperCard]:
        if prefer_manual:
            manual = self.store.get_card(paper_id, "manual")
            if manual:
                return manual
        return self.store.get_card(paper_id, "auto")

    def update_card(self, paper_id: str, fields: Dict[str, Any]) -> PaperCard:
        current = self.get_card(paper_id, prefer_manual=True) or self.generate_card(paper_id)
        updated_fields = dict(current.fields)
        for key, value in fields.items():
            if key not in PAPER_CARD_FIELDS:
                continue
            existing = updated_fields.get(key, PaperCardField())
            updated_fields[key] = PaperCardField(
                value=value,
                evidence_refs=existing.evidence_refs,
                confidence=1.0,
                source="manual",
            )
        card = PaperCard(
            paper_id=paper_id,
            fields=updated_fields,
            version=current.version,
            generated_at=current.generated_at,
            updated_at=utc_now_iso(),
            is_manual=True,
        )
        self.store.upsert_card(card, "manual")
        return card

    def generate_card(self, paper_id: str) -> PaperCard:
        paper = self.store.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        fields = {name: PaperCardField(value="unknown", confidence=0.0) for name in PAPER_CARD_FIELDS}
        title_element = self._first_element(paper, ["title"])
        if title_element:
            fields["title"] = self._field(title_element.text, title_element, 0.95)
        if paper.metadata.authors:
            ref = self._ref(title_element) if title_element else None
            fields["authors"] = PaperCardField(
                value=paper.metadata.authors,
                evidence_refs=[ref] if ref else [],
                confidence=0.6,
            )
        if paper.metadata.year:
            ref = self._ref(title_element) if title_element else None
            fields["year"] = PaperCardField(value=paper.metadata.year, evidence_refs=[ref] if ref else [], confidence=0.5)
        fields["venue"] = PaperCardField(value=paper.metadata.venue, confidence=0.0)

        relevant = self._relevant_elements_for_card(paper)
        joined = "\n".join(element.text for element in relevant)
        keyword_map = {
            "task": ["reconstruction", "human", "avatar", "3d gaussian", "3dgs", "重建"],
            "input": ["input", "monocular", "multi-view", "video", "image"],
            "output": ["output", "avatar", "mesh", "gaussian", "render"],
            "representation": ["gaussian", "3d gaussian", "mesh", "neural radiance", "sdf"],
            "body_model": ["smpl", "smpl-x", "body model"],
            "initialization": ["initialize", "initialization", "anchor", "surface"],
            "canonical_space": ["canonical"],
            "rigid_deformation": ["rigid"],
            "non_rigid_deformation": ["non-rigid", "nonrigid", "deformation"],
            "gaussian_binding": ["bind", "binding", "anchor"],
            "density_control": ["densification", "prune", "density"],
            "losses": ["loss", "regularization"],
            "datasets": ["dataset", "zju", "people-snapshot", "thuman", "h36m"],
            "metrics": ["psnr", "ssim", "lpips", "metric"],
            "main_contributions": ["contribution", "we propose", "novel"],
            "limitations": ["limitation", "fail", "future work"],
            "keywords": ["gaussian", "human", "reconstruction", "avatar", "deformation"],
        }
        for field_name, keywords in keyword_map.items():
            hit = self._find_keyword_hit(relevant, keywords)
            if hit:
                fields[field_name] = self._field(self._summarize_hit(hit, keywords), hit, 0.55)
        return PaperCard(paper_id=paper_id, fields=fields, is_manual=False)

    def search_text(self, query: str, top_k: int = 10, filters: Optional[ResearchSearchFilters] = None) -> List[Dict[str, Any]]:
        results = self.hybrid_retriever.search(
            query_text=query,
            target_modalities=["text"],
            filters=filters,
            top_k=top_k,
            include_graph=False,
        )
        return [
            {
                "paper_id": item.paper_id,
                "element_id": item.element_id or item.result_id,
                "element_type": item.element_type,
                "page_number": item.page_number,
                "section_path": item.metadata.get("section_path", []),
                "text": item.text,
                "raw_score": item.raw_scores,
                "fusion_score": item.fusion_score,
                "rerank_score": item.rerank_score,
                "metadata": item.metadata | {
                    "retrieval_sources": item.retrieval_sources,
                    "rank_positions": item.rank_positions,
                },
            }
            for item in results
        ]

    def search_figures(self, query: str, top_k: int = 10, filters: Optional[ResearchSearchFilters] = None) -> List[Dict[str, Any]]:
        results = self.hybrid_retriever.search(
            query_text=query,
            target_modalities=["figure", "pair"],
            filters=filters,
            top_k=top_k,
            include_graph=False,
        )
        return [
            {
                "figure_id": item.element_id or item.result_id,
                "paper_id": item.paper_id,
                "page_number": item.page_number,
                "section_path": item.metadata.get("section_path", []),
                "caption": item.metadata.get("caption") or item.text,
                "image_url": item.metadata.get("image_url", ""),
                "hit_reason": "both" if len(item.retrieval_sources) > 1 else ("caption match" if item.retrieval_sources else "linked paragraph match"),
                "caption_score": item.raw_scores.get("bm25", 0.0),
                "linked_paragraph_score": item.raw_scores.get("scholarfusion_multimodal", 0.0),
                "score": item.fusion_score,
                "linked_element_ids": item.metadata.get("linked_element_ids", []),
                "metadata": item.metadata | {"retrieval_sources": item.retrieval_sources},
            }
            for item in results
        ]

    def build_multimodal_index(self, paper_id: str, force: bool = False, max_requests: int = AppSettings.SCHOLARFUSION_MAX_REQUESTS) -> Dict[str, Any]:
        indexer = ScholarFusionIndexer(self.store, self.vector_index)
        return indexer.index_paper(paper_id, force=force, max_requests=max_requests)

    def search_any_to_any(
            self,
            query_text: Optional[str] = None,
            query_image: Optional[Any] = None,
            query_element_id: Optional[str] = None,
            target_modalities: Optional[List[str]] = None,
            filters: Optional[ResearchSearchFilters] = None,
            top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if query_element_id and not query_text and query_image is None:
            element = self.store.get_element(query_element_id)
            if element:
                query_text = element.text
                query_image = element.metadata.get("image_path") if element.element_type == "figure" else None
        hybrid_results = self.hybrid_retriever.search(
            query_text=query_text,
            query_image=query_image,
            target_modalities=target_modalities,
            filters=filters,
            top_k=max(top_k, AppSettings.SCHOLARFUSION_RERANK_TOP_N),
            include_graph=True,
        )
        candidates = [
            AnyToAnyResult(
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
            for item in hybrid_results
        ]
        reranked = self.reranker.rerank(query_text or "", candidates, top_n=top_k)
        return [item.to_dict() for item in reranked]

    def similar_papers(self, paper_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return [
            item.to_dict()
            for item in self.retriever.similar_papers(
                paper_id,
                top_k=top_k,
                weights=AppSettings.RESEARCH_SIMILARITY_WEIGHTS,
            )
        ]

    def compare(self, base_paper_id: str, compared_paper_ids: List[str]) -> ComparisonReport:
        compared_paper_ids = compared_paper_ids[:AppSettings.RESEARCH_COMPARE_MAX_PAPERS]
        all_ids = [base_paper_id] + compared_paper_ids
        cards = {paper_id: self.get_card(paper_id) for paper_id in all_ids}
        rows: Dict[str, List[ComparisonCell]] = {}
        for dimension in COMPARISON_DIMENSIONS:
            cells: List[ComparisonCell] = []
            for paper_id in all_ids:
                field = cards[paper_id].fields.get(dimension) if cards.get(paper_id) else None
                value = field.value if field else "unknown"
                fact_type = "missing" if value in ("unknown", "", None, []) else ("explicit" if field and field.evidence_refs else "system_summary")
                cells.append(ComparisonCell(
                    paper_id=paper_id,
                    value=value,
                    evidence_refs=field.evidence_refs if field else [],
                    fact_type=fact_type,
                ))
            rows[dimension] = cells
        report = ComparisonReport(
            base_paper_id=base_paper_id,
            compared_paper_ids=compared_paper_ids,
            rows=rows,
        )
        report.markdown_report = self._comparison_markdown(report)
        return report

    def qa(self, question: str, top_k: int = 5, filters: Optional[ResearchSearchFilters] = None) -> Dict[str, Any]:
        hybrid_hits = self.hybrid_retriever.search(
            query_text=question,
            target_modalities=["text", "figure", "pair", "page"],
            filters=filters,
            top_k=top_k,
            include_graph=True,
        )
        hits = []
        from core.research_retrieval import ResearchSearchResult
        for item in hybrid_hits:
            element = self.store.get_element(item.element_id) if item.element_id else None
            if not element:
                continue
            hits.append(ResearchSearchResult(
                paper_id=element.paper_id,
                element_id=element.element_id,
                element_type=element.element_type,
                page_number=element.page_number,
                section_path=element.section_path,
                text=element.text,
                raw_score=item.fusion_score,
                fusion_score=item.fusion_score,
                rerank_score=item.rerank_score or item.fusion_score,
                metadata=element.metadata,
            ))
        if not hits:
            return {
                "answer": "当前论文库中没有足够证据回答该问题。",
                "answerable": False,
                "citations": [],
                "sources": [],
                "context_manifest": {"element_ids": [], "paper_ids": [], "total_chars": 0},
            }
        bundle = EvidenceBundleBuilder(
            self.store,
            max_chars=AppSettings.RESEARCH_CONTEXT_MAX_CHARS,
            max_elements_per_paper=AppSettings.RESEARCH_MAX_EVIDENCE_PER_PAPER,
        ).build(hits)
        sufficiency = self.answerability.verify(question, bundle.elements)
        if sufficiency.status != "answerable":
            return {
                "answer": f"当前论文库中没有足够证据回答该问题：{sufficiency.reason_summary}",
                "answerable": False,
                "citations": [],
                "sources": [element.element_id for element in bundle.elements],
                "context_manifest": bundle.manifest | {"answerability": sufficiency.to_dict()},
            }
        cited_elements = bundle.elements[: min(3, len(bundle.elements))]
        citations = [
            self.citations.citation_for_element(element, f"c{index + 1}")
            for index, element in enumerate(cited_elements)
        ]
        valid, errors = self.citations.validate(citations)
        if not valid:
            return {
                "answer": "检索到了候选证据，但引用校验失败，已拒绝回答。",
                "answerable": False,
                "citations": [],
                "sources": [],
                "context_manifest": bundle.manifest | {"citation_errors": errors},
            }
        answer_parts = [
            f"根据论文第 {element.page_number or '-'} 页的证据，{self._short_text(element.text)} [c{index + 1}]"
            for index, element in enumerate(cited_elements)
        ]
        answer = "\n".join(answer_parts)
        claim_results = self.answerability.verify_claims(answer, cited_elements)
        if any(item.support_status == "unsupported" for item in claim_results):
            return {
                "answer": "检索到了候选证据，但关键结论未能通过 claim-level 引用校验，已拒绝回答。",
                "answerable": False,
                "citations": citations,
                "sources": [element.element_id for element in bundle.elements],
                "context_manifest": bundle.manifest | {
                    "answerability": sufficiency.to_dict(),
                    "claim_evidence": [item.to_dict() for item in claim_results],
                },
            }
        citation_plan = self.citation_planner.plan(answer, bundle.elements)
        primary_ids = set(citation_plan["primary_evidence_ids"])
        context_ids = set(citation_plan["context_evidence_ids"])
        return {
            "answer": answer,
            "answerable": True,
            "citations": citations,
            "primary_citations": [citation for citation in citations if citation["element_id"] in primary_ids],
            "context_citations": [citation for citation in citations if citation["element_id"] in context_ids or citation["element_id"] not in primary_ids],
            "sources": [element.element_id for element in bundle.elements],
            "context_manifest": bundle.manifest | {
                "answerability": sufficiency.to_dict(),
                "claim_evidence": citation_plan["claim_support"],
            },
        }

    def graph_for_paper(self, paper_id: str) -> Dict[str, Any]:
        build_graph_from_store(self.store, self.graph_store)
        return self.graph_store.graph_for_paper(paper_id)

    def graph_query(self, query: str) -> Dict[str, Any]:
        build_graph_from_store(self.store, self.graph_store)
        return self.graph_store.query(query)

    def related_work(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        graph = self.graph_query(query)
        edges = graph.get("edges", [])[:top_k]
        lines = []
        citations = []
        for edge in edges:
            evidence_ids = edge.get("evidence_ids") or []
            if evidence_ids:
                element = self.store.get_element(evidence_ids[0])
                if element:
                    citations.append(self.citations.citation_for_element(element, f"c{len(citations) + 1}"))
                    lines.append(f"- {edge['relation']} is supported by evidence on page {element.page_number} [c{len(citations)}]")
        return {
            "summary": "\n".join(lines) if lines else "当前图谱中没有足够证据生成 related work summary。",
            "citations": citations,
            "graph_path_count": len(edges),
        }

    def _first_element(self, paper: PaperDocument, types: List[str]) -> Optional[PaperElement]:
        return next((element for element in paper.elements if element.element_type in types), None)

    def _relevant_elements_for_card(self, paper: PaperDocument) -> List[PaperElement]:
        priority_words = ["abstract", "method", "approach", "experiment", "result", "conclusion"]
        relevant = [
            element for element in paper.elements
            if element.element_type in {"title", "abstract", "heading", "paragraph"}
            and (
                element.element_type in {"title", "abstract"}
                or any(word in " ".join(element.section_path).lower() for word in priority_words)
                or any(word in element.text.lower() for word in priority_words)
            )
        ]
        return relevant[:30] or paper.elements[:20]

    def _find_keyword_hit(self, elements: List[PaperElement], keywords: List[str]) -> Optional[PaperElement]:
        for element in elements:
            lowered = element.text.lower()
            if any(keyword in lowered for keyword in keywords):
                return element
        return None

    def _summarize_hit(self, element: PaperElement, keywords: List[str]) -> str:
        text = self._short_text(element.text, 220)
        matched = [keyword for keyword in keywords if keyword in element.text.lower()]
        return text if not matched else f"{text}"

    def _field(self, value: Any, element: PaperElement, confidence: float) -> PaperCardField:
        return PaperCardField(value=value, evidence_refs=[self._ref(element)], confidence=confidence)

    def _ref(self, element: Optional[PaperElement]) -> EvidenceRef:
        if element is None:
            return EvidenceRef(paper_id="", element_id="")
        return EvidenceRef(
            paper_id=element.paper_id,
            element_id=element.element_id,
            page_number=element.page_number,
            section_path=element.section_path,
            element_type=element.element_type,
            evidence_text=self._short_text(element.text, 300),
            figure_id=element.element_id if element.element_type == "figure" else None,
        )

    def _short_text(self, text: str, max_len: int = 280) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    def _comparison_markdown(self, report: ComparisonReport) -> str:
        paper_ids = [report.base_paper_id] + report.compared_paper_ids
        header = "| Dimension | " + " | ".join(paper_ids) + " |"
        sep = "|---" * (len(paper_ids) + 1) + "|"
        rows = [header, sep]
        for dimension, cells in report.rows.items():
            values = [str(cell.value).replace("\n", " ") for cell in cells]
            rows.append("| " + dimension + " | " + " | ".join(values) + " |")
        return "\n".join(rows)
