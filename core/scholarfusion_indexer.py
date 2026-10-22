from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings as AppSettings
from core.research_models import PaperElement
from core.research_multimodal_embedding import FakeMultimodalEmbeddingProvider, MultimodalEmbeddingProvider
from core.research_store import ResearchStore
from core.page_visual import PageVisualRenderer
from core.scholarfusion_figures import unique_non_decorative_figures
from core.scholarfusion_index import IndexedVector, ScholarFusionVectorIndex


class ScholarFusionIndexer:
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

    def index_paper(self, paper_id: str, force: bool = False, max_requests: int = AppSettings.SCHOLARFUSION_MAX_REQUESTS) -> Dict[str, Any]:
        paper = self.store.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        if force:
            self.index.delete_paper(paper_id)
        request_count = 0
        indexed = {"text": 0, "figure": 0, "pair": 0, "page": 0, "paper": 0, "failures": []}
        text_quota = max(5, max_requests // 3)
        figure_pair_quota = max(4, max_requests // 3)
        page_quota = max(2, max_requests // 5)

        for element in paper.elements:
            if request_count >= max_requests or indexed["text"] >= text_quota:
                break
            if element.element_type in {"title", "abstract", "paragraph", "figure_caption", "table", "table_caption"} and element.text.strip():
                result = self.embedding_provider.encode_text(element.text[:1800])
                request_count += 1
                self.index.upsert(IndexedVector(
                    collection=AppSettings.SCHOLARFUSION_TEXT_COLLECTION,
                    object_id=element.element_id,
                    paper_id=paper_id,
                    element_id=element.element_id,
                    modality="text",
                    vector=result.vector,
                    text=element.text,
                    metadata=self._metadata(element, result, "text", False),
                ))
                indexed["text"] += 1

        for figure in unique_non_decorative_figures(paper.elements):
            if request_count >= max_requests or indexed["figure"] + indexed["pair"] >= figure_pair_quota:
                break
            image_ref = figure.metadata.get("image_path") or figure.text
            try:
                result = self.embedding_provider.encode_image(image_ref)
                request_count += 1
                self.index.upsert(IndexedVector(
                    collection=AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION,
                    object_id=figure.element_id,
                    paper_id=paper_id,
                    element_id=figure.element_id,
                    modality="figure",
                    vector=result.vector,
                    text=figure.metadata.get("caption") or figure.text,
                    metadata=self._metadata(figure, result, "image", False),
                ))
                indexed["figure"] += 1
            except Exception as exc:
                indexed["failures"].append({"element_id": figure.element_id, "stage": "figure", "error": str(exc)[:160]})

            if request_count >= max_requests or indexed["figure"] + indexed["pair"] >= figure_pair_quota:
                break
            pair_text = self._pair_text(figure)
            if pair_text:
                try:
                    result = self.embedding_provider.encode_text_image_pair(pair_text[:1600], image_ref)
                    request_count += 1
                    pair_id = f"pair:{figure.element_id}"
                    self.index.upsert(IndexedVector(
                        collection=AppSettings.SCHOLARFUSION_MULTIMODAL_COLLECTION,
                        object_id=pair_id,
                        paper_id=paper_id,
                        element_id=figure.element_id,
                        modality="pair",
                        vector=result.vector,
                        text=pair_text,
                        metadata=self._metadata(figure, result, "text_image_pair", True) | {"pair_id": pair_id},
                    ))
                    indexed["pair"] += 1
                except Exception as exc:
                    indexed["failures"].append({"element_id": figure.element_id, "stage": "pair", "error": str(exc)[:160]})

        page_visuals = self._page_visuals(paper_id)
        for page_info in page_visuals:
            if request_count >= max_requests or indexed["page"] >= page_quota:
                break
            page_number = page_info["page_number"]
            result = self.embedding_provider.encode_image(page_info["page_image_path"])
            request_count += 1
            self.index.upsert(IndexedVector(
                collection=AppSettings.SCHOLARFUSION_PAGE_COLLECTION,
                object_id=f"page:{paper_id}:{page_number}",
                paper_id=paper_id,
                element_id=None,
                modality="page",
                vector=result.vector,
                text=f"Rendered page image {page_number}",
                metadata={
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "page_image_path": page_info["page_image_path"],
                    "page_hash": page_info["page_hash"],
                    "input_modality": "page_image",
                    "provider": result.model,
                    "model": result.model,
                    "dimension": result.dimension,
                    "fusion_enabled": False,
                    "embedding_version": "scholarfusion-page-visual-v2",
                    "source_hash": page_info["page_hash"],
                },
            ))
            indexed["page"] += 1

        paper_text = self._paper_text(paper_id)
        if paper_text and request_count < max_requests:
            result = self.embedding_provider.encode_text(paper_text[:2500])
            request_count += 1
            self.index.upsert(IndexedVector(
                collection=AppSettings.SCHOLARFUSION_PAPER_COLLECTION,
                object_id=f"paper:{paper_id}",
                paper_id=paper_id,
                element_id=None,
                modality="paper",
                vector=result.vector,
                text=paper_text,
                metadata={
                    "paper_id": paper_id,
                    "provider": result.model,
                    "model": result.model,
                    "dimension": result.dimension,
                    "input_modality": "paper_summary",
                    "fusion_enabled": False,
                    "embedding_version": "scholarfusion-paper-v1",
                    "source_hash": _hash(paper_text),
                },
            ))
            indexed["paper"] += 1
        indexed["request_count"] = request_count
        return indexed

    def _metadata(self, element: PaperElement, result: Any, input_modality: str, fusion_enabled: bool) -> Dict[str, Any]:
        return {
            "paper_id": element.paper_id,
            "element_id": element.element_id,
            "page_number": element.page_number,
            "section_path": element.section_path,
            "element_type": element.element_type,
            "modality": "figure" if element.element_type == "figure" else "text",
            "provider": result.model,
            "model": result.model,
            "dimension": result.dimension,
            "input_modality": input_modality,
            "fusion_enabled": fusion_enabled,
            "embedding_version": "scholarfusion-qwen3-vl-v1",
            "source_hash": _hash((element.text or "") + str(element.metadata.get("image_hash") or "")),
            "figure_id": element.element_id if element.element_type == "figure" else None,
            "bbox": element.bbox,
            "figure_type": element.metadata.get("figure_type"),
        }

    def _pair_text(self, figure: PaperElement) -> str:
        parts = [str(figure.metadata.get("caption") or figure.text or "")]
        link_evidence = figure.metadata.get("link_evidence") or []
        high_confidence_ids = [
            link.get("element_id")
            for link in link_evidence
            if link.get("element_id")
            and link.get("link_type") in {"explicit_reference", "caption_neighbor"}
            and float(link.get("confidence", 0.0)) >= 0.8
        ]
        fallback_ids = [
            link.get("element_id")
            for link in link_evidence
            if link.get("element_id")
            and link.get("link_type") in {"same_page_neighbor", "same_section_fallback"}
        ]
        linked_ids = high_confidence_ids or fallback_ids or figure.linked_element_ids
        for linked_id in linked_ids[:4]:
            linked = self.store.get_element(linked_id)
            if linked and linked.text:
                parts.append(linked.text)
        return "\n".join(part for part in parts if part.strip())

    def _paper_text(self, paper_id: str) -> str:
        paper = self.store.get_paper(paper_id)
        if not paper:
            return ""
        card = self.store.get_card(paper_id, "manual") or self.store.get_card(paper_id, "auto")
        parts = [paper.metadata.title]
        abstracts = [element.text for element in paper.elements if element.element_type == "abstract"][:2]
        parts.extend(abstracts)
        if card:
            for field_name in ["task", "representation", "body_model", "datasets", "metrics", "main_contributions"]:
                field = card.fields.get(field_name)
                if field and field.value != "unknown":
                    parts.append(f"{field_name}: {field.value}")
        return "\n".join(parts)

    def _page_visuals(self, paper_id: str) -> List[Dict[str, Any]]:
        paper = self.store.get_paper(paper_id, include_elements=False)
        if not paper:
            return []
        source = paper.metadata.source_file
        candidates = []
        if source:
            candidates = [
                Path(AppSettings.RESEARCH_REAL_PAPER_DIR) / source,
                Path(source),
            ]
        pdf_path = next((path for path in candidates if path.exists()), None)
        if not pdf_path:
            # Test fallback: keep page collection usable without a source PDF, clearly marked.
            pages = sorted({element.page_number for element in self.store.list_elements(paper_id=paper_id) if element.page_number is not None})
            return [
                {"paper_id": paper_id, "page_number": page, "page_image_path": f"page_proxy:{paper_id}:{page}", "page_hash": _hash(f"{paper_id}:{page}")}
                for page in pages
            ]
        return PageVisualRenderer(AppSettings.RESEARCH_ASSET_DIR).render_pdf_pages(str(pdf_path), paper_id, max_pages=None)


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()
