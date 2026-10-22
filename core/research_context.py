from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from core.research_models import PaperElement
from core.research_retrieval import ResearchSearchResult
from core.research_store import ResearchStore


@dataclass
class EvidenceBundle:
    elements: List[PaperElement]
    manifest: Dict[str, object]

    def to_context_text(self) -> str:
        parts = []
        for element in self.elements:
            section = " / ".join(element.section_path) if element.section_path else "-"
            parts.append(
                f"[{element.element_id}] paper={element.paper_id} page={element.page_number} "
                f"section={section} type={element.element_type}\n{element.text}"
            )
        return "\n\n".join(parts)


class EvidenceBundleBuilder:
    def __init__(
            self,
            store: ResearchStore,
            max_chars: int = 6000,
            max_elements_per_paper: int = 6,
            neighbor_window: int = 1,
    ):
        self.store = store
        self.max_chars = max_chars
        self.max_elements_per_paper = max_elements_per_paper
        self.neighbor_window = neighbor_window

    def build(self, hits: List[ResearchSearchResult]) -> EvidenceBundle:
        selected: List[PaperElement] = []
        seen: Set[str] = set()
        per_paper: Dict[str, int] = {}
        current_chars = 0

        for hit in hits:
            candidates = self._expand_hit(hit.element_id)
            for element in candidates:
                if element.element_id in seen:
                    continue
                if per_paper.get(element.paper_id, 0) >= self.max_elements_per_paper:
                    continue
                element_len = len(element.text or "")
                if current_chars + element_len > self.max_chars and selected:
                    continue
                seen.add(element.element_id)
                selected.append(element)
                per_paper[element.paper_id] = per_paper.get(element.paper_id, 0) + 1
                current_chars += element_len

        manifest = {
            "element_ids": [element.element_id for element in selected],
            "paper_ids": sorted({element.paper_id for element in selected}),
            "total_chars": current_chars,
            "max_chars": self.max_chars,
            "max_elements_per_paper": self.max_elements_per_paper,
        }
        return EvidenceBundle(elements=selected, manifest=manifest)

    def _expand_hit(self, element_id: str) -> List[PaperElement]:
        element = self.store.get_element(element_id)
        if not element:
            return []
        expanded = [element]
        linked_ids = list(element.linked_element_ids)
        if element.element_type == "figure":
            link_evidence = element.metadata.get("link_evidence") or []
            primary_link_ids = [
                link.get("element_id")
                for link in link_evidence
                if link.get("element_id")
                and link.get("link_type") in {"explicit_reference", "caption_neighbor"}
                and float(link.get("confidence", 0.0)) >= 0.8
            ]
            fallback_link_ids = [
                link.get("element_id")
                for link in link_evidence
                if link.get("element_id")
                and link.get("link_type") in {"same_page_neighbor", "same_section_fallback"}
            ]
            caption_id = element.metadata.get("caption_element_id")
            if caption_id:
                primary_link_ids.append(caption_id)
            linked_ids = primary_link_ids or fallback_link_ids or linked_ids
        for linked_id in linked_ids:
            linked = self.store.get_element(linked_id)
            if linked:
                expanded.append(linked)
        return expanded
