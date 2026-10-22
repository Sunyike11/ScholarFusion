from __future__ import annotations

from typing import Dict, List, Tuple

from core.research_models import EvidenceRef, PaperElement
from core.research_store import ResearchStore


class CitationValidator:
    """Validates that citations point to persisted paper elements."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def citation_for_element(self, element: PaperElement, citation_id: str) -> Dict[str, object]:
        figure_id = element.element_id if element.element_type == "figure" else None
        return {
            "citation_id": citation_id,
            "paper_id": element.paper_id,
            "element_id": element.element_id,
            "page_number": element.page_number,
            "section_path": element.section_path,
            "element_type": element.element_type,
            "evidence_text": element.text,
            "figure_id": figure_id,
        }

    def validate(self, citations: List[Dict[str, object]]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for citation in citations:
            paper_id = str(citation.get("paper_id") or "")
            element_id = str(citation.get("element_id") or "")
            if not paper_id or not element_id:
                errors.append(f"missing paper_id or element_id in citation {citation.get('citation_id')}")
                continue
            element = self.store.get_element(element_id)
            if not element:
                errors.append(f"unknown element_id: {element_id}")
                continue
            if element.paper_id != paper_id:
                errors.append(f"paper_id mismatch for {element_id}: {paper_id} != {element.paper_id}")
        return not errors, errors
