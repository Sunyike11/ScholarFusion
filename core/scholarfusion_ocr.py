from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_pdf_parser import ResearchPDFParser


class PaperParser(Protocol):
    def parse(self, pdf_path: str) -> PaperDocument:
        ...


class NativePDFParser:
    def __init__(self, image_root: str):
        self.parser = ResearchPDFParser(image_root)

    def parse(self, pdf_path: str) -> PaperDocument:
        return self.parser.parse(pdf_path)


@dataclass
class PageQualityGate:
    min_effective_chars: int = 20

    def low_quality_pages(self, paper: PaperDocument) -> List[int]:
        pages = paper.metadata.parse_quality.get("pages") or []
        low = [item["page_number"] for item in pages if item.get("effective_chars", 0) < self.min_effective_chars and item.get("image_count", 0) > 0]
        if paper.metadata.parser_status == "ocr_required" and not low:
            seen = {element.page_number for element in paper.elements if element.page_number}
            return sorted(seen or {1})
        return low


class OCRLayoutFallbackParser:
    """Minimal fallback boundary for scanned/low-text pages.

    This is a deterministic placeholder parser that records OCR fallback activation.
    It can be replaced by MinerU, PaddleOCR or VL structured parsing without changing
    downstream storage contracts.
    """

    parser_name = "ocr_layout_fallback_v1"

    def parse_pages(self, pdf_path: str, paper_id: str, page_numbers: List[int]) -> List[PaperElement]:
        path = Path(pdf_path)
        elements: List[PaperElement] = []
        for index, page_number in enumerate(page_numbers):
            text = f"OCR/Layout fallback required for page {page_number} of {path.name}."
            digest = hashlib.sha1(f"{paper_id}:{page_number}:{text}".encode("utf-8")).hexdigest()[:12]
            elements.append(PaperElement(
                element_id=f"{paper_id}_ocr_page_{page_number}_{digest}",
                paper_id=paper_id,
                element_type="page",
                page_number=page_number,
                section_path=["OCR/Layout Fallback"],
                text=text,
                reading_order=10_000 + index,
                parser_name=self.parser_name,
                parse_confidence=0.25,
                metadata={"ocr_required": True, "fallback_status": "triggered"},
            ))
        return elements


class HybridPaperParser:
    def __init__(self, native_parser: NativePDFParser, fallback_parser: OCRLayoutFallbackParser, quality_gate: PageQualityGate | None = None):
        self.native_parser = native_parser
        self.fallback_parser = fallback_parser
        self.quality_gate = quality_gate or PageQualityGate()

    def parse(self, pdf_path: str) -> PaperDocument:
        paper = self.native_parser.parse(pdf_path)
        low_pages = self.quality_gate.low_quality_pages(paper)
        if low_pages:
            fallback_elements = self.fallback_parser.parse_pages(pdf_path, paper.paper_id, low_pages)
            paper.elements.extend(fallback_elements)
            paper.metadata.parse_quality["ocr_layout_fallback_pages"] = low_pages
            paper.metadata.parse_quality["ocr_layout_fallback_status"] = "triggered"
            if paper.metadata.parser_status == "ocr_required":
                paper.metadata.parser_status = "native_with_ocr_layout_fallback"
        else:
            paper.metadata.parse_quality["ocr_layout_fallback_status"] = "not_required"
        return paper
