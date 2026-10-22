from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import Settings as AppSettings
from core.research_models import PaperDocument, PaperElement, PaperMetadata, utc_now_iso
from core.scholarfusion_figures import enrich_and_deduplicate_figures


FIGURE_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|图)\s*([0-9]+[a-zA-Z]?)\s*[:：.-]\s+(.+)", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(?:fig(?:ure)?\.?)\s*([0-9]+[a-zA-Z]?)\b|图\s*([0-9]+[a-zA-Z]?)", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^\s*((?:[0-9]+(?:\.[0-9]+)*)\s+)?"
    r"(abstract|introduction|related work|method|methods|approach|experiments?|results?|conclusion|limitations?|references)\b",
    re.IGNORECASE,
)


class ResearchPDFParser:
    """Structured paper parser with optional PyMuPDF native extraction."""

    parser_name = "research_pdf_parser_v1"

    def __init__(self, image_root: str = AppSettings.PDF_IMAGE_DIR):
        self.image_root = Path(image_root)
        self.image_root.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: str) -> PaperDocument:
        path = Path(pdf_path)
        file_bytes = path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        paper_id = f"paper_{file_hash[:16]}"
        image_dir = self.image_root / paper_id
        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            import fitz  # type: ignore

            return self._parse_with_fitz(path, file_bytes, file_hash, paper_id, image_dir, fitz)
        except ImportError:
            return self._parse_text_fallback(path, file_bytes, file_hash, paper_id)
        except Exception as exc:
            fallback = self._parse_text_fallback(path, file_bytes, file_hash, paper_id)
            if fallback.elements:
                fallback.metadata.parse_quality["fitz_error"] = str(exc)
                fallback.metadata.parse_quality["parser_dependency"] = "fitz_failed_text_fallback"
                return fallback
            metadata = PaperMetadata(
                title=path.stem,
                source_file=path.name,
                file_sha256=file_hash,
                parser_status="parse_failed",
                parser_name=self.parser_name,
                parse_quality={"error": str(exc)},
            )
            return PaperDocument(paper_id=paper_id, metadata=metadata, elements=[])

    def _parse_with_fitz(self, path: Path, file_bytes: bytes, file_hash: str, paper_id: str, image_dir: Path, fitz: Any) -> PaperDocument:
        doc = fitz.open(path)
        raw_blocks: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        page_quality: List[Dict[str, Any]] = []

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            text = page.get_text("text") or ""
            blocks = page.get_text("dict").get("blocks", [])
            page_quality.append(self._quality_for_page(page_number, text, len(page.get_images(full=True))))

            for block in blocks:
                if block.get("type") == 0:
                    block_text = self._block_text(block)
                    if not block_text.strip():
                        continue
                    raw_blocks.append({
                        "page_number": page_number,
                        "text": block_text.strip(),
                        "bbox": [float(v) for v in block.get("bbox", [])],
                        "font_size": self._max_font_size(block),
                        "raw_order": len(raw_blocks),
                    })
                elif block.get("type") == 1:
                    image_bytes = block.get("image")
                    ext = block.get("ext", "png")
                    if image_bytes:
                        image_index = len(images) + 1
                        filename = f"figure_p{page_number}_{image_index}.{ext}"
                        image_path = image_dir / filename
                        image_path.write_bytes(image_bytes)
                        images.append({
                            "page_number": page_number,
                            "bbox": [float(v) for v in block.get("bbox", [])],
                            "image_path": str(image_path),
                            "image_url": f"/api/research/assets/{paper_id}/{filename}",
                        })

        elements, markdown_debug = self._blocks_to_elements(raw_blocks, images, paper_id, path.name)
        elements = enrich_and_deduplicate_figures(elements)
        parse_quality = self._aggregate_quality(page_quality, elements, images)
        parser_status = "native" if parse_quality["effective_pages"] > 0 else "ocr_required"
        title = self._infer_title(elements, path.stem)

        metadata = PaperMetadata(
            title=title,
            authors=self._infer_authors(elements),
            year=self._infer_year(elements),
            source_file=path.name,
            file_sha256=file_hash,
            parser_status=parser_status,
            parser_name=self.parser_name,
            parse_quality=parse_quality,
        )
        return PaperDocument(
            paper_id=paper_id,
            metadata=metadata,
            elements=elements,
            markdown_debug=markdown_debug,
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
        )

    def _parse_text_fallback(self, path: Path, file_bytes: bytes, file_hash: str, paper_id: str) -> PaperDocument:
        text = file_bytes.decode("utf-8", errors="ignore")
        pages = text.split("\f") if text else [""]
        raw_blocks: List[Dict[str, Any]] = []
        for page_index, page_text in enumerate(pages):
            page_number = page_index + 1
            for part in page_text.splitlines():
                cleaned = part.strip()
                if cleaned:
                    raw_blocks.append({
                        "page_number": page_number,
                        "text": cleaned,
                        "bbox": None,
                        "font_size": 0,
                        "raw_order": len(raw_blocks),
                    })

        elements, markdown_debug = self._blocks_to_elements(raw_blocks, [], paper_id, path.name)
        elements = enrich_and_deduplicate_figures(elements)
        effective_chars = sum(len(element.text) for element in elements)
        parser_status = "native" if effective_chars >= 50 else "ocr_required"
        parse_quality = {
            "parser_dependency": "fitz_unavailable_text_fallback",
            "effective_pages": len({element.page_number for element in elements if element.text}),
            "total_elements": len(elements),
            "effective_chars": effective_chars,
            "ocr_required_pages": [] if parser_status == "native" else list(range(1, len(pages) + 1)),
            "missing_page_numbers": False,
            "figures_without_caption": 0,
        }
        metadata = PaperMetadata(
            title=self._infer_title(elements, path.stem),
            authors=self._infer_authors(elements),
            year=self._infer_year(elements),
            source_file=path.name,
            file_sha256=file_hash,
            parser_status=parser_status,
            parser_name=self.parser_name,
            parse_quality=parse_quality,
        )
        return PaperDocument(paper_id=paper_id, metadata=metadata, elements=elements, markdown_debug=markdown_debug)

    def _blocks_to_elements(
            self,
            raw_blocks: List[Dict[str, Any]],
            images: List[Dict[str, Any]],
            paper_id: str,
            source_file: str,
    ) -> Tuple[List[PaperElement], str]:
        elements: List[PaperElement] = []
        section_path: List[str] = []
        reading_order = 0

        for block in sorted(raw_blocks, key=lambda item: (item["page_number"], item.get("raw_order", 0), item.get("bbox") or [0, 0, 0, 0])):
            text = self._normalize_text(block["text"])
            if not text:
                continue
            element_type = self._classify_text(text, len(elements), block.get("font_size", 0))
            if element_type == "heading":
                section_path = [text]
            elif element_type == "abstract":
                section_path = ["Abstract"]

            element_id = self._stable_element_id(paper_id, element_type, block["page_number"], reading_order, text, block.get("bbox"))
            metadata: Dict[str, Any] = {}
            caption_match = FIGURE_CAPTION_RE.match(text)
            if caption_match:
                element_type = "figure_caption"
                metadata["figure_label"] = caption_match.group(2)
                metadata["caption_body"] = caption_match.group(3)

            elements.append(PaperElement(
                element_id=element_id,
                paper_id=paper_id,
                element_type=element_type,
                page_number=block["page_number"],
                section_path=list(section_path),
                text=text,
                bbox=block.get("bbox"),
                reading_order=reading_order,
                source_file=source_file,
                parser_name=self.parser_name,
                parse_confidence=0.95 if block.get("bbox") else 0.65,
                metadata=metadata,
            ))
            reading_order += 1

        for image_index, image in enumerate(images):
            same_page_captions = [
                element for element in elements
                if element.element_type == "figure_caption" and element.page_number == image["page_number"]
            ]
            caption = self._nearest_caption(image, same_page_captions)
            figure_text = caption.text if caption else ""
            label = caption.metadata.get("figure_label") if caption else None
            figure_id = self._stable_element_id(
                paper_id,
                "figure",
                image["page_number"],
                reading_order + image_index,
                f"{image.get('image_path')} {figure_text}",
                image.get("bbox"),
            )
            mentioned_by = self._find_figure_mentions(elements, label)
            neighbor_ids = self._neighbor_paragraphs(elements, image["page_number"], caption.element_id if caption else None)
            linked_ids = []
            if caption:
                linked_ids.append(caption.element_id)
                caption.linked_element_ids.append(figure_id)
            linked_ids.extend(mentioned_by)
            linked_ids.extend(neighbor_ids)
            section_path = caption.section_path if caption else self._section_for_page(elements, image["page_number"])
            link_status = "caption_and_context" if caption and (mentioned_by or neighbor_ids) else ("caption_only" if caption else "low_confidence")
            elements.append(PaperElement(
                element_id=figure_id,
                paper_id=paper_id,
                element_type="figure",
                page_number=image["page_number"],
                section_path=section_path,
                text=figure_text,
                bbox=image.get("bbox"),
                reading_order=reading_order + image_index,
                linked_element_ids=list(dict.fromkeys(linked_ids)),
                source_file=source_file,
                parser_name=self.parser_name,
                parse_confidence=0.9 if caption else 0.45,
                metadata={
                    "image_path": image.get("image_path"),
                    "image_url": image.get("image_url"),
                    "caption_element_id": caption.element_id if caption else None,
                    "caption": figure_text,
                    "mentioned_by_element_ids": mentioned_by,
                    "neighbor_element_ids": neighbor_ids,
                    "link_status": link_status,
                    "figure_label": label,
                },
            ))

        linked_caption_ids = {
            element.metadata.get("caption_element_id")
            for element in elements
            if element.element_type == "figure"
        }
        if not images:
            synthetic_index = 0
            for caption in [element for element in elements if element.element_type == "figure_caption"]:
                if caption.element_id in linked_caption_ids:
                    continue
                label = caption.metadata.get("figure_label")
                mentioned_by = self._find_figure_mentions(elements, label)
                neighbor_ids = self._neighbor_paragraphs(elements, caption.page_number or 0, caption.element_id)
                figure_id = self._stable_element_id(
                    paper_id,
                    "figure",
                    caption.page_number,
                    reading_order + synthetic_index,
                    caption.text,
                    caption.bbox,
                )
                linked_ids = [caption.element_id] + mentioned_by + neighbor_ids
                caption.linked_element_ids.append(figure_id)
                elements.append(PaperElement(
                    element_id=figure_id,
                    paper_id=paper_id,
                    element_type="figure",
                    page_number=caption.page_number,
                    section_path=caption.section_path,
                    text=caption.text,
                    bbox=caption.bbox,
                    reading_order=reading_order + synthetic_index,
                    linked_element_ids=list(dict.fromkeys(linked_ids)),
                    source_file=source_file,
                    parser_name=self.parser_name,
                    parse_confidence=0.4,
                    metadata={
                        "image_path": None,
                        "image_url": "",
                        "caption_element_id": caption.element_id,
                        "caption": caption.text,
                        "mentioned_by_element_ids": mentioned_by,
                        "neighbor_element_ids": neighbor_ids,
                        "link_status": "caption_only",
                        "figure_label": label,
                        "synthetic_from_caption": True,
                    },
                ))
                synthetic_index += 1

        markdown_debug = "\n\n".join(element.text for element in elements if element.element_type != "figure")
        return elements, markdown_debug

    def _quality_for_page(self, page_number: int, text: str, image_count: int) -> Dict[str, Any]:
        clean_chars = len(re.sub(r"\s+", "", text or ""))
        replacement_ratio = (text.count("\ufffd") / max(len(text), 1)) if text else 0.0
        return {
            "page_number": page_number,
            "effective_chars": clean_chars,
            "replacement_ratio": replacement_ratio,
            "is_scanned_like": clean_chars < 20 and image_count > 0,
            "image_count": image_count,
        }

    def _aggregate_quality(self, page_quality: List[Dict[str, Any]], elements: List[PaperElement], images: List[Dict[str, Any]]) -> Dict[str, Any]:
        ocr_pages = [item["page_number"] for item in page_quality if item["is_scanned_like"]]
        figures = [element for element in elements if element.element_type == "figure"]
        return {
            "pages": page_quality,
            "effective_pages": len([item for item in page_quality if item["effective_chars"] >= 20]),
            "total_elements": len(elements),
            "effective_chars": sum(item["effective_chars"] for item in page_quality),
            "ocr_required_pages": ocr_pages,
            "missing_page_numbers": any(element.page_number is None for element in elements),
            "figures_without_caption": len([element for element in figures if not element.metadata.get("caption_element_id")]),
            "image_blocks": len(images),
        }

    def _classify_text(self, text: str, element_index: int, font_size: float) -> str:
        if element_index == 0:
            return "title"
        lowered = text.lower()
        if lowered.startswith("abstract"):
            return "abstract"
        if lowered.startswith("references"):
            return "reference"
        if FIGURE_CAPTION_RE.match(text):
            return "figure_caption"
        if SECTION_RE.match(text) and len(text) < 120:
            return "heading"
        if font_size and font_size >= 15 and len(text) < 140:
            return "heading"
        return "paragraph"

    def _block_text(self, block: Dict[str, Any]) -> str:
        parts: List[str] = []
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            if line_text.strip():
                parts.append(line_text.strip())
        return " ".join(parts)

    def _max_font_size(self, block: Dict[str, Any]) -> float:
        sizes = [
            float(span.get("size", 0))
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        return max(sizes) if sizes else 0.0

    def _nearest_caption(self, image: Dict[str, Any], captions: List[PaperElement]) -> Optional[PaperElement]:
        if not captions:
            return None
        image_bbox = image.get("bbox") or [0, 0, 0, 0]
        image_center_y = (image_bbox[1] + image_bbox[3]) / 2
        return min(
            captions,
            key=lambda caption: abs((((caption.bbox or [0, 0, 0, 0])[1] + (caption.bbox or [0, 0, 0, 0])[3]) / 2) - image_center_y),
        )

    def _find_figure_mentions(self, elements: List[PaperElement], label: Optional[str]) -> List[str]:
        if not label:
            return []
        mentioned = []
        for element in elements:
            if element.element_type not in {"paragraph", "abstract"}:
                continue
            for match in FIGURE_REF_RE.finditer(element.text):
                if label in {group for group in match.groups() if group}:
                    mentioned.append(element.element_id)
                    break
        return mentioned

    def _neighbor_paragraphs(self, elements: List[PaperElement], page_number: int, caption_id: Optional[str]) -> List[str]:
        page_paragraphs = [
            element for element in elements
            if element.page_number == page_number and element.element_type in {"paragraph", "abstract"}
        ]
        if not page_paragraphs:
            return []
        if not caption_id:
            return [element.element_id for element in page_paragraphs[:2]]
        caption = next((element for element in elements if element.element_id == caption_id), None)
        if not caption:
            return [element.element_id for element in page_paragraphs[:2]]
        return [
            element.element_id for element in sorted(
                page_paragraphs,
                key=lambda item: abs(item.reading_order - caption.reading_order),
            )[:2]
        ]

    def _section_for_page(self, elements: List[PaperElement], page_number: int) -> List[str]:
        candidates = [element for element in elements if element.page_number == page_number and element.section_path]
        return candidates[-1].section_path if candidates else []

    def _infer_title(self, elements: List[PaperElement], fallback: str) -> str:
        title = next((element.text for element in elements if element.element_type == "title" and element.text), "")
        return title or fallback

    def _infer_authors(self, elements: List[PaperElement]) -> List[str]:
        if len(elements) < 2:
            return []
        candidate = elements[1].text
        if any(token in candidate for token in ["@", "University", "Institute", ","]):
            return [part.strip() for part in re.split(r",| and ", candidate) if part.strip()][:12]
        return []

    def _infer_year(self, elements: List[PaperElement]) -> Optional[int]:
        joined = "\n".join(element.text for element in elements[:8])
        match = re.search(r"\b(20[0-9]{2}|19[0-9]{2})\b", joined)
        return int(match.group(1)) if match else None

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _stable_element_id(
            self,
            paper_id: str,
            element_type: str,
            page_number: Optional[int],
            reading_order: int,
            text: str,
            bbox: Optional[List[float]],
    ) -> str:
        payload = f"{paper_id}|{element_type}|{page_number}|{reading_order}|{text[:160]}|{bbox}"
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{paper_id}_{element_type}_{page_number or 0}_{reading_order}_{digest}"
