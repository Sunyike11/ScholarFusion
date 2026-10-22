from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.research_models import PaperElement


FIGURE_TYPES = {
    "architecture": ["architecture", "framework", "overview", "network"],
    "qualitative_result": ["qualitative", "visual", "render", "result"],
    "quantitative_chart": ["quantitative", "chart", "curve", "plot", "metric"],
    "ablation": ["ablation", "component", "variant"],
    "dataset_sample": ["dataset", "sample", "capture"],
    "pipeline": ["pipeline", "process", "workflow"],
}


def enrich_and_deduplicate_figures(elements: List[PaperElement]) -> List[PaperElement]:
    """Add figure metadata and mark duplicate/decorative image blocks in-place."""

    figures = [element for element in elements if element.element_type == "figure"]
    by_hash: Dict[str, PaperElement] = {}
    by_page_label: Dict[str, PaperElement] = {}
    by_page_caption: Dict[str, PaperElement] = {}
    seen_on_page: List[PaperElement] = []

    for figure in figures:
        metadata = figure.metadata
        image_path = metadata.get("image_path")
        bbox = figure.bbox or []
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        area_ratio = _area_ratio(bbox)
        image_hash = _file_hash(image_path)
        perceptual_hash = _perceptual_hash(image_path, bbox)
        caption = str(metadata.get("caption") or figure.text or "")
        figure_number = metadata.get("figure_label") or _figure_number(caption)
        figure_type = _figure_type(caption)
        is_decorative = _is_decorative(width, height, area_ratio, caption)

        metadata.update({
            "figure_id": figure.element_id,
            "figure_number": figure_number,
            "image_hash": image_hash,
            "perceptual_hash": perceptual_hash,
            "width": width,
            "height": height,
            "area_ratio": area_ratio,
            "caption_id": metadata.get("caption_element_id"),
            "figure_type": "decorative" if is_decorative else figure_type,
            "is_decorative": is_decorative,
            "is_duplicate": False,
            "duplicate_of": None,
            "link_evidence": _link_evidence(figure),
        })

        duplicate_of = None
        if is_decorative:
            figure.parse_confidence = min(figure.parse_confidence, 0.2)
        elif image_hash and image_hash in by_hash:
            duplicate_of = by_hash[image_hash].element_id
        elif figure_number:
            key = f"{figure.paper_id}:{figure.page_number}:{figure_number}"
            if key in by_page_label:
                duplicate_of = by_page_label[key].element_id
            else:
                by_page_label[key] = figure
        if not duplicate_of and caption:
            normalized_caption = _normalize_caption(caption)
            key = f"{figure.paper_id}:{figure.page_number}:{normalized_caption}"
            if normalized_caption and key in by_page_caption:
                duplicate_of = by_page_caption[key].element_id
            elif normalized_caption:
                by_page_caption[key] = figure
        if not duplicate_of:
            overlapping = _overlapping_same_page(figure, seen_on_page)
            if overlapping:
                duplicate_of = overlapping.element_id
        if duplicate_of:
            metadata["is_duplicate"] = True
            metadata["duplicate_of"] = duplicate_of
            figure.parse_confidence = min(figure.parse_confidence, 0.35)
        else:
            seen_on_page.append(figure)
            if image_hash:
                by_hash[image_hash] = figure
    return elements


def unique_non_decorative_figures(elements: List[PaperElement]) -> List[PaperElement]:
    return [
        element for element in elements
        if element.element_type == "figure"
        and not element.metadata.get("is_duplicate")
        and not element.metadata.get("is_decorative")
    ]


def _file_hash(path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perceptual_hash(path_value: Any, bbox: Optional[List[float]]) -> Optional[str]:
    if path_value:
        path = Path(str(path_value))
        if path.exists() and path.is_file():
            data = path.read_bytes()
            return hashlib.sha1(data[:4096]).hexdigest()[:16]
    if bbox:
        payload = ",".join(str(round(float(value), 1)) for value in bbox)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return None


def _bbox_width(bbox: List[float]) -> float:
    return max(float(bbox[2] - bbox[0]), 0.0) if len(bbox) >= 4 else 0.0


def _bbox_height(bbox: List[float]) -> float:
    return max(float(bbox[3] - bbox[1]), 0.0) if len(bbox) >= 4 else 0.0


def _area_ratio(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    area = _bbox_width(bbox) * _bbox_height(bbox)
    # Approximate A4/letter PDF point area. Used only for filtering tiny fragments.
    return round(area / max(612 * 792, 1), 6)


def _figure_number(caption: str) -> Optional[str]:
    match = re.search(r"\bfig(?:ure)?\.?\s*([0-9]+[a-zA-Z]?)\b|图\s*([0-9]+[a-zA-Z]?)", caption, re.I)
    if not match:
        return None
    return next((group for group in match.groups() if group), None)


def _figure_type(caption: str) -> str:
    lowered = caption.lower()
    for figure_type, keywords in FIGURE_TYPES.items():
        if any(keyword in lowered for keyword in keywords):
            return figure_type
    return "other"


def _is_decorative(width: float, height: float, area_ratio: float, caption: str) -> bool:
    if caption.strip():
        return False
    if width <= 40 or height <= 40:
        return True
    if area_ratio and area_ratio < 0.004:
        return True
    return False


def _normalize_caption(caption: str) -> str:
    caption = re.sub(r"\s+", " ", caption or "").strip().lower()
    return caption[:140]


def _overlapping_same_page(figure: PaperElement, seen: List[PaperElement]) -> Optional[PaperElement]:
    for other in seen:
        if other.paper_id != figure.paper_id or other.page_number != figure.page_number:
            continue
        if _iou(figure.bbox or [], other.bbox or []) >= 0.88:
            return other
    return None


def _iou(left: List[float], right: List[float]) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union else 0.0


def _link_evidence(figure: PaperElement) -> List[Dict[str, Any]]:
    evidence = []
    caption_id = figure.metadata.get("caption_element_id")
    if caption_id:
        evidence.append({
            "link_type": "caption_neighbor",
            "confidence": 0.9,
            "rule_name": "figure_number_or_nearest_caption",
            "distance": 0,
            "element_id": caption_id,
        })
    for element_id in figure.metadata.get("mentioned_by_element_ids") or []:
        evidence.append({
            "link_type": "explicit_reference",
            "confidence": 0.85,
            "rule_name": "figure_reference_match",
            "distance": None,
            "element_id": element_id,
        })
    neighbor_ids = figure.metadata.get("neighbor_element_ids") or []
    for distance, element_id in enumerate(neighbor_ids, start=1):
        evidence.append({
            "link_type": "same_page_neighbor",
            "confidence": 0.55,
            "rule_name": "same_page_neighbor",
            "distance": distance,
            "element_id": element_id,
        })
    if not evidence and figure.section_path:
        evidence.append({
            "link_type": "same_section_fallback",
            "confidence": 0.3,
            "rule_name": "same_section_fallback",
            "distance": None,
            "element_id": None,
        })
    return evidence
