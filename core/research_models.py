from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PAPER_ELEMENT_TYPES = {
    "title",
    "abstract",
    "heading",
    "paragraph",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "equation",
    "reference",
    "page",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceRef:
    paper_id: str
    element_id: str
    page_number: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    element_type: str = "paragraph"
    evidence_text: str = ""
    figure_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRef":
        return cls(**data)


@dataclass
class PaperMetadata:
    title: str = "unknown"
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = "unknown"
    source_file: str = ""
    file_sha256: str = ""
    parser_status: str = "native"
    parser_name: str = "research_pdf_parser"
    parse_quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperMetadata":
        return cls(**data)


@dataclass
class PaperElement:
    element_id: str
    paper_id: str
    element_type: str
    page_number: Optional[int]
    section_path: List[str]
    text: str
    bbox: Optional[List[float]] = None
    reading_order: int = 0
    parent_id: Optional[str] = None
    linked_element_ids: List[str] = field(default_factory=list)
    source_file: str = ""
    parser_name: str = "research_pdf_parser"
    parse_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    representation_ids: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.element_type not in PAPER_ELEMENT_TYPES:
            raise ValueError(f"Unsupported paper element type: {self.element_type}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperElement":
        return cls(**data)


@dataclass
class FigureElement(PaperElement):
    image_path: str = ""
    image_url: str = ""
    caption: str = ""
    caption_element_id: Optional[str] = None
    mentioned_by_element_ids: List[str] = field(default_factory=list)
    neighbor_element_ids: List[str] = field(default_factory=list)
    link_status: str = "unlinked"

    def __post_init__(self) -> None:
        self.element_type = "figure"
        super().__post_init__()


@dataclass
class SectionElement(PaperElement):
    def __post_init__(self) -> None:
        self.element_type = "heading"
        super().__post_init__()


@dataclass
class ParagraphElement(PaperElement):
    def __post_init__(self) -> None:
        if self.element_type not in {"paragraph", "abstract"}:
            self.element_type = "paragraph"
        super().__post_init__()


@dataclass
class PaperDocument:
    paper_id: str
    metadata: PaperMetadata
    elements: List[PaperElement] = field(default_factory=list)
    markdown_debug: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "metadata": self.metadata.to_dict(),
            "elements": [element.to_dict() for element in self.elements],
            "markdown_debug": self.markdown_debug,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperDocument":
        return cls(
            paper_id=data["paper_id"],
            metadata=PaperMetadata.from_dict(data["metadata"]),
            elements=[PaperElement.from_dict(item) for item in data.get("elements", [])],
            markdown_debug=data.get("markdown_debug", ""),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )


@dataclass
class PaperCardField:
    value: Any = "unknown"
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperCardField":
        return cls(
            value=data.get("value", "unknown"),
            evidence_refs=[EvidenceRef.from_dict(ref) for ref in data.get("evidence_refs", [])],
            confidence=float(data.get("confidence", 0.0)),
            source=data.get("source", "auto"),
        )


PAPER_CARD_FIELDS = [
    "title",
    "authors",
    "year",
    "venue",
    "task",
    "input",
    "output",
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
    "keywords",
]


@dataclass
class PaperCard:
    paper_id: str
    fields: Dict[str, PaperCardField]
    version: str = "research-card-v1"
    generated_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    is_manual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "fields": {key: value.to_dict() for key, value in self.fields.items()},
            "version": self.version,
            "generated_at": self.generated_at,
            "updated_at": self.updated_at,
            "is_manual": self.is_manual,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperCard":
        return cls(
            paper_id=data["paper_id"],
            fields={key: PaperCardField.from_dict(value) for key, value in data.get("fields", {}).items()},
            version=data.get("version", "research-card-v1"),
            generated_at=data.get("generated_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            is_manual=bool(data.get("is_manual", False)),
        )


@dataclass
class SimilarityDimension:
    dimension: str
    score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimilarPaperResult:
    paper_id: str
    overall_score: float
    matched_dimensions: List[SimilarityDimension] = field(default_factory=list)
    different_dimensions: List[SimilarityDimension] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "overall_score": self.overall_score,
            "matched_dimensions": [item.to_dict() for item in self.matched_dimensions],
            "different_dimensions": [item.to_dict() for item in self.different_dimensions],
        }


@dataclass
class ComparisonCell:
    paper_id: str
    value: Any
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    fact_type: str = "missing"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "value": self.value,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "fact_type": self.fact_type,
        }


@dataclass
class ComparisonReport:
    base_paper_id: str
    compared_paper_ids: List[str]
    rows: Dict[str, List[ComparisonCell]]
    markdown_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_paper_id": self.base_paper_id,
            "compared_paper_ids": self.compared_paper_ids,
            "rows": {
                key: [cell.to_dict() for cell in cells]
                for key, cells in self.rows.items()
            },
            "markdown_report": self.markdown_report,
        }
