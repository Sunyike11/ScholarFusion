from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from core.research_models import PaperElement
from core.scholarfusion_verifier import AnswerabilityVerifier


@dataclass
class ClaimSupport:
    claim: str
    support_status: str
    primary_evidence_ids: List[str] = field(default_factory=list)
    context_evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CitationPlanner:
    def __init__(self):
        self.verifier = AnswerabilityVerifier()

    def plan(self, answer: str, evidence: List[PaperElement]) -> Dict[str, Any]:
        claim_results = self.verifier.verify_claims(answer, evidence)
        primary_ids = []
        context_ids = []
        claims: List[ClaimSupport] = []
        for result in claim_results:
            support_ids = list(result.evidence_ids)
            primary = support_ids[:2]
            context = [element.element_id for element in evidence if element.element_id not in primary][:3]
            primary_ids.extend(primary)
            context_ids.extend(context)
            status = result.support_status
            if status == "unsupported":
                confidence = 0.0
            else:
                confidence = result.confidence
            claims.append(ClaimSupport(
                claim=result.claim,
                support_status=status,
                primary_evidence_ids=primary,
                context_evidence_ids=context,
                confidence=confidence,
            ))
        return {
            "claim_support": [claim.to_dict() for claim in claims],
            "primary_evidence_ids": list(dict.fromkeys(primary_ids)),
            "context_evidence_ids": list(dict.fromkeys(context_ids)),
        }


def citation_quality_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    valid = []
    page = []
    evidence_precision = []
    evidence_recall = []
    claim_support = []
    for record in records:
        relevant = set(record.get("relevant_element_ids") or [])
        primary = set(record.get("primary_citation_ids") or [])
        context = set(record.get("context_citation_ids") or [])
        all_cited = primary | context
        valid.append(1.0 if all_cited else 0.0)
        if relevant:
            evidence_precision.append(len(primary & relevant) / max(len(primary), 1))
            evidence_recall.append(len(primary & relevant) / len(relevant))
        page.append(1.0 if record.get("page_valid", True) else 0.0)
        claim_items = record.get("claim_support") or []
        if claim_items:
            claim_support.append(len([item for item in claim_items if item.get("support_status") == "supported"]) / len(claim_items))
    return {
        "citation_id_validity": _mean(valid),
        "citation_page_accuracy": _mean(page),
        "citation_evidence_precision": _mean(evidence_precision),
        "citation_evidence_recall": _mean(evidence_recall),
        "claim_support_rate": _mean(claim_support),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
