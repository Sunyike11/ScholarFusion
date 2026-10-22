from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Set

from core.research_models import PaperElement
from core.research_retrieval import STOPWORDS, ResearchRetriever


@dataclass
class EvidenceSufficiencyResult:
    status: str
    confidence: float
    supporting_element_ids: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    reason_code: str = ""
    reason_summary: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ClaimEvidenceResult:
    claim: str
    support_status: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class AnswerabilityVerifier:
    """Deterministic evidence sufficiency gate for research QA."""

    PRIVATE_UNSUPPORTED_TERMS = {"private", "undisclosed", "proprietary", "deployment", "temperature"}
    NUMERIC_TERMS = {"how many", "number", "latency", "fps", "psnr", "ssim", "lpips", "%", "percent"}
    FIGURE_TERMS = {"figure", "fig", "diagram", "architecture", "framework", "chart", "plot"}
    COMPARISON_TERMS = {"compare", "difference", "different", "versus", "vs", "better", "worse"}

    def verify(self, question: str, evidence: List[PaperElement]) -> EvidenceSufficiencyResult:
        if not evidence:
            return EvidenceSufficiencyResult(
                status="insufficient_evidence",
                confidence=0.0,
                reason_code="no_evidence",
                reason_summary="No retrieved evidence is available.",
            )
        question_tokens = self._tokens(question)
        evidence_text = "\n".join(element.text for element in evidence)
        evidence_tokens = self._tokens(evidence_text)
        overlap = question_tokens & evidence_tokens
        recall = len(overlap) / max(len(question_tokens), 1)
        supporting = [element.element_id for element in evidence if self._tokens(element.text) & question_tokens]
        missing: List[str] = []
        reason_code = "sufficient_overlap"

        if self._looks_unanswerable(question, evidence_text):
            return EvidenceSufficiencyResult(
                status="insufficient_evidence",
                confidence=min(recall, 0.2),
                supporting_element_ids=supporting[:5],
                missing_information=["direct evidence for private/undisclosed requested information"],
                reason_code="unsupported_private_or_undisclosed",
                reason_summary="The query asks for private or undisclosed information not directly reported in evidence.",
            )
        if self._needs_numeric(question) and not re.search(r"\d", evidence_text):
            missing.append("explicit numeric evidence")
            reason_code = "missing_numeric_evidence"
        if self._needs_figure(question) and not any(element.element_type in {"figure", "figure_caption"} for element in evidence):
            missing.append("figure or caption evidence")
            reason_code = "missing_figure_evidence"
        if self._needs_comparison(question) and len({element.paper_id for element in evidence}) < 2:
            missing.append("evidence from multiple papers")
            reason_code = "missing_comparison_objects"
        if recall < 0.22:
            missing.append("core query terms")
            reason_code = "low_query_term_coverage"

        confidence = min(1.0, recall + min(len(supporting), 3) * 0.12)
        if supporting and recall >= 0.18:
            confidence = max(confidence, 0.42)
        status = "answerable" if not missing and confidence >= 0.34 else "insufficient_evidence"
        return EvidenceSufficiencyResult(
            status=status,
            confidence=round(confidence, 4),
            supporting_element_ids=supporting[:5],
            missing_information=missing,
            reason_code=reason_code if missing else "direct_evidence_available",
            reason_summary="Evidence passed deterministic sufficiency checks." if not missing else "Evidence is related but does not directly answer every required part.",
        )

    def verify_claims(self, answer: str, evidence: List[PaperElement]) -> List[ClaimEvidenceResult]:
        claims = [
            part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", answer or "")
            if part.strip() and not re.fullmatch(r"\[c\d+\]", part.strip())
        ]
        results: List[ClaimEvidenceResult] = []
        for claim in claims:
            claim_tokens = self._tokens(claim)
            matches = []
            for element in evidence:
                if claim_tokens & self._tokens(element.text):
                    matches.append(element.element_id)
            confidence = min(1.0, len(matches) * 0.35)
            results.append(ClaimEvidenceResult(
                claim=claim,
                support_status="supported" if matches else "unsupported",
                evidence_ids=matches[:5],
                confidence=round(confidence, 4),
            ))
        return results

    def _tokens(self, text: str) -> Set[str]:
        words = set()
        for word in re.findall(r"[a-z0-9_]+", (text or "").lower()):
            if word in STOPWORDS or len(word) <= 1:
                continue
            if word.endswith("ization"):
                word = word[:-7]
            elif word.endswith("ation"):
                word = word[:-5]
            elif word.endswith("ing") and len(word) > 5:
                word = word[:-3]
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]
            words.add(word)
        return words

    def _looks_unanswerable(self, question: str, evidence_text: str) -> bool:
        lowered_q = question.lower()
        if not (self.PRIVATE_UNSUPPORTED_TERMS & set(re.findall(r"[a-z]+", lowered_q))):
            return False
        return not any(term in evidence_text.lower() for term in ["private", "undisclosed", "deployment latency", "temperature"])

    def _needs_numeric(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in self.NUMERIC_TERMS)

    def _needs_figure(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in self.FIGURE_TERMS)

    def _needs_comparison(self, question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in self.COMPARISON_TERMS)
