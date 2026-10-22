from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import Settings as AppSettings
from core.research_store import ResearchStore


@dataclass
class ModelReviewResult:
    decision: str
    confidence: float
    question_is_clear: bool
    answer_is_supported: bool
    answerable_is_correct: bool
    primary_evidence_ids: List[str] = field(default_factory=list)
    acceptable_alternative_element_ids: List[str] = field(default_factory=list)
    context_element_ids: List[str] = field(default_factory=list)
    corrected_question: Optional[str] = None
    corrected_reference_answer: Optional[str] = None
    corrected_answerable: Optional[bool] = None
    reason_codes: List[str] = field(default_factory=list)
    safe_summary: str = ""
    reviewer_model: str = ""
    review_status: str = "model_reviewed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationItemReviewer:
    provider_name = "deterministic_reviewer"

    def review(self, payload: Dict[str, Any]) -> ModelReviewResult:
        answerable = bool(payload.get("candidate_answerable"))
        primary = list(payload.get("candidate_primary_evidence") or [])
        context = list(payload.get("candidate_context_evidence") or [])
        question = str(payload.get("question") or "")
        answer = str(payload.get("candidate_reference_answer") or "")
        reason_codes = []
        decision = "approve"
        confidence = 0.88
        if not question.strip() or len(question.split()) < 4:
            decision = "reject"
            confidence = 0.4
            reason_codes.append("unclear_question")
        if answerable and not primary:
            decision = "reject"
            confidence = 0.45
            reason_codes.append("missing_primary_evidence")
        if answerable and _leaks_answer(question, answer):
            decision = "reject"
            confidence = 0.5
            reason_codes.append("answer_leak")
        if not answerable and primary:
            decision = "revise"
            reason_codes.append("unanswerable_has_evidence")
        return ModelReviewResult(
            decision=decision,
            confidence=confidence,
            question_is_clear=decision != "reject",
            answer_is_supported=(not answerable) or bool(primary),
            answerable_is_correct=True,
            primary_evidence_ids=primary,
            acceptable_alternative_element_ids=list(payload.get("candidate_alternative_evidence") or []),
            context_element_ids=context,
            corrected_question=None,
            corrected_reference_answer=None,
            corrected_answerable=False if "unanswerable_has_evidence" in reason_codes else None,
            reason_codes=reason_codes,
            safe_summary="Deterministic reviewer checked schema, evidence presence and leakage.",
            reviewer_model=self.provider_name,
        )


class DashScopeEvaluationItemReviewer(EvaluationItemReviewer):
    provider_name = "dashscope_text_reviewer"

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: str = AppSettings.DASHSCOPE_TEXT_MODEL,
            timeout: float = AppSettings.DASHSCOPE_REQUEST_TIMEOUT,
    ):
        self.api_key = api_key or AppSettings.API_KEY
        self.base_url = self._chat_base_url((base_url or AppSettings.API_BASE_URL or "").rstrip("/"))
        self.model = model
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing")

    def review(self, payload: Dict[str, Any]) -> ModelReviewResult:
        safe_payload = dict(payload)
        prompt = (
            "You are an independent evaluator for a provisional research-paper QA benchmark. "
            "Use only the provided evidence previews. Do not use system retrieval results or outside knowledge. "
            "Return strict JSON with keys: decision, confidence, question_is_clear, "
            "answer_is_supported, answerable_is_correct, primary_evidence_ids, "
            "acceptable_alternative_element_ids, context_element_ids, corrected_question, "
            "corrected_reference_answer, corrected_answerable, reason_codes, safe_summary. "
            "Approve when the candidate reference answer is directly supported by the evidence preview, "
            "including when the answer is a concise excerpt copied or summarized from that evidence. "
            "Use revise when the question or answer can be minimally corrected while keeping the same evidence. "
            "Reject only when evidence is missing, the answer goes beyond evidence, the question is unclear, "
            "or the answerable label is wrong. For unanswerable items, approve if no provided evidence answers "
            "the requested private/undisclosed fact. Do not include reasoning.\n"
            + json.dumps(safe_payload, ensure_ascii=False)
        )
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return JSON only. Keep summaries short and safe."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        raw = self._request(request_body)
        data = self._parse(raw)
        return self._coerce(data)

    def _chat_base_url(self, configured: str) -> str:
        if not configured or "maas.aliyuncs.com/api/v1" in configured or configured.endswith("/api/v1"):
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return configured

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:240]
            raise RuntimeError(f"Reviewer HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Reviewer request failed: {exc.reason}") from exc

    def _parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if "```" in content:
            content = content.split("```", 2)[1].replace("json", "", 1).strip()
        return json.loads(content)

    def _coerce(self, data: Dict[str, Any]) -> ModelReviewResult:
        decision = str(data.get("decision", "reject")).lower()
        if decision not in {"approve", "revise", "reject"}:
            decision = "reject"
        return ModelReviewResult(
            decision=decision,
            confidence=_confidence(data.get("confidence", 0.0)),
            question_is_clear=bool(data.get("question_is_clear")),
            answer_is_supported=bool(data.get("answer_is_supported")),
            answerable_is_correct=bool(data.get("answerable_is_correct")),
            primary_evidence_ids=list(data.get("primary_evidence_ids") or []),
            acceptable_alternative_element_ids=list(data.get("acceptable_alternative_element_ids") or []),
            context_element_ids=list(data.get("context_element_ids") or []),
            corrected_question=data.get("corrected_question"),
            corrected_reference_answer=data.get("corrected_reference_answer"),
            corrected_answerable=data.get("corrected_answerable"),
            reason_codes=list(data.get("reason_codes") or []),
            safe_summary=str(data.get("safe_summary") or "")[:240],
            reviewer_model=self.model,
        )


class ModelReviewedDatasetValidator:
    def __init__(self, store: ResearchStore):
        self.store = store

    def validate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        paper_ids = set(item.get("paper_ids") or [])
        all_ids = (
            list(item.get("gold_primary_element_ids") or [])
            + list(item.get("acceptable_alternative_element_ids") or [])
            + list(item.get("gold_context_element_ids") or [])
        )
        if not item.get("query_id"):
            errors.append("missing_query_id")
        for paper_id in paper_ids:
            if not self.store.get_paper(paper_id, include_elements=False):
                errors.append(f"missing_paper:{paper_id}")
        for element_id in all_ids:
            element = self.store.get_element(element_id)
            if not element:
                errors.append(f"missing_element:{element_id}")
                continue
            if paper_ids and element.paper_id not in paper_ids:
                errors.append(f"evidence_wrong_paper:{element_id}")
            page_numbers = set(item.get("relevant_page_numbers") or [])
            if page_numbers and element.page_number is not None and element.page_number not in page_numbers:
                warnings.append(f"page_not_listed:{element_id}")
        if item.get("answerable") and not item.get("gold_primary_element_ids"):
            errors.append("answerable_without_primary_evidence")
        qtype = item.get("query_type")
        primary_types = [self.store.get_element(eid).element_type for eid in item.get("gold_primary_element_ids", []) if self.store.get_element(eid)]
        if qtype == "text_to_figure" and not any(t in {"figure", "figure_caption"} for t in primary_types):
            errors.append("text_to_figure_without_figure_evidence")
        if qtype == "image_to_image" and not any(t == "figure" for t in primary_types):
            errors.append("image_to_image_without_figure")
        if qtype == "cross_paper_comparison":
            covered = {self.store.get_element(eid).paper_id for eid in item.get("gold_primary_element_ids", []) if self.store.get_element(eid)}
            if paper_ids and not paper_ids.issubset(covered):
                errors.append("cross_paper_evidence_incomplete")
        if qtype == "unanswerable" and item.get("answerable"):
            errors.append("unanswerable_marked_answerable")
        if _leaks_answer(str(item.get("question") or ""), str(item.get("reference_answer") or "")):
            errors.append("answer_leak")
        if len(set(all_ids)) < len(all_ids):
            warnings.append("duplicate_evidence_ids")
        return {
            "validation_status": "valid" if not errors else "invalid",
            "validation_errors": errors,
            "validation_warnings": warnings,
        }


def build_review_payload(record: Dict[str, Any], store: ResearchStore, max_preview_chars: int = 420) -> Dict[str, Any]:
    evidence_ids = list(dict.fromkeys(record.get("relevant_element_ids") or []))
    evidence_previews = []
    for element_id in evidence_ids[:8]:
        element = store.get_element(element_id)
        if not element:
            continue
        evidence_previews.append({
            "element_id": element.element_id,
            "paper_id": element.paper_id,
            "page_number": element.page_number,
            "element_type": element.element_type,
            "section_path": element.section_path,
            "text_preview": _short(element.text, max_preview_chars),
        })
    primary = evidence_ids[:2] if record.get("answerable") else []
    context = evidence_ids[2:6] if record.get("answerable") else []
    return {
        "query_id": record.get("query_id"),
        "query_type": record.get("query_type"),
        "question": record.get("question"),
        "candidate_reference_answer": record.get("reference_answer_candidate") or record.get("reference_answer"),
        "candidate_answerable": record.get("answerable"),
        "paper_ids": record.get("paper_ids") or [],
        "candidate_primary_evidence": primary,
        "candidate_alternative_evidence": evidence_ids[2:4],
        "candidate_context_evidence": context,
        "evidence_previews": evidence_previews,
        "page_metadata": record.get("relevant_page_numbers") or [],
        "query_image_element_id": record.get("query_image_element_id"),
    }


def apply_review(record: Dict[str, Any], review: ModelReviewResult) -> Dict[str, Any]:
    item = dict(record)
    if review.corrected_question:
        item["question"] = review.corrected_question
    if review.corrected_reference_answer:
        item["reference_answer_candidate"] = review.corrected_reference_answer
    if review.corrected_answerable is not None:
        item["answerable"] = bool(review.corrected_answerable)
    item["reference_answer"] = item.get("reference_answer_candidate", "")
    item["gold_primary_element_ids"] = list(dict.fromkeys(review.primary_evidence_ids))
    item["acceptable_alternative_element_ids"] = list(dict.fromkeys(review.acceptable_alternative_element_ids))
    item["gold_context_element_ids"] = list(dict.fromkeys(review.context_element_ids))
    item["review_status"] = "model_reviewed"
    item["dataset_status"] = "provisional"
    item["not_gold"] = True
    item["human_verified"] = False
    item["model_reviewer_confidence"] = review.confidence
    item["model_review_decision"] = review.decision
    item["model_review_reason_codes"] = review.reason_codes
    item["model_review_summary"] = review.safe_summary
    item["reviewer_model"] = review.reviewer_model
    item["relevant_element_ids"] = list(dict.fromkeys(
        item["gold_primary_element_ids"] + item["acceptable_alternative_element_ids"] + item["gold_context_element_ids"]
    ))
    return item


def deduplicate_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = "|".join([
            str(record.get("query_type") or ""),
            str(record.get("query_image_element_id") or ""),
            re.sub(r"\s+", " ", str(record.get("question") or "").strip().lower()),
        ])
        if key in seen:
            rejected.append(record | {"validation_errors": list(record.get("validation_errors", [])) + ["duplicate_question"]})
            continue
        seen.add(key)
        accepted.append(record)
    return accepted, rejected


def _short(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    lowered = str(value or "").strip().lower()
    if lowered in {"high", "very high"}:
        return 0.9
    if lowered in {"medium", "moderate"}:
        return 0.65
    if lowered in {"low"}:
        return 0.3
    try:
        return float(lowered)
    except ValueError:
        return 0.0


def _leaks_answer(question: str, answer: str) -> bool:
    q = set(re.findall(r"[a-z0-9_]{5,}", question.lower()))
    a = set(re.findall(r"[a-z0-9_]{5,}", answer.lower()))
    if not q or not a:
        return False
    return len(q & a) / max(len(a), 1) > 0.65
