from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings as AppSettings
from core.research_models import PAPER_CARD_FIELDS, PaperCard, PaperCardField, EvidenceRef
from core.research_store import ResearchStore


class PaperCardLLMError(RuntimeError):
    pass


class DashScopePaperCardBuilder:
    def __init__(self, store: ResearchStore, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = AppSettings.DASHSCOPE_TEXT_MODEL):
        self.store = store
        self.api_key = api_key or AppSettings.API_KEY
        configured_base_url = (base_url or AppSettings.API_BASE_URL or "").rstrip("/")
        self.base_url = self._chat_base_url(configured_base_url)
        self.model = model
        if not self.api_key:
            raise PaperCardLLMError("DASHSCOPE_API_KEY is missing")
        if not self.base_url:
            raise PaperCardLLMError("DASHSCOPE_BASE_URL is missing")

    def _chat_base_url(self, configured: str) -> str:
        if not configured:
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if "compatible-mode" in configured or configured.endswith("/chat/completions"):
            return configured
        if "maas.aliyuncs.com/api/v1" in configured or configured.endswith("/api/v1"):
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return configured

    def build(self, paper_id: str) -> Dict[str, Any]:
        paper = self.store.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Unknown paper_id: {paper_id}")
        context_elements = [
            element for element in paper.elements
            if element.element_type in {"title", "abstract", "heading", "paragraph", "figure_caption"}
            and any(word in (element.text + " " + " ".join(element.section_path)).lower() for word in ["abstract", "method", "approach", "experiment", "dataset", "metric", "gaussian", "smpl"])
        ][:24]
        manifest = [
            {
                "element_id": element.element_id,
                "page_number": element.page_number,
                "section_path": element.section_path,
                "text": element.text[:500],
            }
            for element in context_elements
        ]
        prompt = (
            "Extract a structured research paper card as strict JSON. "
            "Use only the provided evidence elements. For each field, return value, confidence, evidence_ids. "
            "Use unknown if evidence is insufficient. Fields: "
            + ", ".join(PAPER_CARD_FIELDS)
            + "\nEvidence:\n"
            + json.dumps(manifest, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You extract evidence-grounded paper cards and output JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        raw = self._request(payload)
        candidate = self._parse(raw)
        validated = self._validate(paper_id, candidate)
        return {"raw_candidate": candidate, "validated_card": validated, "validation_errors": validated.get("_validation_errors", [])}

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=AppSettings.DASHSCOPE_REQUEST_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise PaperCardLLMError(f"Paper card LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PaperCardLLMError(f"Paper card LLM request failed: {exc.reason}") from exc

    def _parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        if "```" in content:
            content = content.split("```", 2)[1].replace("json", "", 1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise PaperCardLLMError("Paper card LLM did not return valid JSON") from exc

    def _validate(self, paper_id: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        validated: Dict[str, Any] = {}
        for field in PAPER_CARD_FIELDS:
            item = candidate.get(field) if isinstance(candidate.get(field), dict) else {"value": candidate.get(field, "unknown")}
            evidence_ids = [eid for eid in item.get("evidence_ids", []) if self.store.element_exists(paper_id, eid)]
            if item.get("evidence_ids") and len(evidence_ids) != len(item.get("evidence_ids", [])):
                errors.append(f"{field}: dropped invalid evidence ids")
            value = item.get("value", "unknown")
            if value != "unknown" and not evidence_ids and field not in {"venue"}:
                value = "unknown"
                errors.append(f"{field}: value reset to unknown because evidence is missing")
            validated[field] = {
                "value": value,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "evidence_ids": evidence_ids,
                "source": "llm",
            }
        validated["_validation_errors"] = errors
        return validated


def card_metrics(card: Dict[str, Any]) -> Dict[str, float]:
    fields = [field for field in PAPER_CARD_FIELDS if isinstance(card.get(field), dict)]
    if not fields:
        return {"field_coverage": 0.0, "evidence_validity": 0.0, "unknown_rate": 1.0, "conflict_count": 0.0}
    known = [field for field in fields if _known_value(card[field].get("value"))]
    with_evidence = [field for field in known if card[field].get("evidence_ids")]
    return {
        "field_coverage": len(known) / len(fields),
        "evidence_validity": len(with_evidence) / max(len(known), 1),
        "unknown_rate": 1 - len(known) / len(fields),
        "conflict_count": 0.0,
    }


def _known_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "null", "none"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True
