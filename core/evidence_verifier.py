import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: Optional[float] = None
    source: str = "unknown"
    page: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceCheckResult:
    grounded: bool
    answerable: bool
    cited_chunk_ids: List[str]
    unsupported_claims: List[str]
    suggested_revision: str


class EvidenceVerifier:
    """
    Deterministic evidence verifier for citation-grounded RAG answers.

    This is intentionally not an LLM judge. It only checks whether answer
    sentences are covered by retrieved chunk text using exact/overlap signals.
    """

    def __init__(
            self,
            min_top_score: float = 0.5,
            min_claim_coverage: float = 0.62,
            min_claim_tokens: int = 2
    ) -> None:
        self.min_top_score = min_top_score
        self.min_claim_coverage = min_claim_coverage
        self.min_claim_tokens = min_claim_tokens

    def verify(
            self,
            user_query: str,
            answer: str,
            retrieved_chunks: Sequence[Dict[str, Any] | RetrievedChunk],
    ) -> EvidenceCheckResult:
        chunks = [self._coerce_chunk(chunk, idx) for idx, chunk in enumerate(retrieved_chunks)]

        if not chunks:
            return EvidenceCheckResult(
                grounded=False,
                answerable=False,
                cited_chunk_ids=[],
                unsupported_claims=[],
                suggested_revision=self._refusal("没有检索到可用证据"),
            )

        top_score = self._top_score(chunks)
        if top_score is not None and top_score < self.min_top_score:
            return EvidenceCheckResult(
                grounded=False,
                answerable=False,
                cited_chunk_ids=[],
                unsupported_claims=[],
                suggested_revision=self._refusal(f"最高检索分数 {top_score:.3f} 低于阈值 {self.min_top_score:.3f}"),
            )

        claims = self._split_claims(answer)
        unsupported_claims: List[str] = []
        cited_chunk_ids: List[str] = []
        supported_claims: List[str] = []

        for claim in claims:
            supporting_chunks = self._find_supporting_chunks(claim, chunks)
            if supporting_chunks:
                supported_claims.append(claim)
                for chunk in supporting_chunks:
                    if chunk.chunk_id not in cited_chunk_ids:
                        cited_chunk_ids.append(chunk.chunk_id)
            else:
                unsupported_claims.append(claim)

        grounded = bool(claims) and not unsupported_claims
        suggested_revision = self._build_suggested_revision(
            supported_claims=supported_claims,
            cited_chunk_ids=cited_chunk_ids,
            chunks=chunks,
            unsupported_claims=unsupported_claims,
        )

        return EvidenceCheckResult(
            grounded=grounded,
            answerable=True,
            cited_chunk_ids=cited_chunk_ids,
            unsupported_claims=unsupported_claims,
            suggested_revision=suggested_revision,
        )

    def add_citations(self, answer: str, chunk_ids: Sequence[str], chunks: Sequence[Dict[str, Any] | RetrievedChunk]) -> str:
        if not chunk_ids:
            return answer
        coerced_chunks = [self._coerce_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        citations = self._format_citations(chunk_ids, coerced_chunks)
        if not citations:
            return answer
        if "[source:" in answer:
            return answer
        return f"{answer.rstrip()}\n\n{citations}"

    def _build_suggested_revision(
            self,
            supported_claims: Sequence[str],
            cited_chunk_ids: Sequence[str],
            chunks: Sequence[RetrievedChunk],
            unsupported_claims: Sequence[str],
    ) -> str:
        if not supported_claims:
            return self._refusal("现有证据不足以支持回答")

        prefix = "根据已检索到的资料，"
        body = " ".join(supported_claims)
        if unsupported_claims:
            body += " 其余内容缺少知识库证据支持，已省略。"
        citations = self._format_citations(cited_chunk_ids, chunks)
        return f"{prefix}{body}\n\n{citations}".strip()

    def _find_supporting_chunks(self, claim: str, chunks: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        claim_norm = self._normalize(claim)
        claim_tokens = self._tokens(claim_norm)
        if not claim_norm or len(claim_tokens) < self.min_claim_tokens:
            return []

        supporting: List[RetrievedChunk] = []
        for chunk in chunks:
            chunk_norm = self._normalize(chunk.text)
            if not chunk_norm:
                continue
            if claim_norm in chunk_norm:
                supporting.append(chunk)
                continue

            chunk_tokens = self._tokens(chunk_norm)
            if not chunk_tokens:
                continue
            coverage = len(claim_tokens & chunk_tokens) / max(len(claim_tokens), 1)
            if coverage >= self.min_claim_coverage:
                supporting.append(chunk)

        return supporting

    def _split_claims(self, answer: str) -> List[str]:
        cleaned = re.sub(r"\[source:[^\]]+\]", "", answer or "", flags=re.IGNORECASE)
        parts = re.split(r"(?<=[。！？!?；;])\s+|\n+", cleaned)
        claims: List[str] = []
        for part in parts:
            subparts = re.split(r"(?<=[。！？!?；;])", part)
            for subpart in subparts:
                claim = subpart.strip()
                if claim:
                    claims.append(claim)
        return claims

    def _tokens(self, normalized_text: str) -> Set[str]:
        words = set(re.findall(r"[a-z0-9_]+", normalized_text))
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized_text)
        cjk_bigrams = {cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)}
        return words | cjk_bigrams

    def _normalize(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"\[source:[^\]]+\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _format_citations(self, chunk_ids: Sequence[str], chunks: Sequence[RetrievedChunk]) -> str:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        citations = []
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if not chunk:
                continue
            page = chunk.page if chunk.page is not None else "-"
            citations.append(f"[source: {chunk.source}, page {page}, {chunk.chunk_id}]")
        return " ".join(citations)

    def _refusal(self, reason: str) -> str:
        return f"我无法根据当前知识库证据可靠回答这个问题。原因：{reason}。请补充相关文档或重新提问。"

    def _top_score(self, chunks: Sequence[RetrievedChunk]) -> Optional[float]:
        scores = [chunk.score for chunk in chunks if isinstance(chunk.score, (int, float))]
        return max(scores) if scores else None

    def _coerce_chunk(self, chunk: Dict[str, Any] | RetrievedChunk, idx: int) -> RetrievedChunk:
        if isinstance(chunk, RetrievedChunk):
            return chunk

        metadata = chunk.get("metadata", {}) or {}
        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or metadata.get("chunk_id")
            or metadata.get("node_id")
            or metadata.get("doc_id")
            or f"chunk_{idx}"
        )
        source = (
            chunk.get("source")
            or metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("file_path")
            or "unknown"
        )
        page = chunk.get("page", metadata.get("page", metadata.get("page_number")))
        return RetrievedChunk(
            chunk_id=str(chunk_id),
            text=str(chunk.get("text", chunk.get("content", "")) or ""),
            score=chunk.get("score"),
            source=str(source),
            page=page,
            metadata=metadata,
        )
