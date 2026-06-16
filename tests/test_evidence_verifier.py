from core.evidence_verifier import EvidenceVerifier


def test_supported_answer_is_grounded_with_citation():
    verifier = EvidenceVerifier(min_top_score=0.5)
    chunks = [
        {
            "chunk_id": "handbook#chunk_1",
            "text": "员工年假为10天。病假需要提交医生证明。",
            "score": 0.91,
            "source": "handbook.pdf",
            "page": 3,
        }
    ]

    result = verifier.verify(
        user_query="员工年假是多少？",
        answer="员工年假为10天。",
        retrieved_chunks=chunks,
    )

    assert result.answerable is True
    assert result.grounded is True
    assert result.cited_chunk_ids == ["handbook#chunk_1"]
    assert result.unsupported_claims == []
    assert "[source: handbook.pdf, page 3, handbook#chunk_1]" in result.suggested_revision


def test_unsupported_answer_is_not_grounded():
    verifier = EvidenceVerifier(min_top_score=0.5)
    chunks = [
        {
            "chunk_id": "policy#chunk_1",
            "text": "报销申请需要在30天内提交。",
            "score": 0.88,
            "source": "policy.md",
            "page": 1,
        }
    ]

    result = verifier.verify(
        user_query="公司是否提供免费午餐？",
        answer="公司每天提供免费午餐。",
        retrieved_chunks=chunks,
    )

    assert result.answerable is True
    assert result.grounded is False
    assert result.cited_chunk_ids == []
    assert result.unsupported_claims == ["公司每天提供免费午餐。"]
    assert "无法根据当前知识库证据可靠回答" in result.suggested_revision


def test_empty_retrieval_is_unanswerable():
    verifier = EvidenceVerifier(min_top_score=0.5)

    result = verifier.verify(
        user_query="年假是多少？",
        answer="员工年假为10天。",
        retrieved_chunks=[],
    )

    assert result.answerable is False
    assert result.grounded is False
    assert result.cited_chunk_ids == []
    assert result.unsupported_claims == []


def test_low_score_retrieval_is_unanswerable():
    verifier = EvidenceVerifier(min_top_score=0.5)
    chunks = [
        {
            "chunk_id": "handbook#chunk_2",
            "text": "员工年假为10天。",
            "score": 0.21,
            "source": "handbook.pdf",
            "page": 4,
        }
    ]

    result = verifier.verify(
        user_query="年假是多少？",
        answer="员工年假为10天。",
        retrieved_chunks=chunks,
    )

    assert result.answerable is False
    assert result.grounded is False
    assert result.cited_chunk_ids == []
    assert result.unsupported_claims == []
    assert "低于阈值" in result.suggested_revision


def test_partial_unsupported_claim_keeps_supported_revision():
    verifier = EvidenceVerifier(min_top_score=0.5)
    chunks = [
        {
            "chunk_id": "handbook#chunk_3",
            "text": "员工年假为10天。",
            "score": 0.83,
            "source": "handbook.pdf",
            "page": 5,
        }
    ]

    result = verifier.verify(
        user_query="年假和午餐政策是什么？",
        answer="员工年假为10天。公司每天提供免费午餐。",
        retrieved_chunks=chunks,
    )

    assert result.answerable is True
    assert result.grounded is False
    assert result.cited_chunk_ids == ["handbook#chunk_3"]
    assert result.unsupported_claims == ["公司每天提供免费午餐。"]
    assert "员工年假为10天。" in result.suggested_revision
    assert "公司每天提供免费午餐。" not in result.suggested_revision
    assert "[source: handbook.pdf, page 5, handbook#chunk_3]" in result.suggested_revision
