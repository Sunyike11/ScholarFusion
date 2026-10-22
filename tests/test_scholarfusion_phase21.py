import json

from core.scholarfusion_model_review import EvaluationItemReviewer, ModelReviewedDatasetValidator
from core.scholarfusion_phase21_metrics import answerability_metrics, citation_metrics, retrieval_metrics
from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_store import ResearchStore


def test_model_reviewer_never_marks_human_verified():
    reviewer = EvaluationItemReviewer()
    result = reviewer.review({
        "question": "What does the evidence say about Gaussian initialization?",
        "candidate_reference_answer": "The method initializes Gaussians on a surface.",
        "candidate_answerable": True,
        "candidate_primary_evidence": ["e1"],
        "candidate_alternative_evidence": [],
        "candidate_context_evidence": [],
    })
    data = result.to_dict()
    assert data["review_status"] == "model_reviewed"
    assert "human_verified" not in data


def test_validator_rejects_answerable_without_primary(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    paper = PaperDocument(
        paper_id="p1",
        metadata=PaperMetadata(title="Paper"),
        elements=[
            PaperElement(
                element_id="e1",
                paper_id="p1",
                element_type="paragraph",
                page_number=1,
                section_path=[],
                text="Gaussian initialization evidence.",
            )
        ],
    )
    store.upsert_paper(paper)
    result = ModelReviewedDatasetValidator(store).validate({
        "query_id": "q1",
        "paper_ids": ["p1"],
        "answerable": True,
        "gold_primary_element_ids": [],
        "acceptable_alternative_element_ids": [],
        "gold_context_element_ids": [],
    })
    assert result["validation_status"] == "invalid"
    assert "answerable_without_primary_evidence" in result["validation_errors"]


def test_phase21_strict_and_alternative_metrics():
    items = [{
        "query_id": "q1",
        "answerable": True,
        "gold_primary_element_ids": ["e1"],
        "acceptable_alternative_element_ids": ["e2"],
        "gold_context_element_ids": ["e3"],
    }]
    qa_outputs = {
        "q1": {
            "citations": [{"element_id": "e2", "page_number": 1}],
            "primary_citations": [{"element_id": "e2", "page_number": 1}],
            "context_citations": [{"element_id": "e3", "page_number": 1}],
            "context_manifest": {"claim_evidence": [{"support_status": "supported"}]},
        }
    }
    metrics = citation_metrics(items, qa_outputs)
    assert metrics["strict_metrics"]["primary_citation_precision"] == 0.0
    assert metrics["alternative_aware_metrics"]["alternative_aware_precision"] == 1.0
    assert metrics["alternative_aware_metrics"]["context_citation_precision"] == 1.0


def test_answerability_confusion_metrics():
    items = [
        {"query_id": "q1", "answerable": True, "query_type": "text_fact"},
        {"query_id": "q2", "answerable": False, "query_type": "unanswerable"},
    ]
    metrics = answerability_metrics(items, {"q1": True, "q2": True})
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert metrics["false_answer_rate"] == 1.0
