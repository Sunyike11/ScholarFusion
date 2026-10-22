import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.research_models import PaperDocument, PaperElement, PaperMetadata
from core.research_multimodal_embedding import (
    DashScopeQwen3VLEmbeddingProvider,
    FakeMultimodalEmbeddingProvider,
    MultimodalEmbeddingError,
    validate_embedding_vector,
)
from core.research_store import ResearchStore
from scripts.build_research_eval_candidates import build_candidates


def test_real_smoke_no_pdf_files(tmp_path):
    pdf_dir = tmp_path / "missing_papers"
    output_dir = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_research_mvp_real.py",
            "--pdf-dir",
            str(pdf_dir),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "no_pdf_files"
    assert payload["paper_count"] == 0
    assert (output_dir / "summary.json").exists()


def test_eval_candidate_builder_pending_review_schema(tmp_path):
    store = ResearchStore(str(tmp_path / "research.sqlite3"))
    paper = PaperDocument(
        paper_id="paper_eval",
        metadata=PaperMetadata(title="Eval Paper", source_file="eval.pdf"),
        elements=[
            PaperElement(
                element_id="paper_eval_title_1",
                paper_id="paper_eval",
                element_type="title",
                page_number=1,
                section_path=[],
                text="Gaussian Human Reconstruction",
                reading_order=0,
            ),
            PaperElement(
                element_id="paper_eval_abs_1",
                paper_id="paper_eval",
                element_type="abstract",
                page_number=1,
                section_path=["Abstract"],
                text="We propose a Gaussian human reconstruction method.",
                reading_order=1,
            ),
            PaperElement(
                element_id="paper_eval_cap_1",
                paper_id="paper_eval",
                element_type="figure_caption",
                page_number=2,
                section_path=["Method"],
                text="Fig. 1: Overview of the framework.",
                reading_order=2,
                metadata={"figure_label": "1"},
            ),
            PaperElement(
                element_id="paper_eval_fig_1",
                paper_id="paper_eval",
                element_type="figure",
                page_number=2,
                section_path=["Method"],
                text="Fig. 1: Overview of the framework.",
                reading_order=3,
                linked_element_ids=["paper_eval_cap_1"],
                metadata={"caption": "Fig. 1: Overview of the framework.", "caption_element_id": "paper_eval_cap_1"},
            ),
        ],
    )
    store.upsert_paper(paper)

    records = build_candidates(store, max_per_paper=6)

    assert records
    assert all(record["review_status"] == "pending_review" for record in records)
    assert not any(record["review_status"] == "human_verified" for record in records)
    for record in records:
        if record["answerable"]:
            assert record["relevant_element_ids"]
        else:
            assert record["relevant_element_ids"] == []


def test_fake_multimodal_provider_and_fusion_flag(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    provider = FakeMultimodalEmbeddingProvider(dimension=8)

    text_result = provider.encode_text("Gaussian initialization")
    image_result = provider.encode_image(image)
    pair_result = provider.encode_text_image_pair("Gaussian initialization", image)

    assert text_result.dimension == 8
    assert image_result.dimension == 8
    assert pair_result.dimension == 8
    assert pair_result.request_metadata["enable_fusion"] is True
    assert provider.request_count == 3


def test_embedding_vector_validation_rejects_empty_nan_and_inf():
    with pytest.raises(MultimodalEmbeddingError):
        validate_embedding_vector([])
    with pytest.raises(MultimodalEmbeddingError):
        validate_embedding_vector([0.1, float("nan")])
    with pytest.raises(MultimodalEmbeddingError):
        validate_embedding_vector([0.1, float("inf")])
    assert validate_embedding_vector([0.1, -0.2]) == 2


def test_dashscope_provider_requires_key_and_keeps_fusion_metadata(tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    with pytest.raises(MultimodalEmbeddingError):
        DashScopeQwen3VLEmbeddingProvider(api_key="", endpoint="https://example.invalid")
    with pytest.raises(MultimodalEmbeddingError):
        DashScopeQwen3VLEmbeddingProvider(api_key="not-a-real-key", endpoint="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

    provider = DashScopeQwen3VLEmbeddingProvider(api_key="not-a-real-key", endpoint="https://example.invalid", dimension=256)
    payload = provider._payload([{"text": "a"}, {"image": "data:image/png;base64,AA=="}], enable_fusion=True)
    assert payload["model"] == provider.model
    assert payload["parameters"]["enable_fusion"] is True
    assert payload["parameters"]["dimension"] == 256
    assert provider.last_request_metadata["enable_fusion"] is True


@pytest.mark.real_provider
def test_real_provider_marker_is_explicit():
    pytest.skip("Real DashScope provider tests require explicit local opt-in.")
