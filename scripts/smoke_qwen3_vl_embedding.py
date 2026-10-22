from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_multimodal_embedding import (
    DashScopeQwen3VLEmbeddingProvider,
    FakeMultimodalEmbeddingProvider,
    MultimodalEmbeddingError,
)
from core.research_store import ResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test qwen3-vl-embedding provider.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--output-dir", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "qwen3_vl_embedding_smoke"))
    parser.add_argument("--max-texts", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--real-provider", action="store_true", help="Use DashScope network provider. Default uses fake provider.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = DashScopeQwen3VLEmbeddingProvider() if args.real_provider else FakeMultimodalEmbeddingProvider(
        model=AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL
    )
    store = ResearchStore(args.db_path)
    texts = collect_texts(store, args.max_texts)
    images = collect_images(store, args.max_images)

    request_count = 0
    failures: List[str] = []
    latencies: List[float] = []
    text_vectors: List[Tuple[str, List[float]]] = []
    image_vectors: List[Tuple[str, List[float]]] = []
    pair_vectors: List[Tuple[str, List[float]]] = []

    for element_id, text in texts:
        try:
            result = provider.encode_text(text)
            request_count += 1
            latencies.append(result.latency_ms)
            text_vectors.append((element_id, result.vector))
        except MultimodalEmbeddingError as exc:
            failures.append(f"text:{element_id}:{exc}")

    for figure_id, image_ref in images:
        try:
            result = provider.encode_image(image_ref)
            request_count += 1
            latencies.append(result.latency_ms)
            image_vectors.append((figure_id, result.vector))
        except MultimodalEmbeddingError as exc:
            failures.append(f"image:{figure_id}:{exc}")

    for (element_id, text), (figure_id, image_ref) in zip(texts[:5], images[:5]):
        try:
            result = provider.encode_text_image_pair(text, image_ref)
            request_count += 1
            latencies.append(result.latency_ms)
            pair_vectors.append((f"{element_id}+{figure_id}", result.vector))
        except MultimodalEmbeddingError as exc:
            failures.append(f"pair:{element_id}+{figure_id}:{exc}")

    all_vectors = [vector for _, vector in text_vectors + image_vectors + pair_vectors]
    vector_dimension = len(all_vectors[0]) if all_vectors else 0
    result = {
        "status": "ok" if all_vectors else "no_vectors",
        "provider": "dashscope" if args.real_provider else "fake",
        "model": provider.model,
        "vector_dimension": vector_dimension,
        "request_count": request_count,
        "success_count": len(all_vectors),
        "failure_count": len(failures),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "sample_rankings": sample_rankings(text_vectors, image_vectors, pair_vectors),
        "failure_summaries": failures[:10],
        "fusion_enabled_for_pair": bool(getattr(provider, "last_request_metadata", {}).get("enable_fusion")) if pair_vectors else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"ok", "no_vectors"} else 1


def collect_texts(store: ResearchStore, max_texts: int) -> List[Tuple[str, str]]:
    elements = store.list_elements(element_types=["abstract", "paragraph", "figure_caption"])
    return [(element.element_id, short_text(element.text, 500)) for element in elements if element.text.strip()][:max_texts]


def collect_images(store: ResearchStore, max_images: int) -> List[Tuple[str, str]]:
    figures = store.list_elements(element_types=["figure"])
    refs: List[Tuple[str, str]] = []
    for figure in figures:
        image_path = figure.metadata.get("image_path")
        image_url = figure.metadata.get("image_url")
        if image_path:
            refs.append((figure.element_id, str(image_path)))
        elif image_url:
            refs.append((figure.element_id, str(image_url)))
        elif figure.text:
            refs.append((figure.element_id, figure.text))
        if len(refs) >= max_images:
            break
    return refs


def sample_rankings(
        text_vectors: List[Tuple[str, List[float]]],
        image_vectors: List[Tuple[str, List[float]]],
        pair_vectors: List[Tuple[str, List[float]]],
) -> Dict[str, Any]:
    rankings: Dict[str, Any] = {}
    if len(text_vectors) >= 2:
        base_id, base_vec = text_vectors[0]
        rankings["text_to_text"] = rank_against(base_id, base_vec, text_vectors[1:4])
    if text_vectors and image_vectors:
        base_id, base_vec = text_vectors[0]
        rankings["text_to_figure"] = rank_against(base_id, base_vec, image_vectors[:5])
    if len(image_vectors) >= 2:
        base_id, base_vec = image_vectors[0]
        rankings["image_to_figure"] = rank_against(base_id, base_vec, image_vectors[1:5])
    if len(pair_vectors) >= 2:
        base_id, base_vec = pair_vectors[0]
        rankings["same_pair_vs_unrelated_pair"] = rank_against(base_id, base_vec, pair_vectors[1:4])
    return rankings


def rank_against(base_id: str, base_vec: List[float], candidates: List[Tuple[str, List[float]]]) -> List[Dict[str, Any]]:
    scored = [
        {"query_id": base_id, "candidate_id": candidate_id, "score": round(cosine(base_vec, vector), 6)}
        for candidate_id, vector in candidates
    ]
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def cosine(left: List[float], right: List[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 3)


def short_text(text: Any, max_len: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
