from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from config.settings import Settings as AppSettings


ImageInput = Union[str, Path, bytes]


class MultimodalEmbeddingError(RuntimeError):
    """Raised when a multimodal embedding request cannot produce a valid vector."""


@dataclass
class EmbeddingResult:
    vector: List[float]
    model: str
    modality: str
    latency_ms: float
    dimension: int
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector,
            "model": self.model,
            "modality": self.modality,
            "latency_ms": self.latency_ms,
            "dimension": self.dimension,
            "request_metadata": self.request_metadata,
        }


class MultimodalEmbeddingProvider(Protocol):
    model: str

    def encode_text(self, text: str) -> EmbeddingResult:
        ...

    def encode_image(self, image: ImageInput) -> EmbeddingResult:
        ...

    def encode_text_image_pair(self, text: str, image: ImageInput) -> EmbeddingResult:
        ...


def validate_embedding_vector(vector: List[float]) -> int:
    if not vector:
        raise MultimodalEmbeddingError("embedding vector is empty")
    for value in vector:
        if not isinstance(value, (int, float)):
            raise MultimodalEmbeddingError("embedding vector contains a non-numeric value")
        if math.isnan(float(value)) or math.isinf(float(value)):
            raise MultimodalEmbeddingError("embedding vector contains NaN or Inf")
    return len(vector)


class FakeMultimodalEmbeddingProvider:
    """Deterministic local provider used by unit tests and offline smoke runs."""

    def __init__(self, model: str = "fake-qwen3-vl-embedding", dimension: int = 16):
        self.model = model
        self.dimension = dimension
        self.request_count = 0
        self.last_request_metadata: Dict[str, Any] = {}

    def encode_text(self, text: str) -> EmbeddingResult:
        return self._result("text", text.encode("utf-8"), {"enable_fusion": False})

    def encode_image(self, image: ImageInput) -> EmbeddingResult:
        return self._result("image", self._image_bytes(image), {"enable_fusion": False})

    def encode_text_image_pair(self, text: str, image: ImageInput) -> EmbeddingResult:
        payload = text.encode("utf-8") + b"\0" + self._image_bytes(image)
        return self._result("text_image_pair", payload, {"enable_fusion": True})

    def _result(self, modality: str, payload: bytes, metadata: Dict[str, Any]) -> EmbeddingResult:
        start = time.perf_counter()
        self.request_count += 1
        digest = hashlib.sha256(payload or b"empty").digest()
        vector = []
        for index in range(self.dimension):
            value = digest[index % len(digest)] / 255.0
            vector.append(round(value * 2 - 1, 6))
        dimension = validate_embedding_vector(vector)
        latency_ms = (time.perf_counter() - start) * 1000
        self.last_request_metadata = dict(metadata)
        return EmbeddingResult(
            vector=vector,
            model=self.model,
            modality=modality,
            latency_ms=latency_ms,
            dimension=dimension,
            request_metadata=dict(metadata),
        )

    def _image_bytes(self, image: ImageInput) -> bytes:
        if isinstance(image, bytes):
            return image
        path_or_url = str(image)
        if path_or_url.startswith(("http://", "https://", "data:")):
            return path_or_url.encode("utf-8")
        path = Path(path_or_url)
        if not path.exists():
            return path_or_url.encode("utf-8")
        return path.read_bytes()


class DashScopeQwen3VLEmbeddingProvider:
    """DashScope qwen3-vl-embedding adapter for explicit real-provider smoke runs."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            endpoint: str = AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT,
            model: str = AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_MODEL,
            timeout: float = AppSettings.DASHSCOPE_REQUEST_TIMEOUT,
            dimension: Optional[int] = AppSettings.DASHSCOPE_MULTIMODAL_EMBEDDING_DIMENSION,
    ):
        self.api_key = AppSettings.API_KEY if api_key is None else api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.dimension = dimension
        self.request_count = 0
        self.last_request_metadata: Dict[str, Any] = {}
        if not self.api_key:
            raise MultimodalEmbeddingError("DASHSCOPE_API_KEY is missing")
        if not self.endpoint:
            raise MultimodalEmbeddingError("DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT is missing")
        if "compatible-mode" in self.endpoint:
            raise MultimodalEmbeddingError(
                "DASHSCOPE_MULTIMODAL_EMBEDDING_ENDPOINT must use DashScope multimodal embedding API, "
                "not an OpenAI-compatible /compatible-mode/v1 endpoint"
            )

    def encode_text(self, text: str) -> EmbeddingResult:
        if not text.strip():
            raise MultimodalEmbeddingError("text input is empty")
        payload = self._payload([{"text": text}], enable_fusion=False)
        return self._request(payload, "text")

    def encode_image(self, image: ImageInput) -> EmbeddingResult:
        payload = self._payload([{"image": self._image_reference(image)}], enable_fusion=False)
        return self._request(payload, "image")

    def encode_text_image_pair(self, text: str, image: ImageInput) -> EmbeddingResult:
        if not text.strip():
            raise MultimodalEmbeddingError("text input is empty")
        payload = self._payload(
            [{"text": text}, {"image": self._image_reference(image)}],
            enable_fusion=True,
        )
        return self._request(payload, "text_image_pair")

    def _payload(self, contents: List[Dict[str, str]], enable_fusion: bool) -> Dict[str, Any]:
        # DashScope multimodal embedding uses a dedicated endpoint. The fusion flag is
        # preserved here so the provider does not hand-combine text and image vectors.
        payload = {
            "model": self.model,
            "input": {"contents": contents},
            "parameters": {"enable_fusion": enable_fusion},
        }
        if self.dimension is not None:
            payload["parameters"]["dimension"] = self.dimension
        self.last_request_metadata = {
            "model": self.model,
            "endpoint_configured": bool(self.endpoint),
            "enable_fusion": enable_fusion,
            "dimension": self.dimension,
            "content_count": len(contents),
        }
        return payload

    def _request(self, payload: Dict[str, Any], modality: str) -> EmbeddingResult:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
            raise MultimodalEmbeddingError(f"DashScope embedding HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MultimodalEmbeddingError(f"DashScope embedding request failed: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        self.request_count += 1
        data = json.loads(raw)
        vector = self._extract_vector(data)
        dimension = validate_embedding_vector(vector)
        return EmbeddingResult(
            vector=vector,
            model=str(data.get("model") or self.model),
            modality=modality,
            latency_ms=latency_ms,
            dimension=dimension,
            request_metadata=dict(self.last_request_metadata),
        )

    def _extract_vector(self, data: Dict[str, Any]) -> List[float]:
        candidates = [
            data.get("output", {}).get("embeddings"),
            data.get("output", {}).get("embedding"),
            data.get("embeddings"),
            data.get("embedding"),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                first = candidate[0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    return [float(value) for value in first["embedding"]]
                if isinstance(first, (int, float)):
                    return [float(value) for value in candidate]
            if isinstance(candidate, dict) and isinstance(candidate.get("embedding"), list):
                return [float(value) for value in candidate["embedding"]]
        raise MultimodalEmbeddingError("DashScope response did not contain an embedding vector")

    def _image_reference(self, image: ImageInput) -> str:
        if isinstance(image, bytes):
            return self._data_uri(image, "image/png")
        value = str(image)
        if value.startswith(("http://", "https://", "data:")):
            return value
        path = Path(value)
        if not path.exists():
            raise MultimodalEmbeddingError(f"image file not found: {path}")
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        return self._data_uri(path.read_bytes(), mime_type)

    def _data_uri(self, image_bytes: bytes, mime_type: str) -> str:
        if not image_bytes:
            raise MultimodalEmbeddingError("image input is empty")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
