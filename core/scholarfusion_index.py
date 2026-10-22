from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.research_models import utc_now_iso


@dataclass
class IndexedVector:
    collection: str
    object_id: str
    paper_id: str
    element_id: Optional[str]
    modality: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["vector"] = []
        return data


@dataclass
class VectorSearchHit:
    collection: str
    object_id: str
    paper_id: str
    element_id: Optional[str]
    modality: str
    score: float
    metadata: Dict[str, Any]
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScholarFusionVectorIndex:
    """SQLite vector store with Chroma-compatible collection metadata."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    collection TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    element_id TEXT,
                    modality TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    source_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (collection, object_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_paper ON vectors(paper_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_modality ON vectors(modality)")

    def upsert(self, item: IndexedVector) -> None:
        source_hash = str(item.metadata.get("source_hash") or "")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vectors (collection, object_id, paper_id, element_id, modality, vector_json, metadata_json, text, source_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(collection, object_id) DO UPDATE SET
                    paper_id=excluded.paper_id,
                    element_id=excluded.element_id,
                    modality=excluded.modality,
                    vector_json=excluded.vector_json,
                    metadata_json=excluded.metadata_json,
                    text=excluded.text,
                    source_hash=excluded.source_hash,
                    created_at=excluded.created_at
                """,
                (
                    item.collection,
                    item.object_id,
                    item.paper_id,
                    item.element_id,
                    item.modality,
                    json.dumps(item.vector),
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.text,
                    source_hash,
                    utc_now_iso(),
                ),
            )

    def delete_paper(self, paper_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM vectors WHERE paper_id = ?", (paper_id,))

    def search(
            self,
            collection: str,
            query_vector: List[float],
            top_k: int = 10,
            target_modalities: Optional[List[str]] = None,
            paper_ids: Optional[List[str]] = None,
    ) -> List[VectorSearchHit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM vectors WHERE collection = ?", (collection,)).fetchall()
        hits: List[VectorSearchHit] = []
        modality_set = set(target_modalities or [])
        paper_set = set(paper_ids or [])
        for row in rows:
            if modality_set and row["modality"] not in modality_set:
                continue
            if paper_set and row["paper_id"] not in paper_set:
                continue
            vector = json.loads(row["vector_json"])
            score = cosine(query_vector, vector)
            metadata = json.loads(row["metadata_json"])
            hits.append(VectorSearchHit(
                collection=row["collection"],
                object_id=row["object_id"],
                paper_id=row["paper_id"],
                element_id=row["element_id"],
                modality=row["modality"],
                score=score,
                metadata=metadata,
                text=row["text"],
            ))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def count(self, collection: Optional[str] = None) -> int:
        with self._connect() as conn:
            if collection:
                row = conn.execute("SELECT COUNT(*) AS count FROM vectors WHERE collection = ?", (collection,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS count FROM vectors").fetchone()
        return int(row["count"])

    def list_vectors(self, collection: str) -> List[IndexedVector]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM vectors WHERE collection = ?", (collection,)).fetchall()
        return [
            IndexedVector(
                collection=row["collection"],
                object_id=row["object_id"],
                paper_id=row["paper_id"],
                element_id=row["element_id"],
                modality=row["modality"],
                vector=json.loads(row["vector_json"]),
                metadata=json.loads(row["metadata_json"]),
                text=row["text"],
            )
            for row in rows
        ]


def cosine(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    dim = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(dim))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(dim)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(dim)))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
