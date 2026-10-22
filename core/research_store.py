from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.research_models import (
    PaperCard,
    PaperDocument,
    PaperElement,
    PaperMetadata,
    utc_now_iso,
)


class ResearchStore:
    """SQLite-backed store for structured research paper data."""

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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    markdown_debug TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    element_type TEXT NOT NULL,
                    page_number INTEGER,
                    section_path_json TEXT NOT NULL,
                    text TEXT NOT NULL,
                    bbox_json TEXT,
                    reading_order INTEGER NOT NULL,
                    parent_id TEXT,
                    linked_element_ids_json TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    parse_confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    representation_ids_json TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_elements_paper ON elements(paper_id);
                CREATE INDEX IF NOT EXISTS idx_elements_type ON elements(element_type);
                CREATE INDEX IF NOT EXISTS idx_elements_page ON elements(page_number);

                CREATE TABLE IF NOT EXISTS cards (
                    paper_id TEXT NOT NULL,
                    card_kind TEXT NOT NULL,
                    card_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (paper_id, card_kind),
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parser_status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_paper(self, paper: PaperDocument) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM papers WHERE paper_id = ?",
                (paper.paper_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else paper.created_at
            conn.execute(
                """
                INSERT INTO papers (paper_id, metadata_json, markdown_debug, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    markdown_debug = excluded.markdown_debug,
                    updated_at = excluded.updated_at
                """,
                (
                    paper.paper_id,
                    json.dumps(paper.metadata.to_dict(), ensure_ascii=False),
                    paper.markdown_debug,
                    created_at,
                    now,
                ),
            )
            conn.execute("DELETE FROM elements WHERE paper_id = ?", (paper.paper_id,))
            self._insert_elements(conn, paper.elements)

    def _insert_elements(self, conn: sqlite3.Connection, elements: Iterable[PaperElement]) -> None:
        for element in elements:
            conn.execute(
                """
                INSERT INTO elements (
                    element_id, paper_id, element_type, page_number, section_path_json,
                    text, bbox_json, reading_order, parent_id, linked_element_ids_json,
                    source_file, parser_name, parse_confidence, metadata_json,
                    representation_ids_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    element.element_id,
                    element.paper_id,
                    element.element_type,
                    element.page_number,
                    json.dumps(element.section_path, ensure_ascii=False),
                    element.text,
                    json.dumps(element.bbox, ensure_ascii=False) if element.bbox is not None else None,
                    element.reading_order,
                    element.parent_id,
                    json.dumps(element.linked_element_ids, ensure_ascii=False),
                    element.source_file,
                    element.parser_name,
                    element.parse_confidence,
                    json.dumps(element.metadata, ensure_ascii=False),
                    json.dumps(element.representation_ids, ensure_ascii=False),
                ),
            )

    def get_paper(self, paper_id: str, include_elements: bool = True) -> Optional[PaperDocument]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
            if not row:
                return None
            elements = self.list_elements(paper_id=paper_id) if include_elements else []
            return PaperDocument(
                paper_id=row["paper_id"],
                metadata=PaperMetadata.from_dict(json.loads(row["metadata_json"])),
                elements=elements,
                markdown_debug=row["markdown_debug"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_papers(self) -> List[PaperDocument]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY updated_at DESC").fetchall()
        return [
            PaperDocument(
                paper_id=row["paper_id"],
                metadata=PaperMetadata.from_dict(json.loads(row["metadata_json"])),
                elements=[],
                markdown_debug=row["markdown_debug"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def list_elements(
            self,
            paper_id: Optional[str] = None,
            element_types: Optional[List[str]] = None,
            section_contains: Optional[str] = None,
    ) -> List[PaperElement]:
        clauses: List[str] = []
        params: List[Any] = []
        if paper_id:
            clauses.append("paper_id = ?")
            params.append(paper_id)
        if element_types:
            placeholders = ",".join("?" for _ in element_types)
            clauses.append(f"element_type IN ({placeholders})")
            params.extend(element_types)
        query = "SELECT * FROM elements"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY paper_id, page_number, reading_order"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        elements = [self._row_to_element(row) for row in rows]
        if section_contains:
            needle = section_contains.lower()
            elements = [
                element for element in elements
                if needle in " / ".join(element.section_path).lower()
            ]
        return elements

    def get_element(self, element_id: str) -> Optional[PaperElement]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM elements WHERE element_id = ?", (element_id,)).fetchone()
        return self._row_to_element(row) if row else None

    def element_exists(self, paper_id: str, element_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM elements WHERE paper_id = ? AND element_id = ?",
                (paper_id, element_id),
            ).fetchone()
        return bool(row)

    def upsert_card(self, card: PaperCard, card_kind: str = "auto") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cards (paper_id, card_kind, card_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(paper_id, card_kind) DO UPDATE SET
                    card_json = excluded.card_json,
                    updated_at = excluded.updated_at
                """,
                (
                    card.paper_id,
                    card_kind,
                    json.dumps(card.to_dict(), ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def get_card(self, paper_id: str, card_kind: str = "auto") -> Optional[PaperCard]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT card_json FROM cards WHERE paper_id = ? AND card_kind = ?",
                (paper_id, card_kind),
            ).fetchone()
        return PaperCard.from_dict(json.loads(row["card_json"])) if row else None

    def delete_paper(self, paper_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))

    def record_ingestion_run(self, run_id: str, paper_id: str, status: str, parser_status: str, metrics: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_runs
                (run_id, paper_id, status, parser_status, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    paper_id,
                    status,
                    parser_status,
                    json.dumps(metrics, ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def _row_to_element(self, row: sqlite3.Row) -> PaperElement:
        return PaperElement(
            element_id=row["element_id"],
            paper_id=row["paper_id"],
            element_type=row["element_type"],
            page_number=row["page_number"],
            section_path=json.loads(row["section_path_json"]),
            text=row["text"],
            bbox=json.loads(row["bbox_json"]) if row["bbox_json"] else None,
            reading_order=row["reading_order"],
            parent_id=row["parent_id"],
            linked_element_ids=json.loads(row["linked_element_ids_json"]),
            source_file=row["source_file"],
            parser_name=row["parser_name"],
            parse_confidence=row["parse_confidence"],
            metadata=json.loads(row["metadata_json"]),
            representation_ids=json.loads(row["representation_ids_json"]),
        )
