from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.research_models import PaperCard, PaperElement, utc_now_iso
from core.research_store import ResearchStore


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.6
    source: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScholarFusionGraphStore:
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
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_node(self, node: GraphNode) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_nodes (node_id, node_type, name, metadata_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET node_type=excluded.node_type, name=excluded.name, metadata_json=excluded.metadata_json
                """,
                (node.node_id, node.node_type, node.name, json.dumps(node.metadata, ensure_ascii=False)),
            )

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO graph_edges (edge_id, source_id, target_id, relation, evidence_ids_json, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET evidence_ids_json=excluded.evidence_ids_json, confidence=excluded.confidence
                """,
                (edge.edge_id, edge.source_id, edge.target_id, edge.relation, json.dumps(edge.evidence_ids), edge.confidence, edge.source, utc_now_iso()),
            )

    def graph_for_paper(self, paper_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            nodes = [self._node(row) for row in conn.execute("SELECT * FROM graph_nodes WHERE metadata_json LIKE ?", (f"%{paper_id}%",)).fetchall()]
            node_ids = [node.node_id for node in nodes]
            edges = []
            if node_ids:
                placeholders = ",".join("?" for _ in node_ids)
                rows = conn.execute(f"SELECT * FROM graph_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", node_ids + node_ids).fetchall()
                edges = [self._edge(row) for row in rows]
        return {"nodes": [node.to_dict() for node in nodes], "edges": [edge.to_dict() for edge in edges]}

    def query(self, text: str) -> Dict[str, Any]:
        lowered = f"%{text.lower()}%"
        with self._connect() as conn:
            node_rows = conn.execute("SELECT * FROM graph_nodes WHERE lower(name) LIKE ? OR lower(node_type) LIKE ?", (lowered, lowered)).fetchall()
            nodes = [self._node(row) for row in node_rows]
        node_ids = [node.node_id for node in nodes]
        edges: List[GraphEdge] = []
        if node_ids:
            with self._connect() as conn:
                placeholders = ",".join("?" for _ in node_ids)
                edges = [self._edge(row) for row in conn.execute(f"SELECT * FROM graph_edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})", node_ids + node_ids).fetchall()]
        return {"nodes": [node.to_dict() for node in nodes], "edges": [edge.to_dict() for edge in edges]}

    def _node(self, row: sqlite3.Row) -> GraphNode:
        return GraphNode(row["node_id"], row["node_type"], row["name"], json.loads(row["metadata_json"]))

    def _edge(self, row: sqlite3.Row) -> GraphEdge:
        return GraphEdge(row["edge_id"], row["source_id"], row["target_id"], row["relation"], json.loads(row["evidence_ids_json"]), row["confidence"], row["source"])


def build_graph_from_store(research_store: ResearchStore, graph_store: ScholarFusionGraphStore) -> Dict[str, int]:
    node_count = 0
    edge_count = 0
    for paper in research_store.list_papers():
        paper_node = GraphNode(f"paper:{paper.paper_id}", "Paper", paper.metadata.title, {"paper_id": paper.paper_id})
        graph_store.upsert_node(paper_node)
        node_count += 1
        card = research_store.get_card(paper.paper_id, "manual") or research_store.get_card(paper.paper_id, "auto")
        if card:
            for field_name, node_type, relation in [
                ("representation", "Representation", "USES_REPRESENTATION"),
                ("body_model", "BodyModel", "USES_BODY_MODEL"),
                ("datasets", "Dataset", "EVALUATED_ON"),
                ("metrics", "Metric", "REPORTS_METRIC"),
                ("main_contributions", "Method", "PROPOSES"),
            ]:
                field = card.fields.get(field_name)
                value = str(field.value if field else "unknown")
                if not value or value == "unknown":
                    continue
                target = GraphNode(_node_id(node_type, value), node_type, value[:160], {"paper_id": paper.paper_id, "field": field_name})
                graph_store.upsert_node(target)
                node_count += 1
                edge = GraphEdge(
                    edge_id=_edge_id(paper_node.node_id, target.node_id, relation),
                    source_id=paper_node.node_id,
                    target_id=target.node_id,
                    relation=relation,
                    evidence_ids=[ref.element_id for ref in field.evidence_refs] if field else [],
                    confidence=field.confidence if field else 0.5,
                )
                graph_store.upsert_edge(edge)
                edge_count += 1
        for figure in research_store.list_elements(paper_id=paper.paper_id, element_types=["figure"]):
            if figure.metadata.get("is_duplicate") or figure.metadata.get("is_decorative"):
                continue
            fig_node = GraphNode(f"figure:{figure.element_id}", "Figure", figure.metadata.get("caption") or figure.element_id, {"paper_id": paper.paper_id, "element_id": figure.element_id})
            graph_store.upsert_node(fig_node)
            graph_store.upsert_edge(GraphEdge(_edge_id(paper_node.node_id, fig_node.node_id, "ILLUSTRATES"), paper_node.node_id, fig_node.node_id, "ILLUSTRATES", [figure.element_id], 0.6))
            node_count += 1
            edge_count += 1
    return {"node_count": node_count, "edge_count": edge_count}


def _node_id(node_type: str, name: str) -> str:
    digest = hashlib.sha1(f"{node_type}:{name.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"{node_type.lower()}:{digest}"


def _edge_id(source: str, target: str, relation: str) -> str:
    digest = hashlib.sha1(f"{source}:{target}:{relation}".encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"
