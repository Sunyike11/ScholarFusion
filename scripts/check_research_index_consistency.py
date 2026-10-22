from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.research_store import ResearchStore
from core.scholarfusion_index import ScholarFusionVectorIndex
from core.scholarfusion_graph import ScholarFusionGraphStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Research/ScholarFusion index consistency.")
    parser.add_argument("--db-path", default=AppSettings.RESEARCH_DB_PATH)
    parser.add_argument("--index-db-path", default=AppSettings.SCHOLARFUSION_INDEX_DB_PATH)
    parser.add_argument("--output", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion/index_consistency.json"))
    args = parser.parse_args()
    store = ResearchStore(args.db_path)
    index = ScholarFusionVectorIndex(args.index_db_path)
    result = check(store, index, args.db_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def check(store: ResearchStore, index: ScholarFusionVectorIndex, db_path: str) -> Dict[str, Any]:
    element_ids = {element.element_id for element in store.list_elements()}
    paper_ids = {paper.paper_id for paper in store.list_papers()}
    orphan_vectors = []
    missing_elements = []
    duplicate_ids = []
    stale_pages = []
    with index._connect() as conn:
        rows = conn.execute("SELECT * FROM vectors").fetchall()
    seen = set()
    for row in rows:
        key = (row["collection"], row["object_id"])
        if key in seen:
            duplicate_ids.append(":".join(key))
        seen.add(key)
        if row["paper_id"] not in paper_ids:
            orphan_vectors.append(row["object_id"])
        if row["element_id"] and row["element_id"] not in element_ids:
            missing_elements.append(row["element_id"])
        metadata = json.loads(row["metadata_json"])
        page_image = metadata.get("page_image_path")
        if page_image and not str(page_image).startswith("page_proxy:") and not Path(page_image).exists():
            stale_pages.append(row["object_id"])
    graph_missing = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            edge_rows = conn.execute("SELECT evidence_ids_json FROM graph_edges").fetchall()
        for row in edge_rows:
            for evidence_id in json.loads(row["evidence_ids_json"]):
                if evidence_id not in element_ids:
                    graph_missing.append(evidence_id)
    except Exception:
        pass
    return {
        "paper_count": len(paper_ids),
        "element_count": len(element_ids),
        "vector_count": len(rows),
        "orphan_vector_count": len(orphan_vectors),
        "missing_element_reference_count": len(missing_elements),
        "duplicate_vector_id_count": len(duplicate_ids),
        "stale_page_image_count": len(stale_pages),
        "graph_missing_evidence_count": len(graph_missing),
        "orphan_vectors": orphan_vectors[:20],
        "missing_elements": missing_elements[:20],
        "stale_pages": stale_pages[:20],
    }


if __name__ == "__main__":
    raise SystemExit(main())
