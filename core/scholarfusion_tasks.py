from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from core.research_models import utc_now_iso


class TaskStatusStore:
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
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL,
                    message TEXT NOT NULL,
                    paper_id TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def set_status(self, task_id: str, status: str, progress: float, message: str = "", paper_id: Optional[str] = None, summary: Optional[Dict[str, Any]] = None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute("SELECT created_at FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT INTO tasks (task_id, status, progress, message, paper_id, summary_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET status=excluded.status, progress=excluded.progress, message=excluded.message,
                    paper_id=COALESCE(excluded.paper_id, tasks.paper_id), summary_json=excluded.summary_json, updated_at=excluded.updated_at
                """,
                (task_id, status, progress, message, paper_id, json.dumps(summary or {}, ensure_ascii=False), created_at, now),
            )

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            return {"task_id": task_id, "status": "not_found", "message": "Task not found"}
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "progress": row["progress"],
            "message": row["message"],
            "paper_id": row["paper_id"],
            "summary": json.loads(row["summary_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
