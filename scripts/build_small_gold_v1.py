from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


APPROVED = {"approve", "approved", "yes", "y", "true", "pass", "human_verified"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build small_gold_v1 only from human-approved rows.")
    parser.add_argument("--model-reviewed", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--spot-check", default="evaluation_data/scholarfusion_model_reviewed_v1/spot_check.csv")
    parser.add_argument("--output-dir", default="evaluation_data/small_gold_v1")
    args = parser.parse_args()
    records = {row["query_id"]: row for row in load_jsonl(Path(args.model_reviewed))}
    approved_ids = approved_from_csv(Path(args.spot_check))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = []
    candidates = []
    for query_id, record in records.items():
        if query_id in approved_ids:
            item = dict(record)
            item["human_verified"] = True
            item["review_status"] = "human_verified"
            item["dataset_status"] = "gold"
            item["not_gold"] = False
            gold.append(item)
        else:
            candidates.append(record | {"human_verified": False, "review_status": "pending_human_review"})
    write_jsonl(output_dir / "gold_dataset.jsonl", gold)
    write_jsonl(output_dir / "candidates_pending_human.jsonl", candidates)
    report = {
        "status": "ok" if gold else "blocked_no_human_approved_rows",
        "sample_count": len(gold),
        "candidate_count": len(candidates),
        "human_verified": bool(gold),
        "note": "Rows are marked human_verified only when spot_check_status is explicitly approved.",
    }
    (output_dir / "small_gold_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def approved_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    approved = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = (row.get("spot_check_status") or row.get("review_decision") or "").strip().lower()
            if status in APPROVED:
                approved.add(row["query_id"])
    return approved


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
