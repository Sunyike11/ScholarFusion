from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import human review decisions into candidate JSONL.")
    parser.add_argument("--candidates", default="evaluation_data/scholarfusion_lite_v1/candidates.jsonl")
    parser.add_argument("--review-csv", default="evaluation_data/scholarfusion_lite_v1/phase2_review_queue.csv")
    parser.add_argument("--output", default="evaluation_data/scholarfusion_lite_v1/candidates_with_human_review.jsonl")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.candidates).read_text(encoding="utf-8").splitlines() if line.strip()]
    reviews: Dict[str, Dict[str, str]] = {}
    with Path(args.review_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("review_decision", "").lower() in {"approve", "approved", "yes", "human_verified"}:
                reviews[row["query_id"]] = row
    verified = 0
    for record in records:
        review = reviews.get(record["query_id"])
        if not review:
            continue
        record["review_status"] = "human_verified"
        record["reference_answer_candidate"] = review.get("reference_answer") or record.get("reference_answer_candidate", "")
        record["relevant_element_ids"] = [item for item in review.get("relevant_element_ids", "").split(";") if item]
        record["relevant_page_numbers"] = [int(item) for item in review.get("relevant_page_numbers", "").split(";") if item]
        record["answerable"] = str(review.get("answerable", record.get("answerable"))).lower() in {"true", "1", "yes"}
        record["review_notes"] = review.get("review_notes", "")
        verified += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "human_verified_count": verified, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
