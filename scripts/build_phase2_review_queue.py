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


TARGETS = {
    "text_fact": 6,
    "method_detail": 3,
    "text_to_figure": 5,
    "image_to_text": 4,
    "image_to_image": 3,
    "pair_to_any": 3,
    "cross_paper_comparison": 4,
    "unanswerable": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Phase2 human review queue from provisional candidates.")
    parser.add_argument("--candidates", default="evaluation_data/scholarfusion_lite_v1/candidates.jsonl")
    parser.add_argument("--output", default="evaluation_data/scholarfusion_lite_v1/phase2_review_queue.csv")
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.candidates).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    counts: Dict[str, int] = {}
    for record in records:
        qtype = record["query_type"]
        if counts.get(qtype, 0) >= TARGETS.get(qtype, 0):
            continue
        selected.append(record)
        counts[qtype] = counts.get(qtype, 0) + 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "query_type", "question", "reference_answer", "relevant_element_ids", "relevant_page_numbers", "answerable", "review_decision", "review_notes"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "question": item["question"],
                "reference_answer": item.get("reference_answer_candidate", ""),
                "relevant_element_ids": ";".join(item.get("relevant_element_ids", [])),
                "relevant_page_numbers": ";".join(str(page) for page in item.get("relevant_page_numbers", [])),
                "answerable": item.get("answerable"),
                "review_decision": "",
                "review_notes": "",
            })
    print(json.dumps({"status": "ok", "review_queue_count": len(selected), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
