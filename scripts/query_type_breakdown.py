from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Phase3 query type breakdown markdown.")
    parser.add_argument("--ablation", default="evaluation_results/phase3/final_ablation.json")
    parser.add_argument("--dataset", default="evaluation_data/scholarfusion_model_reviewed_v1/provisional.jsonl")
    parser.add_argument("--output", default="docs/query_breakdown.md")
    args = parser.parse_args()
    data = json.loads(Path(args.ablation).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = defaultdict(int)
    for record in records:
        counts[record.get("query_type", "unknown")] += 1
    lines = ["# Query Type Breakdown", "", "Dataset: model-reviewed provisional, not human Gold.", "", "## Sample Counts", ""]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    lines.extend(["", "## Experiment Summary", "", "| Group | Recall@10 | MRR | False Answer | False Refusal | P95 Latency ms |", "|---|---:|---:|---:|---:|---:|"])
    for name, metrics in data.get("experiments", {}).items():
        lines.append(
            f"| {name} | {metrics.get('recall@10', 0):.4f} | {metrics.get('mrr', 0):.4f} | "
            f"{metrics.get('false_answer_rate', 0):.4f} | {metrics.get('false_refusal_rate', 0):.4f} | {metrics.get('p95_latency_ms', 0):.3f} |"
        )
    lines.extend([
        "",
        "## Findings",
        "",
        "- Multimodal dense is the strongest default retriever on the current dataset.",
        "- Page visual should stay query-gated because aggregate Recall@10 does not improve enough to justify default latency.",
        "- Graph retrieval is kept off by default until cross-paper evidence labels are stronger.",
        "- Evidence verifier calibration reduces false refusal when private/undisclosed refusal is handled separately.",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
