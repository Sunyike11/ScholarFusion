from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings as AppSettings
from core.paper_card_llm import DashScopePaperCardBuilder, card_metrics
from core.research_store import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real LLM Paper Cards for a few papers.")
    parser.add_argument("--db-path", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion_phase2/scholarfusion.sqlite3"))
    parser.add_argument("--max-papers", type=int, default=3)
    parser.add_argument("--output", default=str(Path(AppSettings.RESEARCH_EVAL_RESULTS_DIR) / "scholarfusion_phase2/llm_paper_cards.json"))
    parser.add_argument("--real-provider", action="store_true")
    args = parser.parse_args()
    store = ResearchStore(args.db_path)
    papers = store.list_papers()[: args.max_papers]
    results = []
    if args.real_provider:
        builder = DashScopePaperCardBuilder(store)
        for paper in papers:
            result = builder.build(paper.paper_id)
            result["paper_id"] = paper.paper_id
            result["metrics"] = card_metrics(result["validated_card"])
            results.append(result)
    else:
        for paper in papers:
            card = store.get_card(paper.paper_id, "auto")
            raw = card.to_dict() if card else {}
            results.append({"paper_id": paper.paper_id, "mode": "rule_only", "validated_card": raw})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "ok",
        "paper_count": len(results),
        "real_provider": args.real_provider,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "paper_count": len(results), "real_provider": args.real_provider}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
