from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence


def retrieval_metrics(items: Sequence[Dict[str, Any]], retrieved: Dict[str, List[str]], k_values: Sequence[int] = (5, 10)) -> Dict[str, float]:
    import math

    out: Dict[str, float] = {}
    for k in k_values:
        recalls = []
        ndcgs = []
        for item in items:
            relevant = set(item.get("gold_primary_element_ids") or item.get("relevant_element_ids") or [])
            if not relevant:
                continue
            top = retrieved.get(item["query_id"], [])[:k]
            hits = [eid for eid in top if eid in relevant]
            recalls.append(len(set(hits)) / len(relevant))
            dcg = sum((1.0 if eid in relevant else 0.0) / math.log2(index + 1) for index, eid in enumerate(top, start=1))
            ideal = sum(1.0 / math.log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
            ndcgs.append(dcg / ideal if ideal else 0.0)
        out[f"recall@{k}"] = mean(recalls)
        out[f"ndcg@{k}"] = mean(ndcgs)
    rr = []
    for item in items:
        relevant = set(item.get("gold_primary_element_ids") or item.get("relevant_element_ids") or [])
        rank = 0
        for index, eid in enumerate(retrieved.get(item["query_id"], []), start=1):
            if eid in relevant:
                rank = index
                break
        rr.append(1 / rank if rank else 0.0)
    out["mrr"] = mean(rr)
    return out


def citation_metrics(items: Sequence[Dict[str, Any]], qa_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    strict_p = []
    strict_r = []
    alt_p = []
    alt_r = []
    ctx_p = []
    id_valid = []
    page_acc = []
    support_rates = []
    unsupported_rates = []
    for item in items:
        qa = qa_outputs.get(item["query_id"], {})
        primary_gold = set(item.get("gold_primary_element_ids") or [])
        alternative = set(item.get("acceptable_alternative_element_ids") or [])
        context_gold = set(item.get("gold_context_element_ids") or [])
        primary_pred = set(_ids(qa.get("primary_citations") or []))
        context_pred = set(_ids(qa.get("context_citations") or []))
        all_pred = primary_pred | context_pred
        if all_pred:
            id_valid.append(1.0)
        else:
            id_valid.append(0.0 if item.get("answerable") else 1.0)
        page_acc.extend([1.0 if cite.get("page_number") is not None else 0.0 for cite in (qa.get("citations") or [])])
        if primary_gold:
            strict_p.append(len(primary_pred & primary_gold) / max(len(primary_pred), 1))
            strict_r.append(len(primary_pred & primary_gold) / len(primary_gold))
            alt_gold = primary_gold | alternative
            alt_p.append(len(primary_pred & alt_gold) / max(len(primary_pred), 1))
            alt_r.append(len(primary_pred & alt_gold) / len(alt_gold))
        if context_gold:
            ctx_p.append(len(context_pred & context_gold) / max(len(context_pred), 1))
        claims = qa.get("context_manifest", {}).get("claim_evidence", [])
        if claims:
            supported = len([c for c in claims if c.get("support_status") == "supported"])
            unsupported = len([c for c in claims if c.get("support_status") == "unsupported"])
            support_rates.append(supported / len(claims))
            unsupported_rates.append(unsupported / len(claims))
    return {
        "strict_metrics": {
            "citation_id_validity": mean(id_valid),
            "citation_page_accuracy": mean(page_acc),
            "primary_citation_precision": mean(strict_p),
            "primary_citation_recall": mean(strict_r),
            "claim_support_precision": mean(support_rates),
            "claim_support_recall": mean(support_rates),
            "unsupported_claim_rate": mean(unsupported_rates),
        },
        "alternative_aware_metrics": {
            "alternative_aware_precision": mean(alt_p),
            "alternative_aware_recall": mean(alt_r),
            "context_citation_precision": mean(ctx_p),
        },
    }


def answerability_metrics(items: Sequence[Dict[str, Any]], predictions: Dict[str, bool]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    by_type: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        gold = bool(item.get("answerable"))
        pred = bool(predictions.get(item["query_id"], False))
        qtype = item.get("query_type", "unknown")
        if gold and pred:
            tp += 1
            by_type[qtype]["tp"] += 1
        elif not gold and pred:
            fp += 1
            by_type[qtype]["fp"] += 1
        elif not gold and not pred:
            tn += 1
            by_type[qtype]["tn"] += 1
        else:
            fn += 1
            by_type[qtype]["fn"] += 1
    answerable_precision = tp / max(tp + fp, 1)
    answerable_recall = tp / max(tp + fn, 1)
    unanswerable_precision = tn / max(tn + fn, 1)
    unanswerable_recall = tn / max(tn + fp, 1)
    return {
        "answerable_precision": answerable_precision,
        "answerable_recall": answerable_recall,
        "unanswerable_precision": unanswerable_precision,
        "unanswerable_recall": unanswerable_recall,
        "balanced_accuracy": (answerable_recall + unanswerable_recall) / 2,
        "false_answer_rate": fp / max(fp + tn, 1),
        "false_refusal_rate": fn / max(fn + tp, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_type": {key: dict(value) for key, value in by_type.items()},
    }


def group_metrics(items: Sequence[Dict[str, Any]], retrieved: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    groups = {
        "text_only": {"text_fact", "method_detail"},
        "figure_related": {"text_to_figure"},
        "image_query": {"image_to_text", "image_to_image"},
        "pair_query": {"pair_to_any"},
        "cross_paper": {"cross_paper_comparison"},
        "unanswerable": {"unanswerable"},
    }
    out = {}
    for name, qtypes in groups.items():
        subset = [item for item in items if item.get("query_type") in qtypes]
        out[name] = retrieval_metrics(subset, retrieved, (5, 10)) if subset else {"sample_count": 0}
        out[name]["sample_count"] = len(subset)
    return out


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _ids(citations: List[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(str(cite.get("element_id")) for cite in citations if cite.get("element_id")))
