"""
Error Classification per Error Type (E3–E11 + Overchunking/Underchunking)

For each query in the failure analysis JSON, checks each metric against a
threshold to determine if that error type is present.

Usage:
    python failure_analysis/classify_errors.py
"""

import json
import sys
from pathlib import Path

_fa_root = Path(__file__).resolve().parent
FAILURE_ANALYSIS_PATH = _fa_root / "results" / "gpt-55_failure_analysis_clinical.json"
OUTPUT_PATH = _fa_root / "results" / "error_classification_clinical.json"


ERROR_DEFINITIONS = {
    "E3_context_mismatch": {
        "description": "Context Mismatch — chunk boundaries break semantic relationships needed for the answer",
        "metric_key": "relation_recall",
        "threshold": 0.8,
        "direction": "below",
    },
    "E4_missed_retrieval": {
        "description": "Missed Retrieval — relevant information not found in retrieved context",
        "metric_key": "context_recall",
        "threshold": 0.8,
        "direction": "below",
    },
    "E5E6_low_precision": {
        "description": "Low Context Precision — too many irrelevant chunks retrieved",
        "metric_key": "ranked_precision",
        "threshold": 0.3,
        "direction": "below",
    },
    "E9_hallucination": {
        "description": "Fabricated Content / Hallucination — response contains claims not in context or question",
        "metric_key": "faithfulness",
        "threshold": 0.8,
        "direction": "below",
    },
    "E10_incomplete_answer": {
        "description": "Incomplete Answer — response omits important information from the ground truth",
        "metric_key": "answer_recall",
        "threshold": 0.8,
        "direction": "below",
    },
    "E11_misinterpretation": {
        "description": "Misinterpretation — response distorts or misrepresents information",
        "metric_key": "misinterpretation_rate",
        "threshold": 0.2,
        "direction": "above",
    },
    "overchunking": {
        "description": "Overchunking — too many consecutive chunks retrieved (chunks should have been merged)",
        "metric_key": "consecutive_chunk_ratio",
        "threshold": 0.9,
        "direction": "above",
    },
    "underchunking": {
        "description": "Underchunking — low concept overlap between golden chunk and GT answer (chunk too large or irrelevant content)",
        "metric_key": "concept_iou",
        "threshold": 0.1,
        "direction": "below",
    },
    "abstention_failure": {
        "description": "Abstention Failure — model answered despite insufficient retrieved context",
        "metric_key": "abstention_score",
        "threshold": 0.5,
        "direction": "below",
    },
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Error Classification — Threshold-based Error Detection     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    with open(FAILURE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_question = data.get("per_question_results", [])
    total = len(per_question)
    print(f"Loaded {total} queries from {FAILURE_ANALYSIS_PATH.name}\n")

    print("Thresholds:")
    print("-" * 70)
    for eid, edef in ERROR_DEFINITIONS.items():
        op = "<" if edef["direction"] == "below" else ">"
        print(f"  {eid:.<35} {edef['metric_key']} {op} {edef['threshold']}")
    print("-" * 70)
    print()

    error_results: dict[str, dict] = {}
    for eid in ERROR_DEFINITIONS:
        error_results[eid] = {
            "description": ERROR_DEFINITIONS[eid]["description"],
            "metric_key": ERROR_DEFINITIONS[eid]["metric_key"],
            "threshold": ERROR_DEFINITIONS[eid]["threshold"],
            "direction": ERROR_DEFINITIONS[eid]["direction"],
            "count": 0,
            "query_ids": [],
        }

    per_query_errors: list[dict] = []
    no_data_counts: dict[str, int] = {eid: 0 for eid in ERROR_DEFINITIONS}

    for pq in per_question:
        query_id = pq.get("query_id", "unknown")
        summary = pq.get("summary", {})
        triggered: list[str] = []

        for eid, edef in ERROR_DEFINITIONS.items():
            value = summary.get(edef["metric_key"])

            if not isinstance(value, (int, float)):
                no_data_counts[eid] += 1
                continue

            is_error = False
            if edef["direction"] == "below" and value < edef["threshold"]:
                is_error = True
            elif edef["direction"] == "above" and value > edef["threshold"]:
                is_error = True

            if is_error:
                error_results[eid]["count"] += 1
                error_results[eid]["query_ids"].append(query_id)
                triggered.append(eid)

        per_query_errors.append({
            "query_id": query_id,
            "error_types": triggered,
            "error_count": len(triggered),
        })

    print("Results:")
    print("=" * 70)
    for eid, eres in error_results.items():
        count = eres["count"]
        pct = (count / total * 100) if total > 0 else 0
        no_data = no_data_counts[eid]
        evaluated = total - no_data
        pct_of_evaluated = (count / evaluated * 100) if evaluated > 0 else 0

        op = "<" if eres["direction"] == "below" else ">"
        print(f"\n  {eid}")
        print(f"    {eres['description']}")
        print(f"    Threshold: {eres['metric_key']} {op} {eres['threshold']}")
        print(f"    Errors:    {count} / {evaluated} evaluated ({pct_of_evaluated:.1f}%)")
        if no_data > 0:
            print(f"    No data:   {no_data} queries missing this metric")

    no_errors = sum(1 for pqe in per_query_errors if pqe["error_count"] == 0)
    at_least_one = sum(1 for pqe in per_query_errors if pqe["error_count"] > 0)
    multi_error = sum(1 for pqe in per_query_errors if pqe["error_count"] > 1)

    print(f"\n{'=' * 70}")
    print(f"  Total queries:              {total}")
    print(f"  No errors:                  {no_errors} ({no_errors/total*100:.1f}%)")
    print(f"  At least 1 error:           {at_least_one} ({at_least_one/total*100:.1f}%)")
    print(f"  Multiple errors (>1):       {multi_error} ({multi_error/total*100:.1f}%)")
    print(f"{'=' * 70}")

    output = {
        "source_file": str(FAILURE_ANALYSIS_PATH),
        "total_queries": total,
        "queries_with_no_errors": no_errors,
        "queries_with_errors": at_least_one,
        "queries_with_multiple_errors": multi_error,
        "error_types": {},
        "per_query": per_query_errors,
    }

    for eid, eres in error_results.items():
        output["error_types"][eid] = {
            "description": eres["description"],
            "metric_key": eres["metric_key"],
            "threshold": eres["threshold"],
            "direction": eres["direction"],
            "count": eres["count"],
            "percentage": round(eres["count"] / total * 100, 2) if total > 0 else 0,
            "query_ids": eres["query_ids"],
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
