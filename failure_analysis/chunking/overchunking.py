"""
Overchunking Detection — Longest Consecutive Run

score = max_consecutive_run_length / 5

Usage:
    python failure_analysis/chunking/overchunking.py [answers_json_path]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "model_benchmarking" / "results" / "factual" / "answers_gpt-55.json"
)

RETRIEVAL_WINDOW = 5
DOC_ID_PATTERN = re.compile(r"^(.+)_(\d{4})$")


def parse_doc_id(doc_id: str) -> tuple[str, int] | None:
    m = DOC_ID_PATTERN.match(doc_id)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def find_longest_consecutive_run(doc_ids: list[str]) -> tuple[int, list[str]]:
    parsed = []
    for did in doc_ids:
        result = parse_doc_id(did)
        if result is not None:
            parsed.append((did, result[0], result[1]))

    if not parsed:
        return 0, []

    parsed.sort(key=lambda x: (x[1], x[2]))

    best_run: list[tuple[str, str, int]] = [parsed[0]]
    current_run: list[tuple[str, str, int]] = [parsed[0]]

    for i in range(1, len(parsed)):
        _, prev_prefix, prev_num = current_run[-1]
        _, cur_prefix, cur_num = parsed[i]

        if cur_prefix == prev_prefix and cur_num - prev_num == 1:
            current_run.append(parsed[i])
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [parsed[i]]

    if len(current_run) > len(best_run):
        best_run = current_run

    max_len = len(best_run)
    run_ids = [entry[0] for entry in best_run]

    return max_len, run_ids


def compute_overchunking_score(doc_ids: list[str]) -> dict:
    max_run, run_doc_ids = find_longest_consecutive_run(doc_ids)
    score = round(max_run / RETRIEVAL_WINDOW, 4) if RETRIEVAL_WINDOW else 0.0

    return {
        "doc_ids": doc_ids,
        "max_consecutive_run": max_run,
        "max_consecutive_run_doc_ids": run_doc_ids,
        "overchunking_score": score,
    }


def analyse(input_path: Path) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("results", [])
    total_questions = len(questions)

    per_question: list[dict] = []
    total_max_runs = 0
    questions_with_consecutive = 0

    for item in questions:
        query_id = item.get("query_id", "unknown")
        chunks = item.get("retrieved_context", [])
        doc_ids = [c.get("doc_id", "") for c in chunks]

        result = compute_overchunking_score(doc_ids)
        result["query_id"] = query_id

        if result["max_consecutive_run"] > 1:
            questions_with_consecutive += 1
        total_max_runs += result["max_consecutive_run"]

        per_question.append(result)

    scores = [pq["overchunking_score"] for pq in per_question]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    summary = {
        "input_file": str(input_path),
        "total_questions": total_questions,
        "questions_with_consecutive_chunks": questions_with_consecutive,
        "pct_questions_with_consecutive": (
            round(questions_with_consecutive / total_questions * 100, 2)
            if total_questions else 0.0
        ),
        "average_overchunking_score": round(avg_score, 4),
        "per_question": per_question,
    }
    return summary


WRAP = 80


def display(summary: dict) -> None:
    total_q = summary["total_questions"]
    q_consec = summary["questions_with_consecutive_chunks"]
    pct = summary["pct_questions_with_consecutive"]
    avg_score = summary["average_overchunking_score"]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Overchunking Detection — Longest Consecutive Run / 5      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Input:  {summary['input_file']}")
    print(f"  Total questions:  {total_q}")
    print()
    print("=" * WRAP)
    print(f"  {'Questions with ≥2 consecutive chunks':.<50} {q_consec} / {total_q}  ({pct}%)")
    print(f"  {'Average overchunking score':.<50} {avg_score:.4f}")
    print("=" * WRAP)

    print()
    print("-" * WRAP)
    print("  PER-QUESTION DETAIL  (only questions with consecutive chunks)")
    print("-" * WRAP)

    any_found = False
    for pq in summary["per_question"]:
        if pq["max_consecutive_run"] <= 1:
            continue
        any_found = True
        print(f"\n  📋  {pq['query_id']}")
        print(f"      Retrieved doc_ids: {pq['doc_ids']}")
        print(f"      Longest run ({pq['max_consecutive_run']}): {pq['max_consecutive_run_doc_ids']}")
        print(f"      Score: {pq['overchunking_score']}")

    if not any_found:
        print("\n  ✅  No consecutive chunks found in any question.")

    print()
    print("-" * WRAP)


def main() -> int:
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = DEFAULT_INPUT

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    summary = analyse(input_path)
    display(summary)

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = input_path.stem.replace("answers_", "")
    output_path = output_dir / f"{model_name}_overchunking.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"  💾  Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
