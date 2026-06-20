"""
Abstention Failure (E12) backfill runner.

Evaluates context sufficiency for queries missing abstention_E12 results.
Crash-safe: saves after every query.

Usage:
    python failure_analysis/run_abstention.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent / "pipeline" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai package not found.")
    print("Install it with:  pip install google-genai")
    sys.exit(1)

GEMINI_MODEL = "gemini-3.1-flash-lite"
RATE_LIMIT_DELAY = 5

_fa_root = Path(__file__).resolve().parent
_project_root = _fa_root.parent
_results_dir = _fa_root / "results"

FAILURE_ANALYSIS_PATH = _results_dir / "gpt-55_failure_analysis_factual.json"
ANSWERS_PATH = _project_root / "model_benchmarking" / "results" / "factual" / "results_full" / "answers_gpt-55.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abstention_mod = _load_module("abstention_failure", _fa_root / "generation" / "abstention_failure.py")


def safe_call(func, *args, retries: int = 3, **kwargs):
    RETRYABLE = ["429", "disconnected", "timed out", "deadline", "unavailable", "500", "503"]
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt,):
            raise
        except BaseException as exc:
            err_str = str(exc)
            is_retryable = any(s in err_str.lower() for s in RETRYABLE)
            if is_retryable and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"\n    ⏳ Transient error. Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"\n    ❌ Error in {func.__name__}: {err_str[:200]}")
                return {"error": err_str[:500]}
    return {"error": "Max retries exceeded"}


def format_retrieved_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        doc_id = chunk.get("doc_id", "unknown")
        parts.append(f"[Chunk {i}] (doc: {doc_id})\n{text}")
    return "\n\n".join(parts)


def _save(data: dict, path: Path) -> None:
    pq_results = data.get("per_question_results", [])
    metric_keys = [
        "context_recall", "ranked_precision", "unranked_precision",
        "faithfulness", "answer_recall", "misinterpretation_rate",
        "consecutive_chunk_ratio", "relation_recall", "concept_iou",
        "abstention_score",
    ]
    agg = {}
    for key in metric_keys:
        values = [
            r["summary"][key]
            for r in pq_results
            if isinstance(r.get("summary", {}).get(key), (int, float))
        ]
        agg[key] = sum(values) / len(values) if values else None

    data["aggregate_scores"] = agg

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Abstention Failure (E12) — Backfill Runner                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found. Set it in pipeline/.env")
        return 1

    client = genai.Client(api_key=api_key)

    print("Loading data...")
    with open(FAILURE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        fa_data = json.load(f)

    pq_results = fa_data.get("per_question_results", [])
    total = len(pq_results)
    print(f"  ✓ Failure analysis: {total} queries")

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    answers_lookup: dict[str, list[dict]] = {}
    for item in answers_data.get("results", []):
        qid = item.get("query_id")
        if qid:
            answers_lookup[qid] = item.get("retrieved_context", [])
    print(f"  ✓ Answers: {len(answers_lookup)} queries with retrieved context")

    pq_index: dict[str, int] = {}
    for idx, pq in enumerate(pq_results):
        pq_index[pq.get("query_id", "")] = idx

    already_done = sum(1 for pq in pq_results if "abstention_E12" in pq)
    remaining = total - already_done
    print(f"  ✓ Already done: {already_done} / {total}")
    print(f"  ✓ Remaining: {remaining}")
    print()

    if remaining == 0:
        print("✅ All queries already processed. Nothing to do.")
        return 0

    processed = 0
    errors = 0

    for i, pq_entry in enumerate(pq_results, 1):
        query_id = pq_entry.get("query_id", "unknown")

        if "abstention_E12" in pq_entry:
            continue

        query = pq_entry.get("query", "")
        retrieved_chunks = answers_lookup.get(query_id, [])
        retrieved_text = format_retrieved_context(retrieved_chunks)

        print(f"━━━ [{i}/{total}] {query_id} ━━━")
        print(f"    Question: {query[:80]}...")
        print(f"    Retrieved chunks: {len(retrieved_chunks)}")

        print(f"    Evaluating context sufficiency...", flush=True)
        result = safe_call(
            abstention_mod.evaluate_abstention_with_llm,
            retrieved_text, client, query
        )

        pq_entry["abstention_E12"] = result

        summary = pq_entry.get("summary", {})
        if isinstance(result, dict) and "abstention_score" in result:
            summary["abstention_score"] = result["abstention_score"]
        pq_entry["summary"] = summary

        _save(fa_data, FAILURE_ANALYSIS_PATH)

        if isinstance(result, dict):
            verdict = result.get("verdict", "?")
            score = result.get("abstention_score", "?")
            print(f"    📊 Verdict={verdict}  Score={score}")
        else:
            print(f"    ❌ Error")
            errors += 1

        processed += 1
        print()
        time.sleep(RATE_LIMIT_DELAY)

    print("=" * 70)
    print(f"  DONE — Processed {processed} queries ({errors} errors)")
    print("=" * 70)

    _save(fa_data, FAILURE_ANALYSIS_PATH)
    print(f"\n✅ Results saved to: {FAILURE_ANALYSIS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
