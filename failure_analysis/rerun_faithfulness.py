"""
Rerun Faithfulness (E9) for all queries — now includes the QUESTION in the prompt.

Crash-safe: saves after every query. Re-running skips already updated entries.

Usage:
    python failure_analysis/rerun_faithfulness.py
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

_fa_root = Path(__file__).resolve().parent
_project_root = _fa_root.parent

FAILURE_ANALYSIS_PATH = _fa_root / "results" / "gpt-55_failure_analysis_clinical.json"
ANSWERS_PATH = _project_root / "model_benchmarking" / "results" / "clinical" / "results_full_200" / "answers_gpt-55.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

fabricated_mod = _load_module("fabricated_content", _fa_root / "generation" / "fabricated_content.py")

RATE_LIMIT_DELAY = 5

def safe_call(func, *args, retries: int = 3, **kwargs):
    RETRYABLE = ["429", "disconnected", "timed out", "deadline", "unavailable", "500", "503"]
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
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


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Rerun Faithfulness (E9) — Now with Question Context        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found. Set it in pipeline/.env")
        return 1

    client = genai.Client(api_key=api_key)

    print("Loading data files...")

    with open(FAILURE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        failure_analysis = json.load(f)
    per_question = failure_analysis.get("per_question_results", [])
    print(f"  ✓ Failure analysis: {len(per_question)} queries")

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    answers_lookup: dict[str, dict] = {}
    for item in answers_data.get("results", []):
        qid = item.get("query_id")
        if qid:
            answers_lookup[qid] = item
    print(f"  ✓ Answers: {len(answers_lookup)} queries")

    total = len(per_question)
    processed = 0
    skipped = 0
    errors = 0

    print(f"\nRerunning faithfulness for {total} queries...\n")

    for i, pq_entry in enumerate(per_question, 1):
        query_id = pq_entry.get("query_id", "unknown")
        query = pq_entry.get("query", "")

        ans_item = answers_lookup.get(query_id)
        if not ans_item:
            print(f"  ⚠️  [{i}/{total}] {query_id} — not in answers file, skipping.")
            errors += 1
            continue

        response = ans_item.get("response", "")
        retrieved_chunks = ans_item.get("retrieved_context", [])
        retrieved_text = format_retrieved_context(retrieved_chunks)

        if not response:
            print(f"  ⚠️  [{i}/{total}] {query_id} — no response, skipping.")
            errors += 1
            continue

        print(f"━━━ [{i}/{total}] {query_id} ━━━")
        print(f"    Question: {query[:80]}...")

        print(f"    Faithfulness (E9)...", flush=True)
        result = safe_call(
            fabricated_mod.evaluate_faithfulness_with_llm,
            retrieved_text, response, client, query
        )
        pq_entry["faithfulness_E9"] = result
        time.sleep(RATE_LIMIT_DELAY)

        summary = pq_entry.get("summary", {})
        if isinstance(result, dict) and "error" not in result:
            summary["faithfulness"] = result.get("faithfulness_score", None)
        pq_entry["summary"] = summary

        score = result.get("faithfulness_score", "ERR") if isinstance(result, dict) else "ERR"
        score_fmt = f"{score:.4f}" if isinstance(score, (int, float)) else score
        print(f"    📊 Faithfulness={score_fmt}")

        with open(FAILURE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, ensure_ascii=False, indent=2)

        processed += 1
        print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total queries:    {total}")
    print(f"  Processed:        {processed}")
    print(f"  Errors/Missing:   {errors}")

    values = []
    for pq in per_question:
        e9 = pq.get("faithfulness_E9", {})
        if isinstance(e9, dict) and isinstance(e9.get("faithfulness_score"), (int, float)):
            values.append(e9["faithfulness_score"])
    if values:
        avg = sum(values) / len(values)
        print(f"\n  Avg Faithfulness:  {avg:.4f}  (n={len(values)})")

    failure_analysis["aggregate_scores"]["faithfulness"] = avg if values else None

    with open(FAILURE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {FAILURE_ANALYSIS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
