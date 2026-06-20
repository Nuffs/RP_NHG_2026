"""
Automated Failure Analysis Runner

Reads an answers JSON and runs all failure analysis metrics (retrieval,
generation, chunking) on every question. Crash-safe with resume support.

Usage:
    python failure_analysis/run_all.py <answers_json_path>
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
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
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai package not found.")
    print("Install it with:  pip install google-genai")
    sys.exit(1)

GEMINI_MODEL = "gemini-3.1-flash-lite"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_fa_root = Path(__file__).resolve().parent

missed_retrieval_mod  = _load_module("missed_retrieval",   _fa_root / "retrieval"  / "missed_retrieval.py")
relevance_mod         = _load_module("relevance",          _fa_root / "retrieval"  / "relevance.py")
fabricated_mod        = _load_module("fabricated_content",  _fa_root / "generation" / "fabricated_content.py")
incomplete_mod        = _load_module("incomplete_answer",   _fa_root / "generation" / "incomplete_answer.py")
misinterpretation_mod = _load_module("misinterpretation",  _fa_root / "generation" / "misinterpretation.py")
abstention_mod        = _load_module("abstention_failure", _fa_root / "generation" / "abstention_failure.py")
overchunking_mod      = _load_module("overchunking",       _fa_root / "chunking"   / "overchunking.py")
run_chunking_metrics  = _load_module("run_chunking_metrics", _fa_root / "run_chunking_metrics.py")


def format_retrieved_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        doc_id = chunk.get("doc_id", "unknown")
        parts.append(f"[Chunk {i}] (doc: {doc_id})\n{text}")
    return "\n\n".join(parts)


RATE_LIMIT_DELAY = 5


def safe_call(func, *args, retries: int = 3, **kwargs):
    """Call with retry on transient errors. Catches BaseException because
    individual scripts may call sys.exit(1) on parse errors (SystemExit)."""
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


def analyze_single_question(item: dict, client: genai.Client) -> dict:
    query_id = item.get("query_id", "unknown")
    query = item.get("query", "")
    gt_answer = item.get("gt_answer", "")
    response = item.get("response", "")
    retrieved_chunks = item.get("retrieved_context", [])

    retrieved_text = format_retrieved_context(retrieved_chunks)

    result = {
        "query_id": query_id,
        "query": query,
    }

    # Retrieval metrics
    print(f"    [1/7] Context Recall (E4)...", end=" ", flush=True)
    e4 = safe_call(
        missed_retrieval_mod.evaluate_context_recall_with_llm,
        retrieved_text, gt_answer, client
    )
    result["context_recall_E4"] = e4
    time.sleep(RATE_LIMIT_DELAY)

    print(f"    [2/7] Context Precision (E5/E6)...", end=" ", flush=True)
    e5e6 = safe_call(
        relevance_mod.evaluate_ranked_precision,
        retrieved_text, gt_answer, client
    )
    result["context_precision_E5E6"] = e5e6
    time.sleep(RATE_LIMIT_DELAY)

    # Generation metrics
    print(f"    [3/7] Faithfulness (E9)...", end=" ", flush=True)
    e9 = safe_call(
        fabricated_mod.evaluate_faithfulness_with_llm,
        retrieved_text, response, client, query
    )
    result["faithfulness_E9"] = e9
    time.sleep(RATE_LIMIT_DELAY)

    print(f"    [4/7] Answer Recall (E10)...", end=" ", flush=True)
    e10 = safe_call(
        incomplete_mod.evaluate_answer_recall_with_llm,
        response, gt_answer, client, retrieved_text
    )
    result["answer_recall_E10"] = e10
    time.sleep(RATE_LIMIT_DELAY)

    print(f"    [5/7] Misinterpretation (E11)...", end=" ", flush=True)
    e11 = safe_call(
        misinterpretation_mod.evaluate_misinterpretation_with_llm,
        retrieved_text, response, client
    )
    result["misinterpretation_E11"] = e11
    time.sleep(RATE_LIMIT_DELAY)

    # Chunking (deterministic)
    print(f"    [6/7] Overchunking (longest consecutive run)...", end=" ", flush=True)
    doc_ids = [c.get("doc_id", "") for c in retrieved_chunks]
    overchunking_result = overchunking_mod.compute_overchunking_score(doc_ids)
    consec_ratio = overchunking_result["overchunking_score"]
    result["overchunking"] = overchunking_result
    print("done")

    print(f"    [7/7] Abstention Failure (E12)...", end=" ", flush=True)
    e12 = safe_call(
        abstention_mod.evaluate_abstention_with_llm,
        retrieved_text, client, query
    )
    result["abstention_E12"] = e12
    time.sleep(RATE_LIMIT_DELAY)

    result["summary"] = {
        "context_recall":       e4.get("context_recall", None) if isinstance(e4, dict) else None,
        "ranked_precision":     e5e6.get("ranked_context_precision", None) if isinstance(e5e6, dict) else None,
        "unranked_precision":   e5e6.get("unranked_precision", None) if isinstance(e5e6, dict) else None,
        "faithfulness":         e9.get("faithfulness_score", None) if isinstance(e9, dict) else None,
        "answer_recall":        e10.get("answer_recall", None) if isinstance(e10, dict) else None,
        "misinterpretation_rate": e11.get("misinterpretation_rate", None) if isinstance(e11, dict) else None,
        "consecutive_chunk_ratio": consec_ratio,
        "abstention_score":       e12.get("abstention_score", None) if isinstance(e12, dict) else None,
    }

    return result


def _save_partial(
    results: list[dict],
    output_path: Path,
    model_name: str,
    input_path: Path,
    total: int,
) -> None:
    metric_keys = [
        "context_recall", "ranked_precision", "unranked_precision",
        "faithfulness", "answer_recall", "misinterpretation_rate",
        "consecutive_chunk_ratio", "abstention_score",
    ]
    agg: dict = {}
    for key in metric_keys:
        values = [
            r["summary"][key]
            for r in results
            if isinstance(r.get("summary", {}).get(key), (int, float))
        ]
        agg[key] = {
            "mean": sum(values) / len(values) if values else None,
            "count": len(values),
        }

    payload = {
        "model": model_name,
        "source_file": str(input_path),
        "total_questions": total,
        "completed_questions": len(results),
        "gemini_judge_model": GEMINI_MODEL,
        "aggregate_scores": {k: v["mean"] for k, v in agg.items()},
        "per_question_results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python failure_analysis/run_all.py <answers_json_path>")
        return 1

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1

    model_name = input_path.stem.replace("answers_", "")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("results", [])
    total = len(questions)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Automated Failure Analysis Runner                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Model:      {model_name}")
    print(f"  Questions:  {total}")
    print(f"  Input:      {input_path}")
    print(f"  LLM Judge:  {GEMINI_MODEL}")
    print()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found. Set it in pipeline/.env")
        return 1

    client = genai.Client(api_key=api_key)

    output_dir = _fa_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_failure_analysis.json"

    all_results: list[dict] = []
    done_ids: set[str] = set()

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            all_results = existing.get("per_question_results", [])
            done_ids = {r["query_id"] for r in all_results if "query_id" in r}
            if done_ids:
                print(f"  ▶  Resuming — {len(done_ids)} / {total} questions already done.")
                print()
        except Exception as exc:
            print(f"  ⚠️  Could not read existing output ({exc}). Starting fresh.")
            all_results = []
            done_ids = set()

    for i, item in enumerate(questions, 1):
        query_id = item.get("query_id", "unknown")

        if query_id in done_ids:
            print(f"  ⏭  [{i}/{total}] {query_id} — already done, skipping.")
            continue

        print(f"━━━ [{i}/{total}] {query_id} ━━━")

        result = analyze_single_question(item, client)
        all_results.append(result)

        _save_partial(all_results, output_path, model_name, input_path, total)

        s = result.get("summary", {})
        def _fmt(val):
            return f"{val:.4f}" if isinstance(val, (int, float)) else "ERR"
        print(f"    📊 Recall={_fmt(s.get('context_recall'))}  "
              f"Prec={_fmt(s.get('ranked_precision'))}  "
              f"Faith={_fmt(s.get('faithfulness'))}  "
              f"AnsRecall={_fmt(s.get('answer_recall'))}  "
              f"Misint={_fmt(s.get('misinterpretation_rate'))}  "
              f"Consec={_fmt(s.get('consecutive_chunk_ratio'))}  "
              f"Abst={_fmt(s.get('abstention_score'))}")
        print()

    print("=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)

    metric_keys = [
        ("context_recall",           "Context Recall (E4)"),
        ("ranked_precision",         "Ranked Precision (E6)"),
        ("unranked_precision",       "Unranked Precision (E5)"),
        ("faithfulness",             "Faithfulness (E9)"),
        ("answer_recall",            "Answer Recall (E10)"),
        ("misinterpretation_rate",   "Misinterpretation (E11)"),
        ("consecutive_chunk_ratio",  "Consecutive Chunk Ratio (Overchunking)"),
        ("abstention_score",         "Abstention Score (E12)"),
    ]

    aggregates = {}
    for key, label in metric_keys:
        values = [
            r["summary"][key]
            for r in all_results
            if isinstance(r.get("summary", {}).get(key), (int, float))
        ]
        if values:
            avg = sum(values) / len(values)
            aggregates[key] = {"mean": avg, "count": len(values)}
            print(f"  {label:.<40} {avg:.4f}  (n={len(values)})")
        else:
            aggregates[key] = {"mean": None, "count": 0}
            print(f"  {label:.<40} N/A")

    print("=" * 80)

    _save_partial(all_results, output_path, model_name, input_path, total)

    print(f"\n✅ Results saved to: {output_path}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Running Chunking Metrics (Relation Recall + Concept IoU)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    try:
        run_chunking_metrics.main()
    except BaseException as exc:
        print(f"\n  ⚠️  Chunking metrics failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
