"""
Run Relation Recall + Concept IoU for every query with golden chunks.

Crash-safe: saves after every query. Re-running skips already done.

Usage:
    python failure_analysis/run_chunking_metrics.py
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
_results_dir = _fa_root / "results"
GOLDEN_CHUNKS_PATH = _results_dir / "clinical_analysis_golden_chunks.json"
FAILURE_ANALYSIS_PATH = _results_dir / "gpt-55_failure_analysis_clinical.json"
ALLE_CHUNKS_PATH = _results_dir / "alle_chunks.json"
ANSWERS_PATH = _project_root / "model_benchmarking" / "results" / "clinical" / "results_full_200" / "answers_gpt-55.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

relation_recall_mod = _load_module("relation_recall", _fa_root / "chunking" / "relation_recall.py")
concept_iou_mod = _load_module("concept_iou", _fa_root / "chunking" / "concept_iou.py")

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


def build_chunk_with_headers(chunk: dict) -> str:
    section_path = chunk.get("section_path") or []
    text = chunk.get("text", "").strip()

    header_lines: list[str] = []
    for idx, header in enumerate(section_path, start=1):
        header_text = str(header).strip()
        if not header_text:
            continue
        header_lines.append(f"{'#' * idx} {header_text}")

    parts = header_lines.copy()
    if text:
        parts.append(text)

    return "\n\n".join(part for part in parts if part)


def build_golden_text(chunks: list[dict], include_headers: bool = True) -> str:
    parts = []
    for chunk in chunks:
        if include_headers:
            combined = build_chunk_with_headers(chunk)
        else:
            combined = chunk.get("text", "").strip()
        if combined:
            parts.append(combined)
    return "\n\n---\n\n".join(parts)


def format_golden_chunks_with_tags(chunks: list[dict], include_headers: bool = True) -> str:
    formatted = ""
    for i, chunk in enumerate(chunks, 1):
        if include_headers:
            text = build_chunk_with_headers(chunk)
        else:
            text = chunk.get("text", "").strip()
        formatted += f"<CHUNK_{i}>\n{text}\n</CHUNK_{i}>\n\n"
    return formatted.strip()


def format_golden_chunks_plain(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        parts.append(f"[Chunk {i}]\n{text}")
    return "\n\n".join(parts)


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  Chunking Metrics Runner — Relation Recall + Concept IoU    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found. Set it in pipeline/.env")
        return 1

    client = genai.Client(api_key=api_key)

    print("Loading data files...")

    with open(GOLDEN_CHUNKS_PATH, "r", encoding="utf-8") as f:
        golden_chunks_data = json.load(f)
    print(f"  ✓ Golden chunks: {len(golden_chunks_data)} queries")

    with open(FAILURE_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        failure_analysis = json.load(f)
    per_question = failure_analysis.get("per_question_results", [])
    print(f"  ✓ Failure analysis: {len(per_question)} queries")

    with open(ALLE_CHUNKS_PATH, "r", encoding="utf-8") as f:
        alle_chunks = json.load(f)
    chunk_lookup: dict[str, dict] = {}
    for chunk in alle_chunks:
        cid = chunk.get("chunk_id") or chunk.get("context_id")
        if cid:
            chunk_lookup[cid] = chunk
    print(f"  ✓ All chunks: {len(chunk_lookup)} chunks indexed")

    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    gt_answer_lookup: dict[str, str] = {}
    for item in answers_data.get("results", []):
        qid = item.get("query_id")
        if qid:
            gt_answer_lookup[qid] = item.get("gt_answer", "")
    print(f"  ✓ Answers: {len(gt_answer_lookup)} queries with GT answers")

    pq_index: dict[str, int] = {}
    for idx, pq in enumerate(per_question):
        pq_index[pq["query_id"]] = idx

    total = len(golden_chunks_data)
    processed = 0
    skipped = 0
    errors = 0

    print(f"\nProcessing {total} queries...\n")

    for i, gc_entry in enumerate(golden_chunks_data, 1):
        query_id = gc_entry["query_id"]
        golden_chunk_ids = gc_entry.get("golden_chunk_ids", [])

        if query_id not in pq_index:
            print(f"  ⚠️  [{i}/{total}] {query_id} — not found in failure analysis, skipping.")
            errors += 1
            continue

        pq_idx = pq_index[query_id]
        pq_entry = per_question[pq_idx]

        if "relation_recall_E3" in pq_entry and "concept_iou" in pq_entry:
            existing_rr = pq_entry["relation_recall_E3"]
            existing_iou = pq_entry["concept_iou"]
            if not isinstance(existing_rr, dict) or not isinstance(existing_iou, dict):
                pass
            elif "error" not in existing_rr and "error" not in existing_iou:
                print(f"  ⏭  [{i}/{total}] {query_id} — already done, skipping.")
                skipped += 1
                continue

        golden_chunks_resolved = []
        missing_chunks = []
        for cid in golden_chunk_ids:
            if cid in chunk_lookup:
                golden_chunks_resolved.append(chunk_lookup[cid])
            else:
                missing_chunks.append(cid)

        if missing_chunks:
            print(f"  ⚠️  [{i}/{total}] {query_id} — missing chunks: {missing_chunks}")

        if not golden_chunks_resolved:
            print(f"  ❌  [{i}/{total}] {query_id} — no golden chunks found, skipping.")
            errors += 1
            continue

        gt_answer = gt_answer_lookup.get(query_id, "")
        if not gt_answer:
            print(f"  ⚠️  [{i}/{total}] {query_id} — no GT answer found, skipping.")
            errors += 1
            continue

        query = pq_entry.get("query", "")

        golden_chunks_tagged = format_golden_chunks_with_tags(golden_chunks_resolved, include_headers=True)
        golden_chunks_plain = format_golden_chunks_plain(golden_chunks_resolved)

        print(f"━━━ [{i}/{total}] {query_id} ━━━")
        print(f"    Question: {query[:80]}...")
        print(f"    Golden chunks: {golden_chunk_ids}")
        print(f"    GT answer length: {len(gt_answer)} chars")

        print(f"    [1/2] Relation Recall (E3)...", flush=True)
        rr_result = safe_call(
            relation_recall_mod.evaluate_relation_recall_with_llm,
            query, golden_chunks_tagged, gt_answer, client
        )
        pq_entry["relation_recall_E3"] = rr_result
        time.sleep(RATE_LIMIT_DELAY)

        print(f"    [2/2] Concept IoU...", flush=True)
        iou_result = safe_call(
            concept_iou_mod.evaluate_iou_with_llm,
            golden_chunks_plain, gt_answer, client
        )
        pq_entry["concept_iou"] = iou_result
        time.sleep(RATE_LIMIT_DELAY)

        summary = pq_entry.get("summary", {})
        if isinstance(rr_result, dict) and "error" not in rr_result:
            summary["relation_recall"] = rr_result.get("relation_recall", None)
        if isinstance(iou_result, dict) and "error" not in iou_result:
            summary["concept_iou"] = iou_result.get("iou_score", None)
        pq_entry["summary"] = summary

        rr_score = rr_result.get("relation_recall", "ERR") if isinstance(rr_result, dict) else "ERR"
        iou_score = iou_result.get("iou_score", "ERR") if isinstance(iou_result, dict) else "ERR"
        rr_fmt = f"{rr_score:.4f}" if isinstance(rr_score, (int, float)) else rr_score
        iou_fmt = f"{iou_score:.4f}" if isinstance(iou_score, (int, float)) else iou_score
        print(f"    📊 RelationRecall={rr_fmt}  ConceptIoU={iou_fmt}")

        with open(FAILURE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
            json.dump(failure_analysis, f, ensure_ascii=False, indent=2)

        processed += 1
        print()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total queries:    {total}")
    print(f"  Processed:        {processed}")
    print(f"  Skipped (done):   {skipped}")
    print(f"  Errors/Missing:   {errors}")

    rr_values = []
    iou_values = []
    for pq in per_question:
        rr = pq.get("relation_recall_E3", {})
        if isinstance(rr, dict) and isinstance(rr.get("relation_recall"), (int, float)):
            rr_values.append(rr["relation_recall"])
        iou = pq.get("concept_iou", {})
        if isinstance(iou, dict) and isinstance(iou.get("iou_score"), (int, float)):
            iou_values.append(iou["iou_score"])

    if rr_values:
        avg_rr = sum(rr_values) / len(rr_values)
        print(f"\n  Avg Relation Recall:  {avg_rr:.4f}  (n={len(rr_values)})")
    if iou_values:
        avg_iou = sum(iou_values) / len(iou_values)
        print(f"  Avg Concept IoU:      {avg_iou:.4f}  (n={len(iou_values)})")

    with open(FAILURE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {FAILURE_ANALYSIS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
