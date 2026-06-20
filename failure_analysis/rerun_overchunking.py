"""
Recompute Overchunking Scores

Patches existing failure analysis JSONs with: overchunking_score = max_run / 5

Usage:
    python failure_analysis/rerun_overchunking.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_fa_root = Path(__file__).resolve().parent

FILES_TO_PATCH = [
    _fa_root / "results" / "gpt-55_failure_analysis_factual.json",
    _fa_root / "results" / "gpt-55_failure_analysis_clinical.json",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

overchunking_mod = _load_module("overchunking", _fa_root / "chunking" / "overchunking.py")


def main() -> int:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Recompute Overchunking Scores (max run / 5)               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    for file_path in FILES_TO_PATCH:
        if not file_path.exists():
            print(f"  ⚠️  Skipping (not found): {file_path.name}")
            continue

        print(f"━━━ Processing: {file_path.name} ━━━")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        per_question = data.get("per_question_results", [])
        updated = 0
        all_scores: list[float] = []

        for pq in per_question:
            old_oc = pq.get("overchunking", {})
            doc_ids = old_oc.get("doc_ids", [])

            if not doc_ids:
                all_scores.append(0.0)
                continue

            new_oc = overchunking_mod.compute_overchunking_score(doc_ids)
            pq["overchunking"] = new_oc

            summary = pq.get("summary", {})
            summary["consecutive_chunk_ratio"] = new_oc["overchunking_score"]
            pq["summary"] = summary

            all_scores.append(new_oc["overchunking_score"])
            updated += 1

        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            agg = data.get("aggregate_scores", {})
            old_val = agg.get("consecutive_chunk_ratio", "N/A")
            agg["consecutive_chunk_ratio"] = avg
            data["aggregate_scores"] = agg
            print(f"  Updated {updated} / {len(per_question)} questions")
            print(f"  Aggregate consecutive_chunk_ratio: {old_val} → {avg:.4f}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  ✅ Saved: {file_path.name}")
        print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
