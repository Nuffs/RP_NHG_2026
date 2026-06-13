"""
convert_to_ragchecker
----------------------

Utility script that converts internal benchmark result records into the
input format expected by the third-party `ragchecker` evaluator. This
module reads `results/benchmark_results.json`, filters out items that do
not have successful retrieval/generation statuses, and writes a compact
JSON file with the fields required by RAGChecker.

This file intentionally keeps the transformation minimal and defensive:
it uses `.get()` for optional keys and logs skipped records.

Example usage:
    python factual_benchmark/convert_to_ragchecker.py

Outputs:
    factual_benchmark/results/ragchecker_input.json
"""

import json
from pathlib import Path

INPUT_FILE  = Path(__file__).resolve().parent / "results" / "benchmark_results.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / "ragchecker_input.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    benchmark = json.load(f)

ragchecker_results = []
for item in benchmark:
    if item.get("retrieval_status") != "ok" or item.get("generation_status") != "ok":
        continue

    ragchecker_results.append({
        "query_id": item["chunk_id"],
        "query": item["question"],
        "gt_answer": item["expected_answer"],
        "response": item["generated_answer"],
        "retrieved_context": [
            {
                "doc_id": str(i),
                "text": ctx.get("text", "")
            }
            for i, ctx in enumerate(item["retrieved_contexts"])
        ]
    })

output = {"results": ragchecker_results}
Path(OUTPUT_FILE).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Converted {len(ragchecker_results)} items → {OUTPUT_FILE}")