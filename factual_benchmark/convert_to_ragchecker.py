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