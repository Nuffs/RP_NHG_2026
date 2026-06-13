"""
select_dataset
--------------

Utilities for selecting the best QA pair per source chunk from an
evaluation output. The main routine `select_best_qa_pairs` groups
evaluated QA candidates by their `chunk_id` and returns the highest
scoring candidate per chunk.

Typical input is `factual_benchmark/results/evaluation_scores.json` which
may contain an object with a `per_pair` key or a plain list of
evaluated pair dictionaries.
"""

import json
import os

RESULTS_DIR = "factual_benchmark/results"


def select_best_qa_pairs(evaluated_pairs):
    """
    Select the best QA pair for each chunk based on combined score.

    Javadoc-style summary:
    - Inputs: list of evaluated QA pair dicts (see module docstring for fields)
    - Output: list of best QA pair dicts, one per `chunk_id`.

    Args:
        evaluated_pairs (list): Evaluated QA pair dictionaries. Each dict is
            expected to include a numeric `combined_score` key used for ranking.

    Returns:
        list: Best QA pair for each chunk (unsorted order).
    """
    chunks_by_id = {}
    for pair in evaluated_pairs:
        chunk_id = pair["chunk_id"]
        if chunk_id not in chunks_by_id:
            chunks_by_id[chunk_id] = []
        chunks_by_id[chunk_id].append(pair)

    best_pairs = []
    for chunk_id, pairs in chunks_by_id.items():
        best_pair = max(pairs, key=lambda x: x["combined_score"])
        best_pairs.append(best_pair)

    return best_pairs


if __name__ == "__main__":
    with open(os.path.join(RESULTS_DIR, "evaluation_scores.json"), "r") as f:
        evaluation_data = json.load(f)

    if isinstance(evaluation_data, dict) and "per_pair" in evaluation_data:
        evaluated_pairs = evaluation_data["per_pair"]
    else:
        evaluated_pairs = evaluation_data if isinstance(evaluation_data, list) else [evaluation_data]

    best_pairs = select_best_qa_pairs(evaluated_pairs)

    print(f"Selected {len(best_pairs)} best QA pairs, one per chunk.")

    output_path = os.path.join(RESULTS_DIR, "qa_final_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_pairs, f, ensure_ascii=False, indent=2)