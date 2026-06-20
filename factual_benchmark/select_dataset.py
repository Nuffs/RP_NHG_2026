import json
import os

RESULTS_DIR = "factual_benchmark/results"

def select_best_qa_pairs(evaluated_pairs):
    """
    Select the best QA pair for each chunk based on combined score of grounding and round trip evaluation.

    Args:
        evaluated_pairs (list): List of evaluated QA pair dictionaries with keys:
            - 'chunk_id' (str): Unique chunk identifier
            - 'question' (str): Question text
            - 'answer' (str): Generated answer
            - 'source_text' (str): Original source text for the chunk
            - 'precision_grounding' (float): BERTScore precision for grounding
            - 'recall_grounding' (float): BERTScore recall for grounding
            - 'f1_grounding' (float): BERTScore F1 for grounding
            - 'roundtrip_answer' (str): Answer generated from round trip evaluation
            - 'precision_roundtrip' (float): BERTScore precision for round trip evaluation
            - 'recall_roundtrip' (float): BERTScore recall for round trip evaluation
            - 'f1_roundtrip' (float): BERTScore F1 for round trip evaluation
            - 'combined_score' (float): Average of grounding precision and round trip precision
    
    Returns:
        list: List of best QA pair dictionaries (one per chunk), sorted by chunk_id.
            Each dict contains the same fields as input but only the highest-scoring pair.
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