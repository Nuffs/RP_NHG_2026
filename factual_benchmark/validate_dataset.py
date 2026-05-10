from bert_score import score as bert_score
import json
import os


def evaluate_qa_pairs(scraped_chunks, qa_pairs):
    chunk_lookup = {chunk["chunk_id"]: chunk["text"] for chunk in scraped_chunks}

    references = [] # source chunks
    hypotheses = [] # generated answers
    valid_pairs = []

    for qa in qa_pairs:
        context = chunk_lookup.get(qa["chunk_id"], "NO_SOURCE_FOUND")
        if context == "NO_SOURCE_FOUND":
            print(f"  WARNING: No source chunk found for chunk_id {qa['chunk_id']}")
            continue
        hypotheses.append(qa["answer"])
        references.append(context)
        valid_pairs.append(qa)
    
    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang="nl",
        verbose=True
    )

    per_pair_scores = []
    for i, qa in enumerate(valid_pairs):
        per_pair_scores.append({
            "chunk_id": qa["chunk_id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "precision": P[i].item(),
            "recall": R[i].item(),
            "f1": F1[i].item()
        })

    print(f"Overall BERTScore - Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")  

    return {
        "overall": {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        },
        "per_pair": per_pair_scores
    }


if __name__ == "__main__":
    with open("data/benchmark_chunks.jsonl", "r") as f:
        scraped_chunks = [json.loads(line) for line in f]
    
    with open(os.path.join("factual_benchmark/results", "qa_zero_shot.json"), "r") as f:
        qa_pairs = json.load(f)
    
    results = evaluate_qa_pairs(scraped_chunks, qa_pairs)
    
    output_path = os.path.join("factual_benchmark/results", "evaluation_zero_shot.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved evaluation results to {output_path}")
