"""
validate_dataset
-----------------

Evaluation utilities for QA pairs. This module contains three main
functions:

- evaluate_qa_pairs_grounding: compares generated answers to the source
  text (grounding) using BERTScore.
- evaluate_qa_pairs_roundtrip: asks the model the same question again
  using only the source context and compares the original and roundtrip
  answers.
- evaluate_qa_pairs: combines grounding and roundtrip scores and
  computes overall statistics.

The functions are defensive and print warnings for missing source
chunks. They expect Dutch-language text (`lang='nl'`) for BERTScore.
"""

from bert_score import score as bert_score
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_GPT5"))

RESULTS_DIR = "factual_benchmark/results"


def evaluate_qa_pairs_grounding(scraped_chunks, qa_pairs):
    """
    Evaluate QA pairs using grounding (source text fidelity) with BERTScore.

    Summary (Javadoc-style):
    - Purpose: compute BERTScore metrics between generated answers and the
      corresponding source chunk texts to measure faithfulness.
    - Inputs: scraped_chunks (list of {'chunk_id','text',...}), qa_pairs
      (list of {'chunk_id','question','answer',...}).
    - Output: list of per-pair dicts with keys: 'chunk_id','question','answer',
      'source_text','precision','recall','f1'.

    Args:
        scraped_chunks (list): Source chunk dicts.
        qa_pairs (list): Generated QA pair dicts.

    Returns:
        list: Per-pair BERTScore metrics.
    """
    chunk_lookup = {chunk["chunk_id"]: chunk["text"] for chunk in scraped_chunks}

    # source chunks
    source_chunks = []
    # generated answers
    hypotheses = []
    valid_pairs = []

    for qa in qa_pairs:
        context = chunk_lookup.get(qa["chunk_id"], "NO_SOURCE_FOUND")
        if context == "NO_SOURCE_FOUND":
            print(f"No source chunk found for chunk_id {qa['chunk_id']}")
            continue
        hypotheses.append(qa["answer"])
        source_chunks.append(context)
        valid_pairs.append({**qa, "source_text": context})

    P, R, F1 = bert_score(
        hypotheses,
        source_chunks,
        lang="nl",
        verbose=True,
    )

    per_pair_scores = []
    for i, qa in enumerate(valid_pairs):
        per_pair_scores.append({
            "chunk_id": qa["chunk_id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "source_text": qa["source_text"],
            "precision": P[i].item(),
            "recall": R[i].item(),
            "f1": F1[i].item(),
        })

    return per_pair_scores

def get_roundtrip_answer(question, context):
    """
    Obtain a roundtrip answer from the LLM using only the provided source
    context.

    This function calls the configured OpenAI client and returns the raw
    text response. The function is intentionally narrow: it instructs the
    model to answer factually and without added explanation so the
    resulting text can be compared with the original generated answer
    using BERTScore.

    Args:
        question (str): The question to ask the model.
        context (str): The source text to be provided as context.

    Returns:
        str: The model-produced answer (stripped).
    """

    # Use a conservative, widely-available chat model by default.
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"""Beantwoord de volgende vraag uitsluitend op basis van de gegeven tekst. Geef een direct en feitelijk antwoord zonder uitleg of context toe te voegen. Tekst: {context} Vraag: {question}""",
            }
        ],
        response_format={"type": "text"},
    )

    return response.choices[0].message.content.strip()

def evaluate_qa_pairs_roundtrip(scraped_chunks, qa_pairs):
    """
    Evaluate QA pair consistency using roundtrip evaluation (BERTScore).
    
    Measures question quality/precision. For each QA pair, the model is asked the same question again using only the source chunk as context. 
    The original answer and roundtrip answer are compared using BERTScore to measure question quality and model consistency.
    
    Args:
        scraped_chunks (list): List of chunk dicts with 'chunk_id' and 'text' keys
        qa_pairs (list): List of QA pair dicts with 'chunk_id', 'question', 'answer' keys
    
    Returns:
        list: List of evaluation dicts with keys:
            - chunk_id, question, answer, source_text
            - roundtrip_answer: model's answer when asked again with source context
            - precision, recall, f1: BERTScore metrics comparing original vs roundtrip answers
            (Higher scores indicate better question formulation and model consistency)
    """
    
    chunk_lookup = {chunk["chunk_id"]: chunk["text"] for chunk in scraped_chunks}

    valid_pairs = []

    for qa in qa_pairs:
        context = chunk_lookup.get(qa["chunk_id"], "NO_SOURCE_FOUND")
        if context == "NO_SOURCE_FOUND":
            print(f"No source chunk found for chunk_id {qa['chunk_id']}")
            continue
        valid_pairs.append({**qa, "source_text": context})

    roundtrip_answers = []
    # for each QA pair, generate a roundtrip answer by asking the question again with the source text as context
    for qa in valid_pairs:
        print(f"Roundtrip for chunk {qa['chunk_id']}")
        rt_answer = get_roundtrip_answer(qa["question"], qa["source_text"])
        roundtrip_answers.append(rt_answer)
    
    print("\nCalculating BERTScore for roundtrip consistency")
    P, R, F1 = bert_score(
        roundtrip_answers,
        [qa["answer"] for qa in valid_pairs],
        lang="nl",
        verbose=True
    )

    per_pair_scores = []
    for i, qa in enumerate(valid_pairs):
        per_pair_scores.append({
            "chunk_id": qa["chunk_id"],
            "question": qa["question"],
            "answer": qa["answer"],
            "source_text": qa["source_text"],
            "roundtrip_answer": roundtrip_answers[i],
            "precision": P[i].item(),
            "recall": R[i].item(),
            "f1": F1[i].item()
        })
    
    return per_pair_scores

def evaluate_qa_pairs(scraped_chunks, qa_pairs):
    """
    Comprehensive QA pair evaluation combining grounding and roundtrip consistency metrics.
    
    Calculates two evaluation dimensions:
    1. Grounding (Faithfulness): How well does the answer faithfully represent the source text?
    Uses BERTScore to compare answer against source chunk.
    
    2. Roundtrip (Consistency & Question Quality): Does the model give consistent answers? And how well-formulated is the question?
    Asks the model the question again with source context and compares answers using BERTScore.
    Higher scores indicate: (a) consistent model behavior, and (b) precise, unambiguous questions.
    Ambiguous or poorly-formulated questions may elicit different answers, lowering roundtrip scores.
    
    Args:
        scraped_chunks (list): List of chunk dicts with 'chunk_id' and 'text' keys
        qa_pairs (list): List of QA pair dicts with 'chunk_id', 'question', 'answer' keys
    
    Returns:
        dict: Results with structure:
            {
                "overall": {
                    "precision_grounding": float,
                    "recall_grounding": float,
                    "f1_grounding": float,
                    "precision_roundtrip": float,
                    "recall_roundtrip": float,
                    "f1_roundtrip": float,
                    "combined_score": float
                },
                "per_pair": [
                    {
                        "chunk_id": str,
                        "question": str,
                        "answer": str,
                        "source_text": str,
                        "roundtrip_answer": str,
                        "precision_grounding": float,
                        "recall_grounding": float,
                        "f1_grounding": float,
                        "precision_roundtrip": float,
                        "recall_roundtrip": float,
                        "f1_roundtrip": float,
                        "combined_score": float
                    },
                    ...
                ]
            }
    """
    
    grounding_scores = evaluate_qa_pairs_grounding(scraped_chunks, qa_pairs)
    roundtrip_scores = evaluate_qa_pairs_roundtrip(scraped_chunks, qa_pairs)
    
    roundtrip_lookup = {
        (s["chunk_id"], s["question"]): s for s in roundtrip_scores
    }

    combined_scores = []
    for grounding in grounding_scores:
        key = (grounding["chunk_id"], grounding["question"])
        roundtrip = roundtrip_lookup.get(key, {})
        combined_scores.append({
            "chunk_id": grounding["chunk_id"],
            "question": grounding["question"],
            "answer": grounding["answer"],
            "source_text": grounding["source_text"],
            "precision_grounding": grounding["precision"],
            "recall_grounding": grounding["recall"],
            "f1_grounding": grounding["f1"],
            "roundtrip_answer": roundtrip.get("roundtrip_answer"),
            "precision_roundtrip": roundtrip.get("precision"),
            "recall_roundtrip": roundtrip.get("recall"),
            "f1_roundtrip": roundtrip.get("f1"),
            "combined_score": (grounding["precision"] + roundtrip.get("precision", 0)) / 2
        })

    overall = {
        "precision_grounding": sum(s["precision_grounding"] for s in combined_scores) / len(combined_scores),
        "recall_grounding": sum(s["recall_grounding"] for s in combined_scores) / len(combined_scores),
        "f1_grounding": sum(s["f1_grounding"] for s in combined_scores) / len(combined_scores),
        "precision_roundtrip": sum(s["precision_roundtrip"] for s in combined_scores if s["precision_roundtrip"] is not None) / len([s for s in combined_scores if s["precision_roundtrip"] is not None]),
        "recall_roundtrip": sum(s["recall_roundtrip"] for s in combined_scores if s["recall_roundtrip"] is not None) / len([s for s in combined_scores if s["recall_roundtrip"] is not None]),
        "f1_roundtrip": sum(s["f1_roundtrip"] for s in combined_scores if s["f1_roundtrip"] is not None) / len([s for s in combined_scores if s["f1_roundtrip"] is not None]),
        "combined_score": sum(s["combined_score"] for s in combined_scores) / len(combined_scores)
    }

    return {
        "overall": overall,
        "per_pair": combined_scores
    }

if __name__ == "__main__":
    with open("data/benchmark_chunks.jsonl", "r") as f:
        scraped_chunks = [json.loads(line) for line in f]
    
    with open(os.path.join(RESULTS_DIR, "qa_final.json"), "r") as f:
        qa_pairs = json.load(f)
    
    results = evaluate_qa_pairs(scraped_chunks, qa_pairs)
    
    output_path = os.path.join(RESULTS_DIR, "evaluation_scores.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved round trip & grounding evaluation results to {output_path}")
