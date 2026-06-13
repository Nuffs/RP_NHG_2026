import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import sys
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline.new_query_combined as combined
from openai import OpenAI

load_dotenv()
client = OpenAI()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
HERE = Path(__file__).resolve().parent
BENCHMARK_FILE = str(HERE / "results" / "qa_final_dataset.json")
OUTPUT_FILE = str(HERE / "results" / "benchmark_results.json")
TOP_N_CONTEXT = 5
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"


def load_benchmark_data(benchmark_file):
    """
    Load the benchmark JSON file from disk.

    Args:
        benchmark_file (str): Path to a JSON file containing a list of QA pairs.

    Returns:
        list: Parsed list of QA pair dictionaries.
    """
    with open(benchmark_file, "r") as f:
        return json.load(f)


def format_context(results):
    """
    Convert a list of retrieved result dicts into a single text context
    that can be fed to the LLM.

    Args:
        results (list): List of retrieved result dicts containing
            'section_path' and 'text'.

    Returns:
        str: Concatenated, human-readable context string.
    """
    parts = []
    for idx, res in enumerate(results, start=1):
        section = " > ".join(res.get("section_path", []))
        text = res.get("text", "").strip()
        parts.append(f"[{idx}] {section}\n{text}".strip())
    return "\n\n".join(parts)


def generate_answer(question, context):
    """
    Query the configured LLM for an answer using the provided context.

    Args:
        question (str): The question text.
        context (str): The concatenated retrieved context.

    Returns:
        str: The LLM's generated answer (stripped).
    """
    prompt = f"""Je bent een medisch informatiesysteem. Beantwoord de volgende vraag uitsluitend op basis van de gegeven context.
    Als het antwoord niet in de context te vinden is, zeg dan "Geen antwoord gevonden in de context." Geef geen extra informatie of aannames.
    Geef een feitelijk antwoord in het Nederlands.
    Context:\n{context}\n\nVraag: {question}\n\nAntwoord:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": "Je bent een medisch informatiesysteem dat vragen beantwoordt op basis van gegeven context."},
                {"role": "user", "content": prompt}],
        response_format={"type": "text"}
    )

    return response.choices[0].message.content.strip()


def sample_per_guideline(data, n=5, seed=42):
    """Sample n random Q&A pairs per guideline (based on chunk_id prefix)."""

    groups = defaultdict(list)
    for item in data:
        chunk_id = item.get("chunk_id", "")
        parts = chunk_id.split("_")
        # Guideline = alles vóór de laatste underscore+nummer, bv. "astma_bij_volwassenen"
        guideline = "_".join(parts[:-2]) if len(parts) > 2 else chunk_id
        groups[guideline].append(item)

    random.seed(seed)
    sampled = []
    for guideline, items in groups.items():
        sampled.extend(random.sample(items, min(n, len(items))))

    print(f"Sampled {len(sampled)} pairs across {len(groups)} guidelines.")
    return sampled


def run_benchmark():
    """
    Execute the benchmark runner end-to-end for a sampled subset.

    Steps:
    - sample QA pairs
    - retrieve context via pipeline.run_combined
    - generate answers with the LLM
    - save structured results to OUTPUT_FILE
    """
    benchmark = sample_per_guideline(load_benchmark_data(BENCHMARK_FILE), n=5)
    print(f"Loaded {len(benchmark)} Q&A pairs from benchmark file.")

    output = []

    for item in benchmark:
        qid = item.get("chunk_id", "?")
        question = item["question"]
        expected_answer = item["answer"]

        try:
            results = combined.run_combined(
                prompt=question,
                qdrant_url=QDRANT_URL,
                vector_top_k=100,
                traditional_top_k=20,
                score_threshold_vector=0.7,
                api_key=None,
            )
            top_results = results[:TOP_N_CONTEXT]
            context = format_context(top_results)
            retr_status = "ok"
        except Exception as e:
            print(f"Error processing question {qid}: {e}")
            top_results = []
            context = ""
            retr_status = "error"

        print(retr_status)

        generated_answer = ""
        gen_status = "skipped"
        if context:
            try:
                generated_answer = generate_answer(question, context)
                gen_status = "ok"
                time.sleep(1)
            except Exception as e:
                print(f"Error generating answer for question {qid}: {e}")
                gen_status = "error"
        print(gen_status)

        output.append(
            {
                "chunk_id": qid,
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_contexts": top_results,
                "generated_answer": generated_answer,
                "context_fed_to_llm": context,
                "retrieval_status": retr_status,
                "generation_status": gen_status,
            }
        )

    Path(OUTPUT_FILE).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_benchmark()

