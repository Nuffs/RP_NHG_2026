"""
NHG RAG Benchmark Pipeline — Optimized & Resilient Version
==========================================================
Runs the full evaluation loop in three sequential stages with aggressive token-throttling
to reduce costs, plus continuous intermediate saves per QA pair for both metrics.
Includes a resume mechanism to skip Stage 1 using existing inputs.
"""

import sys
import os
import random
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import new_query_combined as retriever

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_PATH = "../2_generate_benchmark/qa_dataset_20260529_174846.json"
OUTPUT_DIR = "results"

GENERATOR_MODEL = "gpt-4o"
RAGCHECKER_MODEL = "openai/gpt-4o-mini"
RAGAS_MODEL      = "gpt-4o-mini"

# Caps context chunks to 5 to protect budget
MAX_CONTEXT_CHUNKS = 5

# SET HERE HOW MANY QUESTIONS YOU WANT TO DO
MAX_QUESTIONS = 30

# RESUME FLAG: Set to True to skip generation and evaluate the last run's data
RESUME_FROM_STAGE2 = True

# RAGCHECKER & RAGAS FLAGS
RUN_RAGCHECKER = False
RUN_RAGAS = True

SEED = 1

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STAGE 1 — RETRIEVE & GENERATE
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ANSWER = """
Je bent een Nederlandse huisarts. Beantwoord de klinische vraag uitsluitend op basis van de
meegeleverde context uit NHG-richtlijnen. Geef een beknopt antwoord van maximaal 3 zinnen.
Noem geen informatie die niet in de context staat.
""".strip()


def stage1_retrieve_and_generate(
        benchmark: list[dict],
        openai_client: OpenAI,
        timestamp: str,
) -> list[dict]:
    log.info("=== Stage 1: Retrieve & Generate (%d questions) ===", len(benchmark))
    results = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    interim_path = os.path.join(OUTPUT_DIR, f"stage1_interim_{timestamp}.json")

    for i, item in enumerate(benchmark):
        query = item["query"]
        log.info("[%d/%d] %s — %.60s…", i + 1, len(benchmark), item.get("query_id", "?"), query)

        try:
            retrieved_chunks = retriever.run_combined(
                prompt=query,
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                context_collection="context_blocks",
                embedding_collection="embedding_blocks",
                api_key=os.getenv("GEMINI_API_KEY"),
                history_file="results/benchmark_query_history.json",
                fused_top_k=MAX_CONTEXT_CHUNKS,
            )
        except Exception as e:
            log.error("  Retrieval failed: %s", e)
            continue

        if not retrieved_chunks:
            log.warning("  No chunks retrieved — skipping.")
            continue

        context_text = "\n\n---\n\n".join(
            retriever.build_combined_text(c)
            for c in retrieved_chunks
            if c.get("text")
        )

        try:
            resp = openai_client.chat.completions.create(
                model=GENERATOR_MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nVraag: {query}"},
                ],
                temperature=0,
                seed=SEED,
            )
            response_text = resp.choices[0].message.content.strip()
        except Exception as e:
            log.error("  Generation failed: %s", e)
            response_text = ""

        record = {
            "query_id": item.get("query_id", str(i)),
            "chunk_id": item.get("chunk_id", ""),
            "section_path": item.get("section_path", ""),
            "query": query,
            "gt_answer": item["gt_answer"],
            "response": response_text,
            "retrieved_context": [
                {
                    "doc_id": c.get("chunk_id") or c.get("context_id", ""),
                    "text": retriever.build_combined_text(c),
                }
                for c in retrieved_chunks
                if c.get("text")
            ],
            "source_span": item.get("source_span", ""),
            "retrieval_query": item.get("retrieval_query", ""),
            "strategy": item.get("strategy", "few_shot_cot"),
        }

        results.append(record)

        with open(interim_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    out_path = os.path.join(OUTPUT_DIR, f"ragchecker_input_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    if os.path.exists(interim_path):
        os.remove(interim_path)

    log.info("Stage 1 complete — %d results saved to %s", len(results), out_path)
    return results


# ---------------------------------------------------------------------------
# STAGE 2 — RAGCHECKER (Per QA pair logging)
# ---------------------------------------------------------------------------

def stage2_ragchecker(input_path: str, timestamp: str) -> dict:
    log.info("=== Stage 2: RAGChecker ===")

    try:
        from ragchecker import RAGResults, RAGChecker
        from ragchecker.metrics import all_metrics
    except ImportError:
        log.error("ragchecker not installed.")
        return {}

    output_path = os.path.join(OUTPUT_DIR, f"ragchecker_output_{timestamp}.json")
    per_qa_path = os.path.join(OUTPUT_DIR, f"ragchecker_per_qa_{timestamp}.json")

    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)
        rag_results = RAGResults.from_json(json.dumps(raw_data))

    evaluator = RAGChecker(
        extractor_name=RAGCHECKER_MODEL,
        checker_name=RAGCHECKER_MODEL,
        batch_size_extractor=6,
        batch_size_checker=6,
    )

    evaluator.evaluate(rag_results, all_metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rag_results.to_json())

    per_qa_details = {}
    try:
        for idx, item in enumerate(raw_data.get("results", [])):
            q_id = item.get("query_id", f"idx_{idx}")
            per_qa_details[q_id] = {
                "query": item.get("query"),
                "metrics": {
                    "precision": getattr(rag_results, "precision", None),
                    "recall": getattr(rag_results, "recall", None),
                    "f1": getattr(rag_results, "f1", None),
                }
            }
    except Exception as e:
        log.warning("Could not extract fine-grained per-QA metrics for RAGChecker: %s", e)

    with open(per_qa_path, "w", encoding="utf-8") as f:
        json.dump(per_qa_details, f, indent=2, ensure_ascii=False)

    metrics_dict = {
        "overall_metrics": {},
        "retriever_metrics": {},
        "generator_metrics": {}
    }

    for target_key, attributes in [
        ("overall_metrics", ["overall_metrics", "overall"]),
        ("retriever_metrics", ["retriever_metrics", "retriever"]),
        ("generator_metrics", ["generator_metrics", "generator"])
    ]:
        for attr in attributes:
            val = getattr(rag_results, attr, None)
            if val:
                metrics_dict[target_key] = val if isinstance(val, dict) else vars(val)
                break

    print("\n── RAGChecker Results ──────────────────────────────────")
    print(rag_results)
    print("────────────────────────────────────────────────────────\n")

    return metrics_dict


# ---------------------------------------------------------------------------
# STAGE 3 — RAGAS (v0.2+ API with row-by-row logging)
# ---------------------------------------------------------------------------

def stage3_ragas(results: list[dict], timestamp: str) -> dict:
    log.info("=== Stage 3: RAGAS (v0.2+) ===")

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            Faithfulness,
            ContextRecall,
            FactualCorrectness,
        )
    except ImportError as exc:
        log.error(
            "Missing RAGAS base or Langchain dependencies (%s). Ensure 'pip install langchain-openai' is run.",
            exc)
        return {}

    ragas_rows = []
    for item in results:
        if not item.get("response") or not item.get("retrieved_context"):
            continue
        ragas_rows.append({
            "user_input": item["query"],
            "retrieved_contexts": [ctx["text"] for ctx in item["retrieved_context"]],
            "response": item["response"],
            "reference": item["gt_answer"],
            "query_id": item.get("query_id")
        })

    if not ragas_rows:
        log.error("No valid rows for RAGAS.")
        return {}

    dataset = EvaluationDataset.from_list(ragas_rows)
    log.info("Running RAGAS on %d items with model=%s…", len(dataset), RAGAS_MODEL)

    # Configure Modern LLM Wrapper Architecture
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=RAGAS_MODEL, api_key=os.environ.get("OPENAI_API_KEY", ""))
    )

    # FIXED: Explicit metric initializations matching the structural tracking classes
    metrics_list = [
        ContextRecall(),
        ContextPrecision(),
        Faithfulness(),
        AnswerRelevancy(),
        FactualCorrectness(),
    ]

    try:
        scores = evaluate(
            dataset=dataset,
            metrics=metrics_list,
            llm=evaluator_llm,
        )
    except Exception as e:
        log.error("RAGAS evaluate() failed: %s", e)
        return {}

    scores_dict: dict = {}
    per_qa_ragas_path = os.path.join(OUTPUT_DIR, f"ragas_per_qa_{timestamp}.json")

    try:
        df = scores.to_pandas()
        csv_path = os.path.join(OUTPUT_DIR, f"ragas_details_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        log.info("RAGAS per-question details saved to %s", csv_path)

        per_qa_details = {}
        for idx, row in df.iterrows():
            q_id = ragas_rows[idx]["query_id"] if idx < len(ragas_rows) else f"row_{idx}"

            row_metrics = {}
            for col in df.columns:
                if col not in ["user_input", "retrieved_contexts", "response", "reference"]:
                    row_metrics[col] = row.get(col)

            per_qa_details[q_id] = {
                "user_input": row.get("user_input", ""),
                "metrics": row_metrics
            }

        with open(per_qa_ragas_path, "w", encoding="utf-8") as f:
            json.dump(per_qa_details, f, indent=2, ensure_ascii=False)
        log.info("Saved granular individual QA metrics to %s", per_qa_ragas_path)

        for col in df.columns:
            if col not in ["user_input", "retrieved_contexts", "response", "reference"]:
                scores_dict[col] = float(df[col].mean())

    except Exception as e:
        log.warning("Could not extract scores from RAGAS dataframe: %s", e)

    print("\n── RAGAS Results ───────────────────────────────────────")
    for k, v in scores_dict.items():
        print(f"  {k:<28} {v:.4f}")
    print("────────────────────────────────────────────────────────\n")

    return scores_dict

# ---------------------------------------------------------------------------
# COMBINED REPORT
# ---------------------------------------------------------------------------

def save_combined_report(
        ragchecker_metrics: dict,
        ragas_scores: dict,
        n_questions: int,
        timestamp: str,
) -> None:
    report = {
        "timestamp": timestamp,
        "generator_model": GENERATOR_MODEL,
        "n_questions": n_questions,
        "ragchecker": {
            "overall_metrics": ragchecker_metrics.get("overall_metrics", {}),
            "retriever_metrics": ragchecker_metrics.get("retriever_metrics", {}),
            "generator_metrics": ragchecker_metrics.get("generator_metrics", {}),
        },
        "ragas": ragas_scores,
    }

    path = os.path.join(OUTPUT_DIR, f"combined_report_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n══════════════════════════════════════════════════════")
    print("  COMBINED BENCHMARK REPORT")
    print(f"  Timestamp           : {timestamp}")
    print(f"  Questions evaluated : {n_questions}")
    print(f"  Generator model     : {GENERATOR_MODEL}")
    print("══════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if RESUME_FROM_STAGE2:
        log.info("=== RESUME MODE ACTIVE ===")
        search_pattern = os.path.join(OUTPUT_DIR, "ragchecker_input_*.json")
        existing_files = sorted(glob.glob(search_pattern))

        if not existing_files:
            log.error("No existing input files found in '%s' matching pattern. Cannot resume.", OUTPUT_DIR)
            return

        input_path = existing_files[-1]
        log.info("Found existing generated payload: %s", input_path)

        filename = os.path.basename(input_path)
        timestamp = filename.replace("ragchecker_input_", "").replace(".json", "")
        log.info("Reusing historical benchmark execution timestamp: %s", timestamp)

        with open(input_path, encoding="utf-8") as f:
            raw_data = json.load(f)
            results = raw_data.get("results", [])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log.info("Loading benchmark from %s", BENCHMARK_PATH)
        with open(BENCHMARK_PATH, encoding="utf-8") as f:
            raw = json.load(f)

        all_items: list[dict] = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
        benchmark = [item for item in all_items if item.get("doc_id") == "astma_bij_volwassenen"]

        log.info("Filtered dataset: Kept %d asthma questions.", len(benchmark))

        env_max = os.getenv("MAX_QUESTIONS")
        current_max = int(env_max) if env_max is not None else MAX_QUESTIONS

        if current_max is not None and current_max < len(benchmark):
            log.info("Randomly sampling %d questions.", current_max)
            benchmark = random.sample(benchmark, current_max)

        log.info("Running benchmark on %d questions.", len(benchmark))

        results = stage1_retrieve_and_generate(benchmark, openai_client, timestamp)
        if not results:
            log.error("Stage 1 produced no results — aborting.")
            return

        input_path = os.path.join(OUTPUT_DIR, f"ragchecker_input_{timestamp}.json")

    ragchecker_metrics = {}
    if RUN_RAGCHECKER:
        ragchecker_metrics = stage2_ragchecker(input_path, timestamp)

    ragas_scores = {}
    if RUN_RAGAS:
        ragas_scores = stage3_ragas(results, timestamp)

    save_combined_report(ragchecker_metrics, ragas_scores, len(results), timestamp)


if __name__ == "__main__":
    main()