"""
NHG RAG Benchmark Pipeline
============================
Runs the full evaluation loop in three sequential stages against the QA
dataset produced by generate_qa_dataset.py (few_shot_cot strategy).
"""

import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pipeline"))

import json
import logging
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

import new_query_combined as retriever

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_PATH = "../2_generate_benchmark/qa_dataset_20260529_174846.json"
OUTPUT_DIR = "results"

GENERATOR_MODEL  = "gpt-4o"
RAGCHECKER_MODEL = "openai/gpt-4o"
RAGAS_MODEL      = "gpt-4o"

# SET HERE HOW MANY QUESTIONS YOU WANT TO DO
MAX_QUESTIONS = 2

RUN_RAGCHECKER = True
RUN_RAGAS      = True

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

        results.append({
            "query_id"         : item.get("query_id", str(i)),
            "chunk_id"         : item.get("chunk_id", ""),
            "section_path"     : item.get("section_path", ""),
            "query"            : query,
            "gt_answer"        : item["gt_answer"],
            "response"         : response_text,
            "retrieved_context": [
                {
                    "doc_id": c.get("chunk_id") or c.get("context_id", ""),
                    "text"  : retriever.build_combined_text(c),
                }
                for c in retrieved_chunks
                if c.get("text")
            ],
            "source_span"      : item.get("source_span", ""),
            "retrieval_query"  : item.get("retrieval_query", ""),
            "strategy"         : item.get("strategy", "few_shot_cot"),
        })

        time.sleep(0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"ragchecker_input_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    log.info("Stage 1 complete — %d results saved to %s", len(results), out_path)

    return results


# ---------------------------------------------------------------------------
# STAGE 2 — RAGCHECKER
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

    with open(input_path, encoding="utf-8") as f:
        rag_results = RAGResults.from_json(f.read())

    evaluator = RAGChecker(
        extractor_name=RAGCHECKER_MODEL,
        checker_name=RAGCHECKER_MODEL,
        batch_size_extractor=32,
        batch_size_checker=32,
    )
    evaluator.evaluate(rag_results, all_metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rag_results.to_json())
    log.info("RAGChecker results saved to %s", output_path)

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

    if not any(metrics_dict.values()) and getattr(rag_results, "metrics", None):
        raw_m = rag_results.metrics
        metrics_dict["overall_metrics"] = raw_m.get("overall_metrics") or raw_m.get("overall", {})
        metrics_dict["retriever_metrics"] = raw_m.get("retriever_metrics") or raw_m.get("retriever", {})
        metrics_dict["generator_metrics"] = raw_m.get("generator_metrics") or raw_m.get("generator", {})

    print("\n── RAGChecker Results ──────────────────────────────────")
    print(rag_results)
    print("────────────────────────────────────────────────────────\n")

    return metrics_dict


# ---------------------------------------------------------------------------
# STAGE 3 — RAGAS (v0.2+ API)
# ---------------------------------------------------------------------------

def stage3_ragas(results: list[dict], timestamp: str) -> dict:
    log.info("=== Stage 3: RAGAS (v0.2+) ===")

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import llm_factory
        from ragas.metrics._context_recall import LLMContextRecall
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._factual_correctness import FactualCorrectness
        from openai import OpenAI as _OpenAI
    except ImportError as exc:
        log.error("Missing RAGAS dependencies (%s).", exc)
        return {}

    ragas_rows = []
    for item in results:
        if not item.get("response") or not item.get("retrieved_context"):
            continue
        ragas_rows.append({
            "user_input"        : item["query"],
            "retrieved_contexts": [ctx["text"] for ctx in item["retrieved_context"]],
            "response"          : item["response"],
            "reference"         : item["gt_answer"],
        })

    if not ragas_rows:
        log.error("No valid rows for RAGAS.")
        return {}

    dataset = EvaluationDataset.from_list(ragas_rows)
    log.info("Running RAGAS on %d items with model=%s…", len(dataset), RAGAS_MODEL)
    evaluator_llm = llm_factory(RAGAS_MODEL, client=_OpenAI())

    metrics_list = [
        LLMContextRecall(),
        Faithfulness(),
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
    try:
        df = scores.to_pandas()
        csv_path = os.path.join(OUTPUT_DIR, f"ragas_details_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        log.info("RAGAS per-question details saved to %s", csv_path)

        for metric in metrics_list:
            if metric.name in df.columns:
                scores_dict[metric.name] = float(df[metric.name].mean())
    except Exception as e:
        log.warning("Could not extract scores from RAGAS dataframe: %s", e)

    if not scores_dict:
        try:
            scores_dict = {k: float(v) for k, v in scores.items() if isinstance(k, str)}
        except Exception:
            log.warning("Could not extract RAGAS scores.")

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
        "timestamp"      : timestamp,
        "generator_model": GENERATOR_MODEL,
        "n_questions"    : n_questions,
        "ragchecker": {
            "overall_metrics"  : ragchecker_metrics.get("overall_metrics", {}),
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

    if ragas_scores:
        print("\n  ▸ RAGAS")
        for k, v in ragas_scores.items():
            bar = "█" * int(v * 20)
            print(f"    {k:<28} {v:.4f}  {bar}")
    else:
        print("\n  ▸ RAGAS  (skipped or failed)")

    has_ragchecker = ragchecker_metrics and any(
        isinstance(v, dict) and len(v) > 0 for v in ragchecker_metrics.values()
    )

    if has_ragchecker:
        print("\n  ▸ RAGChecker")
        for section in ["overall_metrics", "retriever_metrics", "generator_metrics"]:
            vals = ragchecker_metrics.get(section, {})
            label = section.replace("_metrics", "").capitalize()
            print(f"    [{label}]")
            for k, v in vals.items():
                print(f"      {k:<28} {v}")
    else:
        print("\n  ▸ RAGChecker  (skipped or failed)")

    print(f"\n  Full report → {path}")
    print("══════════════════════════════════════════════════════\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # Seeds python random generation
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("Loading benchmark from %s", BENCHMARK_PATH)
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    benchmark: list[dict] = raw["results"] if isinstance(raw, dict) and "results" in raw else raw

    env_max = os.getenv("MAX_QUESTIONS")
    current_max = int(env_max) if env_max is not None else MAX_QUESTIONS

    # Random selection logic safely implemented here
    if current_max is not None:
        if current_max < len(benchmark):
            log.info("Randomly sampling %d questions out of %d available (Seed: %d).", current_max, len(benchmark), SEED)
            benchmark = random.sample(benchmark, current_max)
        else:
            log.warning("MAX_QUESTIONS (%d) exceeds or matches dataset length (%d). Processing all.", current_max, len(benchmark))

    log.info("Running benchmark on %d questions.", len(benchmark))

    results = stage1_retrieve_and_generate(benchmark, openai_client, timestamp)
    if not results:
        log.error("Stage 1 produced no results — aborting.")
        return

    input_path = os.path.join(OUTPUT_DIR, f"ragchecker_input_{timestamp}.json")

    ragchecker_metrics = {}
    if RUN_RAGCHECKER:
        ragchecker_metrics = stage2_ragchecker(input_path, timestamp)
    else:
        log.info("Stage 2 (RAGChecker) skipped — RUN_RAGCHECKER=False")

    ragas_scores = {}
    if RUN_RAGAS:
        ragas_scores = stage3_ragas(results, timestamp)
    else:
        log.info("Stage 3 (RAGAS) skipped — RUN_RAGAS=False")

    save_combined_report(ragchecker_metrics, ragas_scores, len(results), timestamp)


if __name__ == "__main__":
    main()