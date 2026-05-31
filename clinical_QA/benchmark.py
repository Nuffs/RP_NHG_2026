"""
NHG RAG Benchmark Pipeline
============================
Runs the full evaluation loop in three sequential stages against the QA
dataset produced by generate_qa_dataset.py (few_shot_cot strategy).

  Stage 1 — RETRIEVE & GENERATE
    • Reads qa_dataset_<ts>.json (or qa_dataset.json as a stable symlink)
    • Embeds each query with Gemini, retrieves top-k NHG chunks
    • Generates an answer with GPT-4o
    • Saves results to ragchecker_input_<ts>.json

  Stage 2 — RAGCHECKER  (claim-level, fine-grained)
    • overall  : precision, recall, F1
    • retriever : claim recall, context precision
    • generator : faithfulness, hallucination, self-knowledge, ...

  Stage 3 — RAGAS  (embedding + LLM-as-judge)
    • faithfulness, answer relevancy, context precision, context recall

Outputs (all timestamped so runs never overwrite each other):
  results/ragchecker_input_<ts>.json   — RAG pipeline outputs
  results/ragchecker_output_<ts>.json  — RAGChecker scored results
  results/ragas_scores_<ts>.json       — RAGAS metric scores
  results/ragas_details_<ts>.csv       — per-question RAGAS breakdown
  results/combined_report_<ts>.json    — both frameworks side-by-side

Usage:
  pip install openai ragchecker ragas datasets python-dotenv google-genai
  python -m spacy download en_core_web_sm

  # Point BENCHMARK_PATH at your generated file, then:
  python run_rag_benchmark.py

  # Quick smoke-test on 10 questions:
  MAX_QUESTIONS=10 python run_rag_benchmark.py
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to the dataset produced by generate_qa_dataset.py.
BENCHMARK_PATH   = "results/qa_dataset_20260529_174846.json"   # ← update to your filename

EMBEDDINGS_PATH  = "../pipeline/embeddings.json"
CHUNKS_PATH      = "../data/nhg_subset_guidelines.jsonl"
OUTPUT_DIR       = "results"

RETRIEVAL_TOP_K  = 5
GEMINI_MODEL     = "gemini-embedding-2"

# All three stages use GPT-4o for maximum quality.
GENERATOR_MODEL  = "gpt-4o"
RAGCHECKER_MODEL = "openai/gpt-4o"   # RAGChecker prefix format
RAGAS_MODEL      = "gpt-4o"          # passed to the RAGAS LLM wrapper

# Fallback value. Overridden if the MAX_QUESTIONS environment variable is provided.
MAX_QUESTIONS    = 2

# Set to True to run each evaluation framework; False to skip.
RUN_RAGCHECKER   = True
RUN_RAGAS        = False

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
# HELPERS — embedding & retrieval
# ---------------------------------------------------------------------------

_embeddings_cache: list[dict] | None = None


def load_embeddings(path: str) -> list[dict]:
    global _embeddings_cache
    if _embeddings_cache is None:
        log.info("Loading embeddings from %s …", path)
        with open(path, encoding="utf-8") as f:
            _embeddings_cache = json.load(f)
        log.info("Loaded %d embedded chunks.", len(_embeddings_cache))
    return _embeddings_cache


def embed_query(text: str) -> list[float]:
    """Embed a query string with Gemini."""
    from google import genai
    from google.genai import types

    gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = gclient.models.embed_content(
        model=GEMINI_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(resp.embeddings[0].values)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec)
    m_norms = np.linalg.norm(matrix, axis=1)
    scores = np.zeros(len(matrix))
    if q_norm == 0:
        return scores
    denom = m_norms * q_norm
    valid = denom > 0
    scores[valid] = np.dot(matrix[valid], query_vec) / denom[valid]
    return scores


def retrieve(query: str, top_k: int = RETRIEVAL_TOP_K, max_retries: int = 3) -> list[str]:
    """Return the top-k chunk_ids for a query, forced to strings with exponential back-off."""
    index = load_embeddings(EMBEDDINGS_PATH)
    vectors = np.array([item["embedding"] for item in index])

    for attempt in range(1, max_retries + 1):
        try:
            q_vec = np.array(embed_query(query))
            scores = cosine_similarity(q_vec, vectors)
            top_idx = np.argsort(scores)[::-1][:top_k]
            # Convert chunk_id explicitly to string to match string-cast lookup keys
            return [str(index[i]["chunk_id"]) for i in top_idx]
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 5 * (2 ** (attempt - 1))
                log.warning("  Rate limit (attempt %d/%d) — retrying in %ds…", attempt, max_retries, wait)
                time.sleep(wait)
            else:
                raise
    log.error("  Giving up on retrieval after %d retries.", max_retries)
    return []


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
    chunks_lookup: dict[str, dict],
    openai_client: OpenAI,
    timestamp: str,
) -> list[dict]:
    log.info("=== Stage 1: Retrieve & Generate (%d questions) ===", len(benchmark))
    results = []

    for i, item in enumerate(benchmark):
        query = item["query"]
        log.info("[%d/%d] %s — %.60s…", i + 1, len(benchmark), item.get("query_id", "?"), query)

        # — Retrieve —
        retrieved_ids = retrieve(query)
        retrieved_chunks = [chunks_lookup[cid] for cid in retrieved_ids if cid in chunks_lookup]

        if not retrieved_chunks:
            log.warning("  No chunks retrieved — skipping.")
            log.info("  [Debug Info] Retrieved IDs from index: %s", retrieved_ids)
            log.info("  [Debug Info] Sample guideline lookup IDs available: %s", list(chunks_lookup.keys())[:5])
            continue

        context_text = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)

        # — Generate —
        try:
            resp = openai_client.chat.completions.create(
                model=GENERATOR_MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context_text}\n\n"
                            f"Vraag: {query}"
                        ),
                    },
                ],
                temperature=0,
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
                {"doc_id": c.get("doc_id", c["chunk_id"]), "text": c["text"]}
                for c in retrieved_chunks
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
        log.error("ragchecker not installed. Run: pip install ragchecker && python -m spacy download en_core_web_sm")
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

    metrics_dict = {}
    try:
        summary = json.loads(rag_results.to_json())
        for section in ("overall_metrics", "retriever_metrics", "generator_metrics"):
            if section in summary:
                metrics_dict[section] = summary[section]
    except Exception as e:
        log.warning("Could not parse RAGChecker summary: %s", e)

    print("\n── RAGChecker Results ──────────────────────────────────")
    print(rag_results)
    print("────────────────────────────────────────────────────────\n")

    return metrics_dict


# ---------------------------------------------------------------------------
# STAGE 3 — RAGAS
# ---------------------------------------------------------------------------

def stage3_ragas(results: list[dict], timestamp: str) -> dict:
    log.info("=== Stage 3: RAGAS ===")

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import llm_factory
        # Import from ragas.metrics (the stable module whose classes inherit from
        # ragas.metrics.base.Metric), NOT from ragas.metrics.collections.
        # The collections classes inherit from SimpleBaseMetric, which is a
        # different ABC — evaluate() checks isinstance(m, Metric) and raises
        # TypeError: "All metrics must be initialised metric objects" if that
        # check fails.
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        log.error(
            "Missing dependencies (%s). "
            "Run: pip install ragas datasets langchain-openai",
            exc,
        )
        return {}

    ragas_rows = []
    for item in results:
        if not item.get("response") or not item.get("retrieved_context"):
            continue
        ragas_rows.append({
            "question"    : item["query"],
            "answer"      : item["response"],
            "contexts"    : [ctx["text"] for ctx in item["retrieved_context"]],
            "ground_truth": item["gt_answer"],
        })

    if not ragas_rows:
        log.error("No valid rows for RAGAS — check that Stage 1 produced responses.")
        return {}

    dataset = Dataset.from_list(ragas_rows)
    log.info("Running RAGAS on %d items with model=%s…", len(dataset), RAGAS_MODEL)

    # LLM wrapper (LangchainLLMWrapper) — evaluate() accepts this directly.
    ragas_llm = llm_factory(model=RAGAS_MODEL)

    # Embeddings wrapper — use langchain_openai so ragas can call .embed_query().
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Bind llm / embeddings on each metric at construction time so that
    # evaluate() receives fully-initialised Metric objects.
    metrics_list = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    # Do NOT pass llm= or embeddings= to evaluate() — it is not needed when
    # the metrics are already initialised, and passing them can trigger the
    # same TypeError in some RAGAS versions.
    scores = evaluate(
        dataset=dataset,
        metrics=metrics_list,
    )

    # Safely extract metrics from RAGAS evaluation result wrapper.
    scores_dict = {}
    for metric in metrics_list:
        metric_name = metric.name
        try:
            if hasattr(scores, "get") and scores.get(metric_name) is not None:
                scores_dict[metric_name] = scores.get(metric_name)
            elif metric_name in scores:
                scores_dict[metric_name] = scores[metric_name]
        except Exception:
            pass

    if not scores_dict:
        try:
            scores_dict = {k: v for k, v in scores.items() if isinstance(k, str)}
        except Exception:
            log.warning("Could not automatically flatten RAGAS Result object keys.")

    scores_path = os.path.join(OUTPUT_DIR, f"ragas_scores_{timestamp}.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f, indent=2, ensure_ascii=False)
    log.info("RAGAS scores saved to %s", scores_path)

    try:
        df = scores.to_pandas()
        csv_path = os.path.join(OUTPUT_DIR, f"ragas_details_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        log.info("RAGAS per-question details saved to %s", csv_path)
    except Exception as e:
        log.warning("Could not save RAGAS CSV: %s", e)

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
        "timestamp"          : timestamp,
        "generator_model"    : GENERATOR_MODEL,
        "retrieval_top_k"    : RETRIEVAL_TOP_K,
        "n_questions"        : n_questions,
        "ragchecker"         : ragchecker_metrics,
        "ragas"              : ragas_scores,
    }
    path = os.path.join(OUTPUT_DIR, f"combined_report_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n──────────────────────────────────────────────────────")
    print("  COMBINED BENCHMARK REPORT")
    print(f"  Questions evaluated : {n_questions}")
    print(f"  Generator model     : {GENERATOR_MODEL}")
    print(f"  Retrieval top-k     : {RETRIEVAL_TOP_K}")
    print("──────────────────────────────────────────────────────")
    if ragas_scores:
        print("  RAGAS")
        for k, v in ragas_scores.items():
            print(f"    {k:<28} {v:.4f}")
    if ragchecker_metrics:
        print("  RAGChecker")
        for section, vals in ragchecker_metrics.items():
            print(f"    [{section}]")
            for k, v in vals.items():
                print(f"      {k:<26} {v}")
    print(f"Full report → {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Load benchmark ────────────────────────────────────────────────────
    log.info("Loading benchmark from %s", BENCHMARK_PATH)
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    benchmark: list[dict] = raw["results"] if isinstance(raw, dict) and "results" in raw else raw

    # Verify the generated dataset has the fields we need.
    required = {"query_id", "query", "gt_answer", "chunk_id"}
    sample = benchmark[0] if benchmark else {}
    missing = required - sample.keys()
    if missing:
        log.error("Benchmark is missing required fields: %s — wrong file?", missing)
        return

    # ── Handle MAX_QUESTIONS dynamically via env var or configuration ─────
    env_max = os.getenv("MAX_QUESTIONS")
    current_max = int(env_max) if env_max is not None else MAX_QUESTIONS

    if current_max is not None:
        benchmark = benchmark[:current_max]
        log.info("Capped execution to %d questions via MAX_QUESTIONS setting", current_max)

    log.info("Running benchmark on %d questions.", len(benchmark))

    # ── Load chunk text lookup (cast keys to str for cross-compatibility) ─
    log.info("Loading chunks from %s", CHUNKS_PATH)
    chunks_lookup: dict[str, dict] = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_lookup[str(c["chunk_id"])] = c
    log.info("Loaded %d chunks.", len(chunks_lookup))

    # ── Stage 1: Retrieve & Generate ─────────────────────────────────────
    results = stage1_retrieve_and_generate(benchmark, chunks_lookup, openai_client, timestamp)
    if not results:
        log.error("Stage 1 produced no results — aborting pipeline execution.")
        return

    input_path = os.path.join(OUTPUT_DIR, f"ragchecker_input_{timestamp}.json")

    # ── Stage 2: RAGChecker ───────────────────────────────────────────────
    ragchecker_metrics = {}
    if RUN_RAGCHECKER:
        ragchecker_metrics = stage2_ragchecker(input_path, timestamp)
    else:
        log.info("Stage 2 (RAGChecker) skipped — RUN_RAGCHECKER=False")

    # ── Stage 3: RAGAS ────────────────────────────────────────────────────
    ragas_scores = {}
    if RUN_RAGAS:
        ragas_scores = stage3_ragas(results, timestamp)
    else:
        log.info("Stage 3 (RAGAS) skipped — RUN_RAGAS=False")

    # ── Combined report ───────────────────────────────────────────────────
    save_combined_report(ragchecker_metrics, ragas_scores, len(results), timestamp)


if __name__ == "__main__":
    main()