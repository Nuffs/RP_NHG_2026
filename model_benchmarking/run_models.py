"""
model_benchmarking/run_models.py

Runs 6 LLMs against the QA benchmark (qa_few_shot.json).

Retrieval flow (Leander's pipeline):
  question → Gemini embed → cosine sim over embeddings.json → top-K chunk IDs
           → look up text in nhg_subset_guidelines.jsonl → retrieved_context

Generation flow (this file):
  retrieved_context + question → 6 LLMs (concurrent) → answers

Outputs:
  model_benchmarking/results/answers_<model>.json  (RAGChecker format)
  model_benchmarking/results/metrics.csv           (latency + token counts)
"""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

import anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Make the repo root importable so we can reach pipeline/
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.query import embed_query, retrieve_top_chunks  # noqa: E402

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH      = Path("factual_benchmark/results/qa_final_dataset.json")
GUIDELINES_PATH = Path("pipeline/nhg_subset_guidelines.jsonl")
EMBEDDINGS_PATH = Path("pipeline/embeddings.json")
RESULTS_DIR     = Path("model_benchmarking/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Max concurrent LLM requests per model
CONCURRENCY = 5
# Number of chunks to retrieve per question
TOP_K = 5

# ── Pricing dictionary (blended price per 1 million tokens, USD) ───────────────
# Blended = single rate applied to input + output + reasoning tokens combined.
PRICE_PER_1M: dict[str, float] = {
    "gpt-5.5":         11.25,   # gpt-5.5-turbo
    "gpt-5.4":          5.63,   # gpt-5.4-turbo
    "claude-opus-4-7": 10.00,   # claude-4-7-opus-latest
    "kimi-k2.6":        1.70,   # moonshot-v1-32k
    "deepseek-v4-pro":  2.20,   # deepseek-reasoner
    "glm-5.1":          2.10,   # zhipuai/glm-5.1
}

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS: list[dict] = [
    {
        "name": "gpt-5.5",
        "type": "openai_responses",
        "model_id": "gpt-5.5",
    },
    {
        "name": "gpt-5.4",
        "type": "openai_responses",
        "model_id": "gpt-5.4",
    },
    {
        "name": "claude-opus-4-7",
        "type": "anthropic",
        "model_id": "claude-opus-4-7",
    },
    {
        "name": "kimi-k2.6",
        "type": "openai_chat",
        "model_id": "kimi-k2.6",
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "KIMI_API_KEY",
    },
    {
        "name": "deepseek-v4-pro",
        "type": "deepseek",
        "model_id": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    {
        "name": "glm-5.1",
        "type": "openrouter_stream",
        "model_id": "z-ai/glm-5.1",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
]


# ── Retrieval helpers ──────────────────────────────────────────────────────────

def load_chunk_lookup(path: Path) -> dict[str, dict]:
    """
    Build a {chunk_id: chunk_obj} index from nhg_subset_guidelines.jsonl.
    Called once at startup so JSONL is only read once regardless of query count.
    """
    lookup: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid = chunk.get("chunk_id")
            if cid:
                lookup[cid] = chunk
    return lookup


def load_embedding_index(path: Path) -> list[dict]:
    """Load pre-computed embeddings produced by pipeline/embed.py."""
    raw = path.read_text(encoding="utf-8").strip()
    return json.loads(raw) if raw else []


def get_retrieved_context(
    question: str,
    chunk_lookup: dict[str, dict],
    embedding_index: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run Leander's retrieval pipeline for a single question.

    Returns a list of RAGChecker-compatible dicts:
        [{"doc_id": chunk_id, "text": markdown_text}, ...]

    Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
    """
    query_embedding = embed_query(question)
    top_chunks = retrieve_top_chunks(query_embedding, embedding_index, top_k=top_k)

    results: list[dict] = []
    for item in top_chunks:
        cid = item["chunk_id"]
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            print(f"  \u26a0 chunk_id {cid!r} not found in guidelines JSONL \u2014 skipping")
            continue
        results.append({"doc_id": cid, "text": chunk["text"]})
    return results


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(retrieved_context: list[dict], question: str) -> str:
    """
    Concatenate retrieved chunks and form the LLM prompt.
    Each chunk is separated by a divider so the model can distinguish sources.
    """
    context_block = "\n\n---\n\n".join(c["text"] for c in retrieved_context)
    return (
        f"Based on the following retrieved medical guidelines:\n\n"
        f"{context_block}\n\n"
        f"Answer this question: {question}"
    )


# ── Retry decorator factory ────────────────────────────────────────────────────

def _retry():
    """Exponential back-off: 2 s -> 4 s -> 8 s ... up to 60 s, max 4 attempts."""
    return retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )


# ── API call implementations ───────────────────────────────────────────────────

@_retry()
async def _call_openai_responses(
    client: AsyncOpenAI, model_id: str, prompt: str
) -> dict:
    """OpenAI Responses API (gpt-5.5 / gpt-5.4) with extended reasoning."""
    t0 = time.perf_counter()
    response = await client.responses.create(
        model=model_id,
        reasoning={"effort": "xhigh"},
        input=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0

    usage = response.usage
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0

    return {
        "answer": response.output_text,
        "latency": latency,
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "reasoning_tokens": reasoning_tokens,
    }


@_retry()
async def _call_anthropic(
    client: anthropic.AsyncAnthropic, model_id: str, prompt: str
) -> dict:
    """Anthropic Messages API (claude-opus-4-7) with adaptive thinking."""
    t0 = time.perf_counter()
    response = await client.messages.create(
        model=model_id,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0

    answer = ""
    for block in response.content:
        if block.type == "text":
            answer = block.text
            break

    usage = response.usage
    return {
        "answer": answer,
        "latency": latency,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": 0,  # bundled into output_tokens by Anthropic
    }


@_retry()
async def _call_openai_chat(
    client: AsyncOpenAI, model_id: str, prompt: str
) -> dict:
    """Standard OpenAI-compatible chat completions (Kimi K2.6)."""
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0

    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0

    return {
        "answer": response.choices[0].message.content or "",
        "latency": latency,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


@_retry()
async def _call_deepseek(
    client: AsyncOpenAI, model_id: str, prompt: str
) -> dict:
    """DeepSeek V4 Pro via DeepSeek API with max reasoning effort."""
    t0 = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning_effort": "max", "thinking": {"type": "enabled"}},
    )
    latency = time.perf_counter() - t0

    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0

    return {
        "answer": response.choices[0].message.content or "",
        "latency": latency,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


@_retry()
async def _call_openrouter_stream(
    client: AsyncOpenAI, model_id: str, prompt: str
) -> dict:
    """GLM-5.1 via OpenRouter using streaming (required by provider)."""
    t0 = time.perf_counter()

    answer_parts: list[str] = []
    input_tokens = output_tokens = reasoning_tokens = 0

    stream = await client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            answer_parts.append(chunk.choices[0].delta.content)

        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            reasoning_tokens = getattr(usage, "reasoningTokens", 0)

    latency = time.perf_counter() - t0
    return {
        "answer": "".join(answer_parts),
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


# ── Dispatcher ─────────────────────────────────────────────────────────────────

async def _dispatch(model_cfg: dict, clients: dict, prompt: str) -> dict:
    mtype = model_cfg["type"]
    mid   = model_cfg["model_id"]
    mname = model_cfg["name"]

    if mtype == "openai_responses":
        return await _call_openai_responses(clients["openai"], mid, prompt)
    if mtype == "anthropic":
        return await _call_anthropic(clients["anthropic"], mid, prompt)
    if mtype == "openai_chat":
        return await _call_openai_chat(clients[mname], mid, prompt)
    if mtype == "deepseek":
        return await _call_deepseek(clients[mname], mid, prompt)
    if mtype == "openrouter_stream":
        return await _call_openrouter_stream(clients[mname], mid, prompt)

    raise ValueError(f"Unknown model type: {mtype!r}")


# ── Per-model runner ───────────────────────────────────────────────────────────

async def run_model(
    model_cfg: dict,
    items: list[dict],
    clients: dict,
    metrics_path: "Path | None" = None,
    answers_path: "Path | None" = None,
) -> tuple[list[dict], list[dict]]:
    """
    Process all QA items concurrently for one model.

    Each item must already contain a "retrieved_context" key (pre-computed in
    main() so the retrieval cost is paid once, not once per model).

    metrics_path  - if provided, each metrics row is appended immediately
                    (file opened/closed per write so it's never locked).
    answers_path  - if provided, the RAGChecker JSON is rewritten after every
                    completed question (incremental safety).

    Returns:
        ragchecker_results  - list in checking_inputs.json format
        metrics_rows        - list for metrics.csv
    """
    sem   = asyncio.Semaphore(CONCURRENCY)
    name  = model_cfg["name"]
    total = len(items)

    async def _one_with_item(item: dict) -> tuple[dict, dict]:
        """Returns (item, api_result); raises on unrecoverable error."""
        prompt = build_prompt(item["retrieved_context"], item["question"])
        async with sem:
            result = await _dispatch(model_cfg, clients, prompt)
        return item, result

    tasks = [_one_with_item(item) for item in items]

    ragchecker_results: list[dict] = []
    metrics_rows: list[dict]       = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        try:
            item, outcome = await coro
        except Exception as exc:
            completed += 1
            print(f"  [FAIL] [{name}] ({completed}/{total}) error: {exc}")
            continue

        completed += 1

        ragchecker_results.append(
            {
                "query_id": item["chunk_id"],
                "query": item["question"],
                "gt_answer": item["answer"],
                "response": outcome["answer"],
                "retrieved_context": item["retrieved_context"],
            }
        )

        latency          = outcome["latency"]
        input_tokens     = outcome["input_tokens"]
        output_tokens    = outcome["output_tokens"]
        reasoning_tokens = outcome["reasoning_tokens"]
        total_tokens     = input_tokens + output_tokens + reasoning_tokens

        inference_speed_tps = output_tokens / latency if latency > 0 else 0.0
        blended_price = PRICE_PER_1M.get(name, 0.0)
        cost_dollars  = total_tokens * (blended_price / 1_000_000)

        row = {
            "model_name": name,
            "query_id": item["chunk_id"],
            "latency_seconds": round(latency, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "inference_speed_tps": round(inference_speed_tps, 2),
            "cost_dollars": f"{cost_dollars:.5f}",
        }
        metrics_rows.append(row)

        # ── Incremental CSV append (open/close so the file is never locked) ──
        if metrics_path is not None:
            with open(metrics_path, "a", newline="", encoding="utf-8") as _f:
                csv.DictWriter(_f, fieldnames=list(row.keys())).writerow(row)

        # ── Incremental JSON rewrite ───────────────────────────────────────
        if answers_path is not None:
            with open(answers_path, "w", encoding="utf-8") as jf:
                json.dump({"results": ragchecker_results}, jf,
                          ensure_ascii=False, indent=2)

        print(f"  [OK] [{name}] ({completed}/{total}) {item['chunk_id']} "
              f"| {round(latency,1)}s | {output_tokens} tok")

    return ragchecker_results, metrics_rows


# ── Client factory ─────────────────────────────────────────────────────────────

def build_clients() -> dict:
    clients: dict = {
        "openai": AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "anthropic": anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    }
    for cfg in MODELS:
        if cfg["type"] in ("openai_chat", "deepseek", "openrouter_stream"):
            clients[cfg["name"]] = AsyncOpenAI(
                api_key=os.getenv(cfg["api_key_env"]),
                base_url=cfg["base_url"],
            )
    return clients


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── 1. Load benchmark questions ────────────────────────────────────────────
    with open(INPUT_PATH, encoding="utf-8") as f:
        items: list[dict] = json.load(f)
    print(f"Loaded {len(items)} QA items from {INPUT_PATH}")

    # ── Optional filter: restrict to a single guideline topic ─────────────────
    FILTER_TOPIC: str | None = "astma_bij_volwassenen"   # set None for full run
    if FILTER_TOPIC:
        items = [i for i in items if i["chunk_id"].startswith(FILTER_TOPIC)]
        print(f"Filtered to '{FILTER_TOPIC}': {len(items)} items")
    print()

    # ── 2. Load Leander's retrieval artefacts (once) ───────────────────────────
    print("Loading retrieval index...")
    chunk_lookup     = load_chunk_lookup(GUIDELINES_PATH)
    embedding_index  = load_embedding_index(EMBEDDINGS_PATH)
    print(f"  {len(chunk_lookup)} chunks | {len(embedding_index)} embeddings\n")

    # ── 3. Pre-compute retrieved context for every question ────────────────────
    # Retrieval is shared across all 6 models - run it once here, not 6x.
    print(f"Retrieving top-{TOP_K} chunks per question (Gemini embed)...")
    for i, item in enumerate(items, 1):
        ctx = get_retrieved_context(
            item["question"], chunk_lookup, embedding_index, top_k=TOP_K
        )
        item["retrieved_context"] = ctx
        print(f"  [{i:02d}/{len(items)}] {item['chunk_id']} -> {len(ctx)} chunks retrieved")
    print()

    # ── 4. Build LLM clients ───────────────────────────────────────────────────
    clients = build_clients()

    # ── 5. Open metrics CSV once (incremental writes happen inside run_model) ──
    metrics_path = RESULTS_DIR / "metrics.csv"
    fieldnames   = [
        "model_name", "query_id", "latency_seconds",
        "input_tokens", "output_tokens", "reasoning_tokens",
        "inference_speed_tps", "cost_dollars",
    ]
    all_metrics: list[dict] = []

    # Write header once; rows are appended incrementally inside run_model.
    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_f:
        csv.DictWriter(csv_f, fieldnames=fieldnames).writeheader()

    # ── 6. Run each model (generation step) ───────────────────────────────────
    for model_cfg in MODELS:
        name      = model_cfg["name"]
        safe_name = name.replace("/", "-").replace(".", "")
        out_path  = RESULTS_DIR / f"answers_{safe_name}.json"

        print(f"--- {name} ---")
        ragchecker_results, metrics_rows = await run_model(
            model_cfg, items, clients,
            metrics_path=metrics_path,
            answers_path=out_path,
        )
        all_metrics.extend(metrics_rows)
        print(f"  -> {len(ragchecker_results)} answers saved to {out_path}\n")

    print(f"Metrics saved -> {metrics_path}  ({len(all_metrics)} rows total)")

    # ── 7. Export metrics to Excel ─────────────────────────────────────────────
    excel_path = RESULTS_DIR / "metrics.xlsx"
    df = pd.read_csv(metrics_path)
    df.to_excel(excel_path, index=False)
    print(f"Excel saved  -> {excel_path}")


if __name__ == "__main__":
    asyncio.run(main())
