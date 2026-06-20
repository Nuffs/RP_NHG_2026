"""
LLM-as-a-Judge Context Precision (Ranked) — E6: Low Ranked

Gemini beoordeelt de relevantie per chunk, waarna Python de Context
Precision Ranked formule berekent: Sum(P@k * I(k)) / Totaal_Relevante_Chunks
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import TypedDict

_env_path = Path(__file__).resolve().parent.parent.parent / "pipeline" / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai package not found.")
    print("Install it with:  pip install google-genai")
    sys.exit(1)

GEMINI_MODEL = "gemini-3.1-flash-lite"


class ChunkEval(TypedDict):
    chunk_index: int
    is_relevant: int
    reason: str

class RankedPrecisionEval(TypedDict):
    evaluations: list[ChunkEval]
    total_retrieved: int
    total_relevant: int
    ranked_context_precision: float
    unranked_precision: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for E5 (Low Relevance) and E6 (Low Ranked) errors.

Your task is to determine the relevance of each retrieved chunk INDIVIDUALLY based on the GOLDEN TEXT (Ground Truth).

CRITICAL RULES:
- Read the GOLDEN TEXT to understand the core facts, claims, and context needed to answer the user's query.
- Read the RETRIEVED CHUNKS. They are provided in a specific ranked order.
- Evaluate EACH chunk independently.
- If a chunk contains ANY information that directly supports, verifies, or is highly relevant to the Golden Text, mark it as RELEVANT (1).
- If a chunk is off-topic, pure noise, or does not help reconstruct the Golden Text, mark it as IRRELEVANT (0).
- Provide a very brief reason (max 1 sentence) for your decision to ensure accuracy.

Return ONLY a raw JSON object matching this exact structure, with no markdown fences or outside explanations:
{{
  "evaluations": [
    {{"chunk_index": 1, "is_relevant": 1, "reason": "Contains exact dosage criteria."}},
    {{"chunk_index": 2, "is_relevant": 0, "reason": "Discusses an unrelated medical condition."}},
    {{"chunk_index": 3, "is_relevant": 1, "reason": "Mentions the correct patient age range."}}
  ]
}}

GOLDEN TEXT (Ground Truth):
{golden}

RETRIEVED CHUNKS (Evaluate each in order):
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_ranked_precision(retrieved: str, golden: str, client: genai.Client) -> RankedPrecisionEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(golden=golden.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to classify chunk relevance (0 or 1)...", end=" ", flush=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            break
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"\n  ⏳ Rate limited. Waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise

    raw = response.text.strip()
    print("done")

    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        evals = data.get("evaluations", [])

        total_retrieved = len(evals)
        total_relevant = sum(1 for e in evals if e.get("is_relevant") == 1)

        if total_relevant == 0:
            ranked_cp = 0.0
            unranked_cp = 0.0
        else:
            unranked_cp = total_relevant / total_retrieved if total_retrieved > 0 else 0.0

            score_sum = 0.0
            relevant_so_far = 0

            for k, evaluation in enumerate(evals, start=1):
                is_rel = evaluation.get("is_relevant", 0)
                if is_rel == 1:
                    relevant_so_far += 1
                    p_at_k = relevant_so_far / k
                    score_sum += p_at_k

            ranked_cp = score_sum / total_relevant

        return {
            "evaluations": evals,
            "total_retrieved": total_retrieved,
            "total_relevant": total_relevant,
            "ranked_context_precision": ranked_cp,
            "unranked_precision": unranked_cp
        }

    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON from LLM: {e}")
        print(f"Raw response:\n{raw}")
        sys.exit(1)


def _metric_bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {score:.4f}"

def display_results(result: RankedPrecisionEval) -> None:
    print()
    print("=" * WRAP)
    print("  CONTEXT PRECISION (E5 & E6) — LLM JUDGE REPORT")
    print("=" * WRAP)

    ranked = result.get("ranked_context_precision", 0.0)
    unranked = result.get("unranked_precision", 0.0)

    print(f"\n{'Ranked Precision (E6)':.<30} {_metric_bar(ranked)}")
    print(f"{'Unranked Precision (E5)':.<30} {_metric_bar(unranked)}")
    print(f"{'Total Chunks Retrieved':.<30} {result.get('total_retrieved', 0)}")
    print(f"{'Total Relevant Chunks':.<30} {result.get('total_relevant', 0)}")

    print("\n🔍 DIAGNOSE:")
    if ranked >= 0.8 and unranked >= 0.8:
        print("✅ PERFECT: Hoge relevantie én perfect bovenaan gerangschikt.")
    elif unranked < 0.5 and ranked >= 0.8:
        print("⚠️ E5 (Low Relevance): Er zit veel ruis in de chunks, maar de juiste chunk stond gelukkig wel bovenaan (Rank 1).")
    elif unranked >= 0.5 and ranked < 0.6:
        print("📉 E6 (Low Ranked): Goede chunks zijn gevonden, maar de retriever plaatst ze onnodig laag in de ranking, onder de ruis.")
    else:
        print("❌ E5 + E6 COMBI: Veel irrelevante ruis, én de weinige goede chunks staan onderaan.")

    print("\n" + "-" * WRAP)
    print("  CHUNK-VOOR-CHUNK BEOORDELING (P@k):")

    relevant_so_far = 0
    for k, evaluation in enumerate(result.get("evaluations", []), start=1):
        is_rel = evaluation.get('is_relevant', 0)
        reason = evaluation.get('reason', '')

        if is_rel == 1:
            relevant_so_far += 1
            p_at_k = relevant_so_far / k
            symbol = "✅"
            calc_text = f"P@{k} = {p_at_k:.2f}"
        else:
            symbol = "❌"
            calc_text = f"Genegeerd (I=0)"

        print(f"  {symbol} Rank {k} | {calc_text}")
        wrapped_reason = textwrap.fill(reason, width=WRAP-8, initial_indent="        ", subsequent_indent="        ")
        print(wrapped_reason)
    print("-" * WRAP)


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("  (Paste your text. When done, type END on a new line and press Enter)\n")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def main() -> int:
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  LLM-as-a-Judge Ranked Precision — E6 Low Ranked Analyzer    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    print("🏆  Provide the GOLDEN ANSWER (The ground truth facts we need):")
    golden_text = read_multiline("")
    if not golden_text: return 1

    print()
    print("📥  Provide the RETRIEVED CHUNKS IN ORDER (e.g. [1] Chunk A, [2] Chunk B):")
    retrieved_text = read_multiline("")
    if not retrieved_text: return 1

    print("\n🔍  Analyzing chunk sequence...")
    result = evaluate_ranked_precision(retrieved_text, golden_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())