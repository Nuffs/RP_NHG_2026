"""
LLM-as-a-Judge Context Recall (E4: Missed Retrieval)

Extracts factual claims from the golden text and checks whether each claim
is supported by the retrieved context.
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


class RecallEval(TypedDict):
    gold_claims: list[str]
    supported_claims: list[str]
    missed_claims: list[str]
    total_gold: int
    total_supported: int
    context_recall: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Missed Retrieval (E4).
Missed Retrieval occurs when the retrieval engine fails to fetch the necessary facts to fully answer the query.

Your task is to compute Context Recall.

CRITICAL RULES FOR EXTRACTION & MATCHING:
- Step 1: Extract individual "Gold Claims" from the GOLDEN TEXT. A claim is a single, independent factual statement, clinical criterion, or recommendation (e.g., "The starting dose for bisoprolol is 1.25 mg", or "Spirometry should be repeated if the z-score is between -1.96 and -1.64"). Break down complex sentences into atomic claims.
- Step 2: Read the RETRIEVED TEXT. For EACH Gold Claim, check if it can be fully inferred, verified, or supported using ONLY the Retrieved Text. Synonyms and paraphrasing are allowed, but the core fact must be present.
- Step 3: If the retrieved text lacks the information to verify the claim, classify the claim as MISSED.
- Step 4: Calculate total_gold (number of claims extracted) and total_supported (number of claims found in the context).
- Step 5: Calculate context_recall = total_supported / total_gold. (If total_gold is 0, return 0.0).

Return ONLY a raw JSON object matching this exact structure, with no markdown fences, no explanations outside the JSON:
{{
  "gold_claims": ["claim 1", "claim 2", "claim 3"],
  "supported_claims": ["claim 1", "claim 2"],
  "missed_claims": ["claim 3"],
  "total_gold": 3,
  "total_supported": 2,
  "context_recall": 0.6667
}}

GOLDEN TEXT (Extract Truth From Here):
{golden}

RETRIEVED TEXT (Evaluate Against This):
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_context_recall_with_llm(retrieved: str, golden: str, client: genai.Client) -> RecallEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(golden=golden.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to extract claims and compute Context Recall...", end=" ", flush=True)

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
        if data.get("total_gold", 0) == 0:
            data["context_recall"] = 0.0
        return data
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON from LLM: {e}")
        print(f"Raw response:\n{raw}")
        sys.exit(1)


def _wrap_list(label: str, claims: list[str]) -> str:
    if not claims:
        return f"{label}:\n  (none)"

    formatted_claims = []
    for claim in claims:
        wrapped = textwrap.fill(claim, width=WRAP-4, initial_indent="  - ", subsequent_indent="    ")
        formatted_claims.append(wrapped)

    return f"{label}:\n" + "\n".join(formatted_claims)

def _recall_bar(recall: float) -> str:
    filled = round(recall * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {recall:.4f}"

def display_results(result: RecallEval) -> None:
    print()
    print("=" * WRAP)
    print("  CONTEXT RECALL (E4: Missed Retrieval) — LLM JUDGE REPORT")
    print("=" * WRAP)

    recall = result.get("context_recall", 0.0)
    print(f"\n{'Context Recall':.<35} {_recall_bar(recall)}")
    print(f"{'Supported Claims (Found)':.<35} {result.get('total_supported', 0)}")
    print(f"{'Total Gold Claims (Required)':.<35} {result.get('total_gold', 0)}")

    if recall >= 0.9:
        label, symbol = "EXCELLENT — Retrieval is complete", "✅"
    elif recall >= 0.5:
        label, symbol = "PARTIAL — Missing some key information", "⚠️ "
    else:
        label, symbol = "POOR — Severe Missed Retrieval (E4)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  SUPPORTED Claims (Found in retrieved text)", result.get("supported_claims", [])))
    print()
    print(_wrap_list(f"❌  MISSED Claims (Absent from retrieved text)", result.get("missed_claims", [])))
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
    print("║  LLM-as-a-Judge Context Recall — Missed Retrieval Analyzer   ║")
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
    print("📥  Provide the RETRIEVED CONTEXT (The chunk(s) fetched by RAG):")
    retrieved_text = read_multiline("")
    if not retrieved_text: return 1

    print("\n🔍  Analyzing claims...")
    result = evaluate_context_recall_with_llm(retrieved_text, golden_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())