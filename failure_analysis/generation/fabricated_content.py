"""
LLM-as-a-Judge Faithfulness (E9: Fabricated Content / Hallucinations)

Extracts factual claims from the generated response and checks whether each
claim is supported by the retrieved context or the original question.
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


class FaithfulnessEval(TypedDict):
    response_claims: list[str]
    supported_claims: list[str]
    fabricated_claims: list[str]
    total_claims: int
    total_supported: int
    faithfulness_score: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Fabricated Content (E9 - Hallucinations).
Fabricated Content occurs when the RAG system generates a response containing facts or claims that are NOT supported by the retrieved context or the original question.

Your task is to compute the Faithfulness score.

CRITICAL RULES FOR EXTRACTION & MATCHING:
- Step 1: Extract individual "Response Claims" from the GENERATED RESPONSE. A claim is a single, independent factual statement, clinical criterion, or recommendation made by the AI. Break down complex sentences into atomic claims.
- Step 2: Read the QUESTION and RETRIEVED CONTEXT. For EACH Response Claim, check if it can be directly inferred, verified, or supported using ONLY the Question and Retrieved Context. The question may contain factual information (e.g. patient details, conditions) that the response legitimately uses. Do NOT use your outside medical knowledge. If it's not in the question or context, it's not supported.
- Step 3: If neither the question nor the retrieved context contains the information to verify the claim, classify the claim as FABRICATED (Hallucination).
- Step 4: Calculate total_claims (number of claims extracted from the response) and total_supported (number of claims found in the question or context).
- Step 5: Calculate faithfulness_score = total_supported / total_claims. (If total_claims is 0, return 1.0 as there are no hallucinations).

Return ONLY a raw JSON object matching this exact structure, with no markdown fences, no explanations outside the JSON:
{{
  "response_claims": ["claim 1", "claim 2", "claim 3"],
  "supported_claims": ["claim 1", "claim 2"],
  "fabricated_claims": ["claim 3"],
  "total_claims": 3,
  "total_supported": 2,
  "faithfulness_score": 0.6667
}}

QUESTION:
{query}

GENERATED RESPONSE (Extract Claims From Here):
{response}

RETRIEVED CONTEXT (Verify Claims Against This):
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_faithfulness_with_llm(retrieved: str, response: str, client: genai.Client, query: str = "") -> FaithfulnessEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(query=query.strip(), response=response.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to extract claims and compute Faithfulness...", end=" ", flush=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_response = client.models.generate_content(
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

    raw = api_response.text.strip()
    print("done")

    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
        if data.get("total_claims", 0) == 0:
            data["faithfulness_score"] = 1.0
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

def _score_bar(score: float) -> str:
    filled = round(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {score:.4f}"

def display_results(result: FaithfulnessEval) -> None:
    print()
    print("=" * WRAP)
    print("  FAITHFULNESS (E9: Fabricated Content) — LLM JUDGE REPORT")
    print("=" * WRAP)

    score = result.get("faithfulness_score", 0.0)
    print(f"\n{'Faithfulness Score':.<35} {_score_bar(score)}")
    print(f"{'Supported Claims (Verified)':.<35} {result.get('total_supported', 0)}")
    print(f"{'Total Claims in Response':.<35} {result.get('total_claims', 0)}")

    if score >= 0.99:
        label, symbol = "PERFECT — No hallucinations detected", "✅"
    elif score >= 0.7:
        label, symbol = "WARNING — Minor fabrications or extrapolations", "⚠️ "
    else:
        label, symbol = "FAILED — Severe Hallucinations / Fabricated Content (E9)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  SUPPORTED Claims (Grounded in context)", result.get("supported_claims", [])))
    print()
    print(_wrap_list(f"❌  FABRICATED Claims (Hallucinations / Not in context)", result.get("fabricated_claims", [])))
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
    print("║  LLM-as-a-Judge Faithfulness — Hallucination Analyzer (E9)   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    print("❓  Provide the QUESTION:")
    query_text = read_multiline("")
    if not query_text: return 1

    print("\n🤖  Provide the GENERATED RESPONSE (The final output from your RAG LLM):")
    generated_text = read_multiline("")
    if not generated_text: return 1

    print()
    print("📥  Provide the RETRIEVED CONTEXT (The chunks given to the LLM):")
    retrieved_text = read_multiline("")
    if not retrieved_text: return 1

    print("\n🔍  Analyzing claims for hallucinations...")
    result = evaluate_faithfulness_with_llm(retrieved_text, generated_text, client, query=query_text)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())