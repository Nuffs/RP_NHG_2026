"""
LLM-as-a-Judge Misinterpretation Rate (E11: Distorted Meaning)

Evaluates claims from the generated response against the retrieved context,
looking for claims where the LLM distorted, reversed, or misunderstood
the meaning.
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


class MisinterpretationEval(TypedDict):
    accurate_claims: list[str]
    misinterpreted_claims: list[str]
    total_evaluated: int
    total_misinterpreted: int
    misinterpretation_rate: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Misinterpretation (E11).
Misinterpretation occurs when the generator extracts information from the retrieved context but distorts the meaning, reverses the logic, exaggerates, or misses critical nuances (e.g., confusing a side effect for an indication, or ignoring a "NOT" condition).

Your task is to compute the Misinterpretation Rate.

CRITICAL RULES FOR EXTRACTION & MATCHING:
- Step 1: Extract individual claims from the GENERATED RESPONSE.
- Step 2: Compare each claim to the RETRIEVED CONTEXT to see if the LLM attempted to use that context.
  * If a claim is completely fabricated and has NO basis in the text whatsoever (pure hallucination), IGNORE IT for this specific metric (that is an E9 error). We only care about claims where the LLM tried to interpret the provided text.
- Step 3: For the claims that are based on the context, classify them into two buckets:
  * ACCURATE: The claim faithfully and correctly reflects the meaning of the context.
  * MISINTERPRETED: The claim is based on the context, but the meaning is distorted, factually flawed, or logically reversed.
- Step 4: Calculate total_evaluated (Accurate + Misinterpreted). This is the denominator (Supported Claims).
- Step 5: Calculate total_misinterpreted. This is the numerator.
- Step 6: Calculate misinterpretation_rate = total_misinterpreted / total_evaluated. (If total_evaluated is 0, return 0.0).

Return ONLY a raw JSON object matching this exact structure, with no markdown fences, no explanations outside the JSON:
{{
  "accurate_claims": ["The starting dose is 1.25 mg (Correctly interpreted)"],
  "misinterpreted_claims": ["Bisoprolol is recommended for asthma patients (Distorted: context said it is contraindicated)"],
  "total_evaluated": 2,
  "total_misinterpreted": 1,
  "misinterpretation_rate": 0.5
}}

RETRIEVED CONTEXT (The ground truth source material):
{retrieved}

GENERATED RESPONSE (The output to evaluate for distorted logic):
{response}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_misinterpretation_with_llm(retrieved: str, response: str, client: genai.Client) -> MisinterpretationEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(retrieved=retrieved.strip(), response=response.strip())

    print("  → Asking Gemini to analyze claims for logical distortion (E11)...", end=" ", flush=True)

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
        if data.get("total_evaluated", 0) == 0:
            data["misinterpretation_rate"] = 0.0
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

def _error_bar(rate: float) -> str:
    filled = round(rate * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {rate:.4f}"

def display_results(result: MisinterpretationEval) -> None:
    print()
    print("=" * WRAP)
    print("  MISINTERPRETATION RATE (E11) — LLM JUDGE REPORT")
    print("=" * WRAP)

    rate = result.get("misinterpretation_rate", 0.0)

    print(f"\n{'Misinterpretation Rate':.<35} {_error_bar(rate)}")
    print(f"{'Misinterpreted Claims':.<35} {result.get('total_misinterpreted', 0)}")
    print(f"{'Total Context-Derived Claims':.<35} {result.get('total_evaluated', 0)}")

    if rate == 0.0:
        label, symbol = "PERFECT — All claims accurately reflect the context", "✅"
    elif rate <= 0.2:
        label, symbol = "WARNING — Minor distortions detected", "⚠️ "
    else:
        label, symbol = "FAILED — Severe Misinterpretation of context (E11)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  ACCURATE Claims (Correctly understood by LLM)", result.get("accurate_claims", [])))
    print()
    print(_wrap_list(f"❌  MISINTERPRETED Claims (Meaning distorted / reversed)", result.get("misinterpreted_claims", [])))
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
    print("║  LLM-as-a-Judge Misinterpretation Analyzer (E11)             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    print("📥  Provide the RETRIEVED CONTEXT (The text the LLM had to read):")
    retrieved_text = read_multiline("")
    if not retrieved_text: return 1

    print()
    print("🤖  Provide the GENERATED RESPONSE (The output produced by the LLM):")
    response_text = read_multiline("")
    if not response_text: return 1

    print("\n🔍  Analyzing claims for logical distortion...")
    result = evaluate_misinterpretation_with_llm(retrieved_text, response_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())