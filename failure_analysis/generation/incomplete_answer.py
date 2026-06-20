"""
LLM-as-a-Judge Answer Recall (E10: Incomplete Answer)

Extracts factual claims from the ground truth and checks whether the
generated response includes all facts that were available in the context.
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


class AnswerRecallEval(TypedDict):
    gold_claims: list[str]
    included_claims: list[str]
    omitted_claims: list[str]
    total_gold: int
    total_included: int
    answer_recall: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Incomplete Answers (E10).
An Incomplete Answer occurs when the generation model fails to include facts in its final response, despite those facts being available in the retrieved context.

Your task is to compute the Answer Recall.

CRITICAL RULES FOR EXTRACTION & MATCHING:
- Step 1: Extract individual "Gold Claims" from the GOLDEN FACTS text. Break down complex sentences into atomic claims.
- Step 2: For EACH Gold Claim, first check if it is supported by the RETRIEVED CONTEXT. If the claim is NOT present in the retrieved context, SKIP it entirely — that is a retrieval failure (E4), not a generation failure (E10).
- Step 3: For the remaining Gold Claims (those that ARE in the context), check if the GENERATED RESPONSE successfully includes, answers, or conveys them. Synonyms and paraphrasing are allowed.
- Step 4: If a Gold Claim IS in the context but is missing, glossed over, or not explicitly answered in the Generated Response, classify it as OMITTED.
- Step 5: Calculate total_gold (number of context-supported claims) and total_included (number successfully written in the response).
- Step 6: Calculate answer_recall = total_included / total_gold. (If total_gold is 0, return 1.0).

Return ONLY a raw JSON object matching this exact structure, with no markdown fences, no explanations outside the JSON:
{{
  "gold_claims": ["claim 1 (in context)", "claim 2 (in context)"],
  "skipped_claims": ["claim 3 (not in retrieved context — E4 issue)"],
  "included_claims": ["claim 1"],
  "omitted_claims": ["claim 2"],
  "total_gold": 2,
  "total_included": 1,
  "answer_recall": 0.5
}}

GOLDEN FACTS (The ground truth facts):
{golden}

RETRIEVED CONTEXT (What was available to the LLM):
{retrieved}

GENERATED RESPONSE (The actual answer produced by the LLM):
{response}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_answer_recall_with_llm(response: str, golden: str, client: genai.Client, retrieved: str = "") -> AnswerRecallEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(golden=golden.strip(), response=response.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to extract claims and compute Answer Recall...", end=" ", flush=True)

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
        if data.get("total_gold", 0) == 0:
            data["answer_recall"] = 1.0
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

def display_results(result: AnswerRecallEval) -> None:
    print()
    print("=" * WRAP)
    print("  ANSWER RECALL (E10: Incomplete Answer) — LLM JUDGE REPORT")
    print("=" * WRAP)

    score = result.get("answer_recall", 0.0)
    print(f"\n{'Answer Recall Score':.<35} {_score_bar(score)}")
    print(f"{'Included Claims (In Response)':.<35} {result.get('total_included', 0)}")
    print(f"{'Total Gold Claims (Required)':.<35} {result.get('total_gold', 0)}")

    if score >= 0.99:
        label, symbol = "EXCELLENT — Response is fully complete", "✅"
    elif score >= 0.6:
        label, symbol = "WARNING — Response is partially incomplete", "⚠️ "
    else:
        label, symbol = "FAILED — Severe E10 (LLM omitted crucial facts)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  INCLUDED Claims (Successfully written in response)", result.get("included_claims", [])))
    print()
    print(_wrap_list(f"❌  OMITTED Claims (LLM forgot to mention these)", result.get("omitted_claims", [])))
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
    print("║  LLM-as-a-Judge Answer Recall — Incomplete Answer Analyzer   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    print("🏆  Provide the GOLDEN FACTS (The facts from the context that MUST be in the answer):")
    golden_text = read_multiline("")
    if not golden_text: return 1

    print()
    print("🤖  Provide the GENERATED RESPONSE (The actual output from your RAG LLM):")
    response_text = read_multiline("")
    if not response_text: return 1

    print("\n🔍  Analyzing response completeness...")
    result = evaluate_answer_recall_with_llm(response_text, golden_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())