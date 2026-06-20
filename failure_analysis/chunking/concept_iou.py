"""
LLM-as-a-Judge Concept-level Intersection-over-Union (IoU)

Uses Gemini to extract key concepts from both retrieved and golden context,
then computes semantic IoU to measure underchunking.
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


class ConceptEval(TypedDict):
    golden_concepts: list[str]
    retrieved_concepts: list[str]
    shared_concepts: list[str]
    missing_concepts: list[str]
    irrelevant_concepts: list[str]
    intersection_count: int
    union_count: int
    iou_score: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Underchunking.
Underchunking means a retrieved chunk is too large and contains too much irrelevant information alongside the correct answer.

Your task is to compute the Concept-level Intersection-over-Union (IoU).

CRITICAL RULE FOR GRANULARITY (CONCEPT LEVEL):
- Do NOT extract concepts at the individual entity/word level (e.g., do not split a list of specific medications, symptoms, or allergens into separate concepts).
- DO extract concepts at the THEMATIC or TOPIC level (e.g., group an entire list or sentence into a single overarching concept like "medication-related triggers" or "allergic symptoms").
- One sentence or one distinct bullet point should generally represent a maximum of ONE or TWO concepts.

INSTRUCTIONS FOR EXTRACTION:
Step 1: Extract distinct high-level/thematic clinical, medical, or lifestyle topics present in the RETRIEVED TEXT, applying the Granularity Rule above. Do not skip any sections, but ensure you group related items into broad concepts (e.g., guidelines, insurance/costs, study statistics, separate advice).
Step 2: Extract the core thematic concepts from the GOLDEN TEXT using the same level of granularity.
Step 3: Carefully compare the two lists. Identify which concepts from the Golden Text are semantically matched (synonyms allowed) inside the Retrieved list.
Step 4: Put all the remaining concepts from the Retrieved Text that did NOT match the Golden Text into "irrelevant_concepts".
Step 5: Calculate Intersection count and the total Union count.
Step 6: Calculate IoU = Intersection / Union.

Return ONLY a raw JSON object matching this exact structure, with no markdown fences, no explanations outside the JSON:
{{
  "golden_concepts": ["concept 1", "concept 2"],
  "retrieved_concepts": ["concept 2", "concept 3"],
  "shared_concepts": ["concept 2 (matches synonym X)"],
  "missing_concepts": ["concept 1"],
  "irrelevant_concepts": ["concept 3"],
  "intersection_count": 1,
  "union_count": 3,
  "iou_score": 0.3333
}}

GOLDEN TEXT:
{golden}

RETRIEVED TEXT:
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_iou_with_llm(retrieved: str, golden: str, client: genai.Client) -> ConceptEval:
    import time
    prompt = LLM_JUDGE_PROMPT.format(golden=golden.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to extract concepts and compute semantic IoU...", end=" ", flush=True)

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
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON from LLM: {e}")
        print(f"Raw response:\n{raw}")
        sys.exit(1)


def _wrap_list(label: str, words: list[str]) -> str:
    word_list = ", ".join(words) if words else "(none)"
    wrapped = textwrap.fill(word_list, width=WRAP, initial_indent="  ", subsequent_indent="  ")
    return f"{label}:\n{wrapped}"

def _iou_bar(iou: float) -> str:
    filled = round(iou * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {iou:.4f}"

def display_results(result: ConceptEval) -> None:
    print()
    print("=" * WRAP)
    print("  SEMANTIC CONCEPT-LEVEL IoU — LLM JUDGE REPORT")
    print("=" * WRAP)

    iou = result.get("iou_score", 0.0)
    print(f"\n{'IoU Score':.<30} {_iou_bar(iou)}")
    print(f"{'Intersection (shared)':.<30} {result.get('intersection_count', 0)}")
    print(f"{'Union (total unique)':.<30} {result.get('union_count', 0)}")

    if iou >= 0.7:
        label, symbol = "GOOD — High concept overlap", "✅"
    elif iou >= 0.4:
        label, symbol = "PARTIAL — Moderate overlap", "⚠️ "
    else:
        label, symbol = "POOR — Low overlap (Underchunking or Miss)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  Shared concepts (Semantically matched)", result.get("shared_concepts", [])))
    print()
    print(_wrap_list(f"➕  Only in RETRIEVED (Irrelevant / Ruis)", result.get("irrelevant_concepts", [])))
    print()
    print(_wrap_list(f"❌  Missing from RETRIEVED (Missed Golden)", result.get("missing_concepts", [])))
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
    print("║  LLM-as-a-Judge Concept IoU — Semantic RAG Analyzer          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    retrieved_text = read_multiline("📥  Paste the RETRIEVED context:")
    if not retrieved_text: return 1

    print()
    golden_text = read_multiline("🏆  Paste the GOLDEN context:")
    if not golden_text: return 1

    print("\n🔍  Analyzing...")
    result = evaluate_iou_with_llm(retrieved_text, golden_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())