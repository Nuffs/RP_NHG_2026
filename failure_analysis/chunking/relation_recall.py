"""
LLM-as-a-Judge Relation Recall (E3: Context Mismatch)

Uses Gemini to extract semantic relationships (triplets) from the golden text
and question, then checks if those relationships are preserved within
individual chunk boundaries in the retrieved context.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
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


class RelationEval(TypedDict):
    gold_triplets: list[str]
    preserved_triplets: list[str]
    broken_triplets: list[str]
    gold_count: int
    preserved_count: int
    relation_recall: float

LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating an AI Retrieval-Augmented Generation (RAG) system for Context Mismatch (E3).
Context Mismatch occurs when chunk boundaries break logical relationships between entities and their attributes.

Your task is to compute the Relation Recall.

CRITICAL RULES FOR CHUNK BOUNDARIES:
- The Retrieved Text is formatted with clear <CHUNK_X> </CHUNK_X> tags.
- A relationship is considered "PRESERVED" if the core semantic meaning is fully intact, logically implied, or clear via synonyms and context WITHIN A SINGLE CHUNK (e.g., if a heading covers a list of items, the relationship between the heading and the items is preserved).
- If Entity 1 is inside <CHUNK_1> and Entity 2 / Value is inside <CHUNK_2>, the relationship is severed by the chunk boundary. You MUST classify this triplet as "BROKEN". The LLM is not allowed to stitch chunks together to form an answer.

INSTRUCTIONS:
- Step 1: Read the QUESTION and GOLDEN TEXT. Extract core semantic relations as concise triplets: (Entity 1, Relation/Attribute, Entity 2 / Value). Focus on high-level clinical facts.
  *CRITICAL:* Use the QUESTION to resolve any pronouns (like "he", "it") or missing subjects in the GOLDEN TEXT so your triplets contain explicit names/entities.
- Step 2: Evaluate the RETRIEVED CHUNKS. For EACH triplet, check if there is at least ONE individual chunk that independently and semantically satisfies the relationship. Allow flexible phrasing, clinical implications, and synonyms.
- Step 3: If no single chunk contains the complete relationship, the triplet is BROKEN.
- Step 4: Calculate gold_count and preserved_count.
- Step 5: Calculate relation_recall = preserved_count / gold_count.

Return ONLY a raw JSON object:
{{
  "gold_triplets": ["(EntityA, relation, EntityB)"],
  "preserved_triplets": ["(EntityA, relation, EntityB)"],
  "broken_triplets": ["(EntityC, relation, EntityD)"],
  "gold_count": 2,
  "preserved_count": 1,
  "relation_recall": 0.5
}}

QUESTION:
{query}

GOLDEN TEXT (Extract Truth From Here):
{golden}

RETRIEVED CHUNKS (Evaluate Against These, respecting boundaries):
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_relation_recall_with_llm(query: str, retrieved: str, golden: str, client: genai.Client) -> RelationEval:
    prompt = LLM_JUDGE_PROMPT.format(
        query=query.strip(),
        golden=golden.strip(),
        retrieved=retrieved.strip()
    )

    print("  → Asking Gemini to extract triplets and compute Relation Recall...", end=" ", flush=True)

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
        if data.get("gold_count", 0) == 0:
            data["relation_recall"] = 0.0
        return data
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON from LLM: {e}")
        print(f"Raw response:\n{raw}")
        sys.exit(1)


def _wrap_list(label: str, triplets: list[str]) -> str:
    triplet_list = "\n  ".join(triplets) if triplets else "(none)"
    return f"{label}:\n  {triplet_list}"

def _recall_bar(recall: float) -> str:
    filled = round(recall * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}]  {recall:.4f}"

def display_results(result: RelationEval) -> None:
    print()
    print("=" * WRAP)
    print("  SEMANTIC RELATION RECALL (E3) — LLM JUDGE REPORT")
    print("=" * WRAP)

    recall = result.get("relation_recall", 0.0)
    print(f"\n{'Relation Recall':.<35} {_recall_bar(recall)}")
    print(f"{'Preserved Triplets (Intact)':.<35} {result.get('preserved_count', 0)}")
    print(f"{'Total Gold Triplets (Required)':.<35} {result.get('gold_count', 0)}")

    if recall >= 0.8:
        label, symbol = "GOOD — Context boundaries intact", "✅"
    elif recall >= 0.4:
        label, symbol = "PARTIAL — Some relations severed", "⚠️ "
    else:
        label, symbol = "POOR — Severe Context Mismatch (E3)", "❌"
    print(f"\n{symbol}  {label}")

    print("\n" + "-" * WRAP)
    print(_wrap_list(f"✅  PRESERVED Relations (Intact in context)", result.get("preserved_triplets", [])))
    print()
    print(_wrap_list(f"❌  BROKEN / MISSING Relations (Context Mismatch)", result.get("broken_triplets", [])))
    print("-" * WRAP)


def read_multiline(prompt: str) -> str:
    if prompt:
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

def format_chunks(raw_retrieved_text: str) -> str:
    """Split pasted text into chunks and wrap in XML tags."""
    raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', raw_retrieved_text) if c.strip()]

    formatted = ""
    for i, chunk in enumerate(raw_chunks, 1):
        formatted += f"<CHUNK_{i}>\n{chunk}\n</CHUNK_{i}>\n\n"
    return formatted.strip()


def main() -> int:
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  LLM-as-a-Judge Relation Recall — Context Mismatch Analyzer  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        return 1

    client = genai.Client(api_key=api_key)

    print("❓  Provide the QUESTION:")
    query_text = read_multiline("")
    if not query_text:
        return 1

    print("\n🏆  Provide the GOLDEN ANSWER (The ground truth facts we need):")
    golden_text = read_multiline("")
    if not golden_text:
        return 1

    print()
    print("📥  Provide the RETRIEVED CONTEXT (Separate chunks with an empty line):")
    raw_retrieved = read_multiline("")
    if not raw_retrieved:
        return 1

    formatted_retrieved = format_chunks(raw_retrieved)

    print("\n🔍  Analyzing semantic triplets...")
    result = evaluate_relation_recall_with_llm(query_text, formatted_retrieved, golden_text, client)
    display_results(result)
    return 0

if __name__ == "__main__":
    sys.exit(main())