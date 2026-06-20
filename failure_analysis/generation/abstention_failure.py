"""
LLM-as-a-Judge Abstention Failure (E12: Insufficient Context Detection)

Evaluates whether the retrieved context contains sufficient information
to answer the question. If insufficient, the system should have abstained.
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


class AbstentionEval(TypedDict):
    verdict: str
    confidence: float
    reasoning: str
    missing_information: list[str]
    abstention_score: float


LLM_JUDGE_PROMPT = """\
You are an expert clinical NLP judge evaluating a Retrieval-Augmented Generation (RAG) system for Abstention Failure.

Abstention Failure occurs when the RAG system answers a question even though the retrieved context does NOT contain sufficient information to produce a correct, complete answer. In such cases, the system should have abstained (e.g., responded with "I don't have enough information to answer this question").

Your task: Given a QUESTION and the RETRIEVED CONTEXT (the chunks the system had available), determine whether the context contains enough information to properly answer the question.

EVALUATION CRITERIA:
- "sufficient": The retrieved context contains the key facts, clinical guidelines, or information needed to answer the question correctly. Minor gaps are acceptable if the core answer can be derived.
- "insufficient": The retrieved context is missing critical information needed to answer the question. The context may be completely irrelevant, only tangentially related, or missing the specific clinical detail the question asks about.

IMPORTANT NOTES:
- Judge ONLY whether the context has enough information. Do NOT judge the quality of any response.
- A context that contains related but not directly answering information is "insufficient".
- If the context covers the topic but not the specific detail asked about, it is "insufficient".
- If no chunks were retrieved (empty context), it is automatically "insufficient".

Return ONLY a raw JSON object matching this exact structure, with no markdown fences:
{{
  "verdict": "sufficient",
  "confidence": 0.85,
  "reasoning": "The context contains the specific guideline about X, which directly answers the question.",
  "missing_information": [],
  "abstention_score": 1.0
}}

Or for insufficient context:
{{
  "verdict": "insufficient",
  "confidence": 0.9,
  "reasoning": "The context discusses Y but does not contain the specific recommendation about X that the question asks for.",
  "missing_information": ["specific recommendation about X", "dosage guidelines for Z"],
  "abstention_score": 0.0
}}

QUESTION:
{query}

RETRIEVED CONTEXT (The chunks the system had available):
{retrieved}
"""

WRAP = 80
BAR_WIDTH = 40


def evaluate_abstention_with_llm(retrieved: str, client: genai.Client, query: str = "") -> AbstentionEval:
    import time

    if not retrieved or not retrieved.strip():
        return {
            "verdict": "insufficient",
            "confidence": 1.0,
            "reasoning": "No chunks were retrieved. The system has no context to answer the question.",
            "missing_information": ["all information — no context retrieved"],
            "abstention_score": 0.0,
        }

    prompt = LLM_JUDGE_PROMPT.format(query=query.strip(), retrieved=retrieved.strip())

    print("  → Asking Gemini to evaluate context sufficiency...", end=" ", flush=True)

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
        verdict = data.get("verdict", "").lower().strip()
        if verdict not in ("sufficient", "insufficient"):
            verdict = "insufficient" if data.get("abstention_score", 1.0) < 0.5 else "sufficient"
            data["verdict"] = verdict
        if "abstention_score" not in data:
            data["abstention_score"] = 1.0 if verdict == "sufficient" else 0.0
        return data
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON from LLM: {e}")
        print(f"Raw response:\n{raw}")
        sys.exit(1)


def display_results(result: AbstentionEval) -> None:
    print()
    print("=" * WRAP)
    print("  ABSTENTION FAILURE (E12) — LLM JUDGE REPORT")
    print("=" * WRAP)

    score = result.get("abstention_score", 0.0)
    verdict = result.get("verdict", "unknown")
    confidence = result.get("confidence", 0.0)

    filled = round(score * BAR_WIDTH)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    print(f"\n{'Abstention Score':.<35} [{bar}]  {score:.4f}")
    print(f"{'Verdict':.<35} {verdict.upper()}")
    print(f"{'Confidence':.<35} {confidence:.2f}")

    if verdict == "sufficient":
        label, symbol = "PASS — Context has sufficient information", "✅"
    else:
        label, symbol = "FAIL — Abstention Failure: model should have abstained", "❌"
    print(f"\n{symbol}  {label}")

    reasoning = result.get("reasoning", "")
    if reasoning:
        print(f"\n{'Reasoning':.<35}")
        for line in textwrap.wrap(reasoning, width=WRAP - 4):
            print(f"    {line}")

    missing = result.get("missing_information", [])
    if missing:
        print(f"\n{'Missing Information':.<35}")
        for item in missing:
            wrapped = textwrap.fill(item, width=WRAP - 6, initial_indent="  - ", subsequent_indent="    ")
            print(wrapped)

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


def main() -> int:
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  LLM-as-a-Judge Abstention Failure Analyzer (E12)          ║")
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

    print("\n📥  Provide the RETRIEVED CONTEXT (The chunks given to the LLM):")
    retrieved_text = read_multiline("")

    print("\n🔍  Evaluating context sufficiency...")
    result = evaluate_abstention_with_llm(retrieved_text, client, query=query_text)
    display_results(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
