"""
ragchecker/main.py

Evaluate one model's answers against the RAGChecker framework.

Usage:
    python ragchecker/main.py model_benchmarking/results/answers_claude-opus-4-7.json
    python ragchecker/main.py model_benchmarking/results/answers_gpt-55.json --model gpt-4o
    python ragchecker/main.py --list          # show all available answer files

Outputs:
    ragchecker/results/<model_name>_ragchecker.json   (full per-query scores)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 output so Dutch characters (é, ë, ï …) don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from ragchecker import RAGChecker, RAGResults
from ragchecker.metrics import all_metrics

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ANSWERS_DIR  = Path("model_benchmarking/results")
RESULTS_DIR  = Path("ragchecker/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default LLM used by RAGChecker as extractor + checker (via litellm / OpenAI)
# gpt-5.5 and other reasoning models are NOT compatible — they reject
# the non-default temperature that RAGChecker hardcodes internally.
# gpt-4o-mini is accurate, fast, cheap, and fully compatible.
DEFAULT_MODEL = "openai/gpt-4o-mini"


# ── Helpers ────────────────────────────────────────────────────────────────────

def list_answer_files() -> None:
    files = sorted(ANSWERS_DIR.glob("answers_*.json"))
    if not files:
        print("No answer files found in", ANSWERS_DIR)
        return
    print("Available model answer files:")
    for f in files:
        print(f"  {f}")


def derive_model_name(path: Path) -> str:
    """answers_claude-opus-4-7.json  ->  claude-opus-4-7"""
    return path.stem.removeprefix("answers_")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGChecker on a single model's answer file."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to answers_<model>.json (e.g. model_benchmarking/results/answers_gpt-55.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM for RAGChecker extractor/checker (litellm format, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available answer files and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_answer_files()
        return

    if not args.input_file:
        parser.print_help()
        print("\nTip: use --list to see available files.")
        sys.exit(1)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    model_name = derive_model_name(input_path)
    print(f"Evaluating: {model_name}")
    print(f"Input     : {input_path}")
    print(f"Checker   : {args.model}\n")

    # ── Load results ───────────────────────────────────────────────────────────
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    rag_results = RAGResults.from_json(raw)
    print(f"Loaded {len(rag_results.results)} queries\n")

    # ── Run RAGChecker ─────────────────────────────────────────────────────────
    evaluator = RAGChecker(
        extractor_name=args.model,
        checker_name=args.model,
        batch_size_extractor=32,
        batch_size_checker=32,
    )

    evaluator.evaluate(rag_results, all_metrics)

    # ── Save full results (before print, so file is never lost on crash) ───────
    out_path = RESULTS_DIR / f"{model_name}_ragchecker.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            json.loads(rag_results.to_json()),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull results saved -> {out_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print(rag_results)
    print("-" * 60)


if __name__ == "__main__":
    main()
