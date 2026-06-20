"""
ragchecker/main.py

Evaluate model answers against the RAGChecker framework.

Single file:
    python ragchecker/main.py model_benchmarking/results/factual/answers_gpt-55.json

Batch (all answer files in a directory):
    python ragchecker/main.py --batch model_benchmarking/results/factual/
    python ragchecker/main.py --batch model_benchmarking/results/clinical/

The RAGChecker results are saved alongside the input answers:
    <input_dir>/ragchecker/<model_name>_ragchecker.json

Options:
    --model   LLM for extractor/checker (litellm format, default: openai/gpt-4o-mini)
    --batch   Run on all answers_*.json files in the given directory
"""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 output so Dutch characters don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from ragchecker import RAGChecker, RAGResults
from ragchecker.metrics import all_metrics

load_dotenv()

# Default LLM used by RAGChecker as extractor + checker (via litellm / OpenAI).
# gpt-5.5 and reasoning models are NOT compatible — they reject the non-default
# temperature that RAGChecker hardcodes internally. gpt-4o-mini is accurate,
# fast, cheap, and fully compatible.
DEFAULT_MODEL = "openai/gpt-4o-mini"


# ── Helpers ────────────────────────────────────────────────────────────────────

def derive_model_name(path: Path) -> str:
    """answers_claude-opus-4-7.json  ->  claude-opus-4-7"""
    return path.stem.removeprefix("answers_")


def evaluate_file(input_path: Path, output_dir: Path, checker_model: str) -> None:
    model_name = derive_model_name(input_path)
    out_path   = output_dir / f"{model_name}_ragchecker.json"

    print(f"\n{'='*60}")
    print(f"Model  : {model_name}")
    print(f"Input  : {input_path}")
    print(f"Output : {out_path}")
    print(f"Checker: {checker_model}")
    print('='*60)

    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    rag_results = RAGResults.from_json(raw)
    print(f"Loaded {len(rag_results.results)} queries")

    evaluator = RAGChecker(
        extractor_name=checker_model,
        checker_name=checker_model,
        batch_size_extractor=32,
        batch_size_checker=32,
    )
    evaluator.evaluate(rag_results, all_metrics)

    # Save before printing so results are never lost on crash
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json.loads(rag_results.to_json()), f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")

    print("\n" + "-" * 60)
    print(rag_results)
    print("-" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGChecker on model answer files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a single answers_<model>.json, or a directory when using --batch",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM for extractor/checker (litellm format, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run on all answers_*.json files found in the given directory",
    )
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            print(f"Error: --batch requires a directory, got: {input_path}")
            sys.exit(1)
        answer_files = sorted(input_path.glob("answers_*.json"))
        if not answer_files:
            print(f"No answers_*.json files found in {input_path}")
            sys.exit(1)
        output_dir = input_path / "ragchecker"
        output_dir.mkdir(exist_ok=True)
        print(f"Batch mode: {len(answer_files)} files in {input_path}")
        print(f"Output dir: {output_dir}")
        for f in answer_files:
            evaluate_file(f, output_dir, args.model)
        print(f"\nAll done. Results in {output_dir}")
    else:
        if not input_path.exists():
            print(f"Error: file not found: {input_path}")
            sys.exit(1)
        output_dir = input_path.parent / "ragchecker"
        output_dir.mkdir(exist_ok=True)
        evaluate_file(input_path, output_dir, args.model)


if __name__ == "__main__":
    main()
