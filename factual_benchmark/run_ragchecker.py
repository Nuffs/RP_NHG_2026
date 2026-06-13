import json
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run():
    """
    Load RAG results and evaluate them with RAGChecker.

    Reads: factual_benchmark/results/ragchecker_input.json
    Writes: factual_benchmark/results/ragchecker_metrics.json
    """
    # initialize ragresults from json/dict
    with open(str(HERE / "results" / "ragchecker_input.json")) as fp:
        rag_results = RAGResults.from_json(fp.read())

    # set-up the evaluator
    evaluator = RAGChecker(
        extractor_name="openai/gpt-4o",
        checker_name="openai/gpt-4o",
        batch_size_extractor=32,
        batch_size_checker=32,
    )

    # evaluate results with selected metrics or certain groups, e.g., retriever_metrics,
    # generator_metrics, all_metrics
    evaluator.evaluate(rag_results, all_metrics)
    print(rag_results)

    Path(str(HERE / "results" / "ragchecker_metrics.json")).write_text(
        json.dumps(rag_results.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved RAGChecker results to results/ragchecker_output.json")


if __name__ == "__main__":
    run()
