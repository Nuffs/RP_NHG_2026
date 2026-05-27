from dotenv import load_dotenv
load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

with open("ragchecker_input.json", encoding="utf-8") as fp:
    rag_results = RAGResults.from_json(fp.read())

evaluator = RAGChecker(
    extractor_name="openai/gpt-4o-mini",
    checker_name="openai/gpt-4o-mini",
    batch_size_extractor=32,
    batch_size_checker=32
)

evaluator.evaluate(rag_results, all_metrics)
print(rag_results)