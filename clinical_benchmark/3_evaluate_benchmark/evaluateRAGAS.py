import json
import os
from datasets import Dataset
from dotenv import load_dotenv

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas import evaluate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def convert_to_ragas_format(results):
    ragas_items = []
    for item in results:
        ragas_items.append({
            "question": item["query"],
            "answer": item["response"],
            "ground_truth": item["gt_answer"],
            "contexts": [ctx["text"] for ctx in item["retrieved_context"]]
        })
    return Dataset.from_list(ragas_items)

with open("ragchecker_input.json", "r") as f:
    data = json.load(f)

dataset = convert_to_ragas_format(data["results"])

metrics = [
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
]

scores = evaluate(
    dataset=dataset,
    metrics=metrics
)

print(scores)