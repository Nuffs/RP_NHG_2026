from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-embedding-2"
DEFAULT_EMBEDDINGS_FILE = "embeddings.json"
DEFAULT_HISTORY_FILE = "query_history.json"
DEFAULT_TOP_K = 5


def load_embedding_index(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")

    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def get_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key:
        return explicit_api_key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    return api_key


def embed_query(prompt: str, api_key: str | None = None) -> list[float]:
    client = genai.Client(api_key=get_api_key(api_key))
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not response.embeddings:
        raise RuntimeError("No embeddings returned by Gemini")
    return list(response.embeddings[0].values)


def cosine_similarity(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)

    if query_norm == 0:
        return np.zeros(len(vectors), dtype=float)

    denom = vector_norms * query_norm
    scores = np.zeros(len(vectors), dtype=float)
    valid = denom > 0
    scores[valid] = np.dot(vectors[valid], query_vector) / denom[valid]
    return scores


def retrieve_top_chunks(
    query_embedding: list[float],
    embedding_index: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    valid_items: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    for item in embedding_index:
        chunk_id = item.get("chunk_id")
        embedding = item.get("embedding")
        if not chunk_id or not isinstance(embedding, list) or not embedding:
            continue
        valid_items.append(item)
        vectors.append(embedding)

    if not valid_items:
        return []

    query_vec = np.asarray(query_embedding, dtype=float)
    matrix = np.asarray(vectors, dtype=float)
    scores = cosine_similarity(query_vec, matrix)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        item = valid_items[int(idx)]
        results.append(
            {
                "chunk_id": item["chunk_id"],
                "score": float(scores[int(idx)]),
            }
        )
    return results


def load_history(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"History file must contain a JSON array: {file_path}")
    return data


def save_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(
    history_file: str | Path,
    prompt: str,
    query_embedding: list[float],
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    history = load_history(history_file)
    query_id = len(history) + 1

    record = {
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "embedding": query_embedding,
        "retrieved_chunks": retrieved_chunks,
    }
    history.append(record)
    save_history(history_file, history)
    return record


def run_query(
    prompt: str,
    embeddings_file: str | Path = DEFAULT_EMBEDDINGS_FILE,
    history_file: str | Path = DEFAULT_HISTORY_FILE,
    top_k: int = DEFAULT_TOP_K,
    api_key: str | None = None,
) -> list[str]:
    embedding_index = load_embedding_index(embeddings_file)
    query_embedding = embed_query(prompt, api_key=api_key)
    retrieved = retrieve_top_chunks(query_embedding, embedding_index, top_k=top_k)
    append_history(history_file, prompt, query_embedding, retrieved)
    return [item["chunk_id"] for item in retrieved]


def prompt_user() -> str:
    prompt = input("Enter your query: ").strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    return prompt


def main() -> int:
    load_dotenv()

    embeddings_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EMBEDDINGS_FILE
    history_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_HISTORY_FILE
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_TOP_K
    api_key = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        prompt = prompt_user()
        chunk_ids = run_query(
            prompt=prompt,
            embeddings_file=embeddings_file,
            history_file=history_file,
            top_k=top_k,
            api_key=api_key,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("\nTop chunk IDs:")
    for chunk_id in chunk_ids:
        print(chunk_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

