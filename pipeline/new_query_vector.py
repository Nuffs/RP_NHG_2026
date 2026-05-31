from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, SearchParams

DEFAULT_SCORE_THRESHOLD = 0.7  # global threshold: return neighbors with score >= this
DEFAULT_TOP_K = 100
MODEL_NAME = "gemini-embedding-2"

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_COLLECTION = "embedding_blocks"
DEFAULT_CONTEXT_COLLECTION = "context_blocks"
DEFAULT_HISTORY_FILE = "query_history_qdrant.json"


def get_api_key(explicit_api_key: str | None = None) -> str:
    if explicit_api_key:
        return explicit_api_key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    return api_key


def build_combined_text(payload: dict[str, Any]) -> str:
    section_path = payload.get("section_path") or []
    text = payload.get("text") or ""

    header_lines: list[str] = []
    for idx, header in enumerate(section_path, start=1):
        header_text = str(header).strip()
        if not header_text:
            continue
        header_lines.append(f"{'#' * idx} {header_text}")

    parts = header_lines.copy()
    if text:
        parts.append(str(text).strip())

    return "\n\n".join(part for part in parts if part)

def embed_query(prompt: str, api_key: str | None = None) -> list[float]:
    client = genai.Client(api_key=get_api_key(api_key))
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    if not response.embeddings:
        raise RuntimeError("No embeddings returned by Gemini")
    values = response.embeddings[0].values or []
    return list(values)


def query_embedding_collection(
    client: QdrantClient,
    embedding_collection: str,
    query_embedding: list[float],
    top_k: int,
    hnsw_ef: int,
) -> list[dict[str, Any]]:
    # exact=False forces ANN search, and Qdrant executes it using HNSW.
    response = client.query_points(
        collection_name=embedding_collection,
        query=query_embedding,  # type: ignore[arg-type]
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(exact=False, hnsw_ef=hnsw_ef),
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        context_id = payload.get("context_id") or payload.get("chunk_id")
        if not context_id:
            continue
        results.append(
            {
                "context_id": str(context_id),
                "score": float(point.score),
                "embedding_id": payload.get("embedding_id"),
            }
        )
    return results


def get_context_payload(
    client: QdrantClient,
    context_collection: str,
    context_id: str,
) -> dict[str, Any] | None:
    points, _ = client.scroll(
        collection_name=context_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="context_id", match=MatchValue(value=context_id))]
        ),
        with_payload=True,
        with_vectors=False,
        limit=1,
    )
    if not points:
        return None
    return points[0].payload or {}


def retrieve_top_chunks(
    client: QdrantClient,
    embedding_collection: str,
    context_collection: str,
    query_embedding: list[float],
    top_k: int = DEFAULT_TOP_K,
    hnsw_ef: int = 128,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    # Request up to `top_k` ANN candidates from Qdrant, then filter by similarity
    # score so we return all neighbours above the global threshold.
    ann_hits = query_embedding_collection(
        client=client,
        embedding_collection=embedding_collection,
        query_embedding=query_embedding,
        top_k=top_k,
        hnsw_ef=hnsw_ef,
    )

    # Keep only hits meeting the score threshold.
    filtered_hits = [hit for hit in ann_hits if (hit.get("score") or 0) >= score_threshold]

    results: list[dict[str, Any]] = []
    for hit in filtered_hits:
        context_payload = get_context_payload(client, context_collection, hit["context_id"])
        if context_payload is None:
            continue

        results.append(
            {
                "chunk_id": context_payload.get("chunk_id") or hit["context_id"],
                "context_id": hit["context_id"],
                "score": hit["score"],
                "doc_id": context_payload.get("doc_id"),
                "doc_title": context_payload.get("doc_title"),
                "section_path": context_payload.get("section_path"),
                "text": context_payload.get("text"),
            }
        )

    return results


def ensure_collections_exist(
    client: QdrantClient,
    embedding_collection: str,
    context_collection: str,
) -> None:
    available = {c.name for c in client.get_collections().collections}
    missing: list[str] = []
    if embedding_collection not in available:
        missing.append(embedding_collection)
    if context_collection not in available:
        missing.append(context_collection)

    if missing:
        available_text = ", ".join(sorted(available)) if available else "<none>"
        raise ValueError(
            f"Missing collection(s): {', '.join(missing)}. Available collections: {available_text}"
        )


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
    qdrant_url: str = DEFAULT_QDRANT_URL,
    embedding_collection: str = DEFAULT_EMBEDDING_COLLECTION,
    context_collection: str = DEFAULT_CONTEXT_COLLECTION,
    history_file: str | Path = DEFAULT_HISTORY_FILE,
    top_k: int = DEFAULT_TOP_K,
    hnsw_ef: int = 128,
    api_key: str | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    print(f"Using Qdrant server: {qdrant_url}")

    client = QdrantClient(url=qdrant_url)
    ensure_collections_exist(client, embedding_collection, context_collection)
    query_embedding = embed_query(prompt, api_key=api_key)
    retrieved = retrieve_top_chunks(
        client=client,
        embedding_collection=embedding_collection,
        context_collection=context_collection,
        query_embedding=query_embedding,
        top_k=top_k,
        hnsw_ef=hnsw_ef,
        score_threshold=score_threshold,
    )
    append_history(history_file, prompt, query_embedding, retrieved)
    return retrieved


def prompt_user() -> str:
    prompt = input("Stel je vraag: ").strip()
    if not prompt:
        raise ValueError("Vraag mag niet leeg zijn")
    return prompt


def main() -> None:
    load_dotenv()

    try:
        prompt = prompt_user()
        chunks = run_query(
            prompt=prompt,
            qdrant_url=DEFAULT_QDRANT_URL,
            embedding_collection=DEFAULT_EMBEDDING_COLLECTION,
            context_collection=DEFAULT_CONTEXT_COLLECTION,
            history_file=DEFAULT_HISTORY_FILE,
            top_k=DEFAULT_TOP_K,
            api_key=None,
            hnsw_ef=128, # Smaller for more speed, larger for more accuracy
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
    except Exception as exc:
        print(f"Something went wrong: {exc}")
        raise exc

    print("\nTop chunks:")
    for i, chunk in enumerate(chunks, start=1):
        score = chunk.get("score")
        if isinstance(score, (int, float)):
            score_text = f"{float(score):.4f}"
        else:
            score_text = "n/a"
        print(f"\n[{i}] {chunk.get('chunk_id')} (score={score_text})")
        if chunk.get("doc_title"):
            print(f"Title: {chunk.get('doc_title')}")
        text = build_combined_text(chunk)
        # (chunk.get("text") or "").strip()
        print(text if text else "<no text available>")


if __name__ == "__main__":
    main()
