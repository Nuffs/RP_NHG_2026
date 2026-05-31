import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

MODEL_NAME = "gemini-embedding-2"
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_CONTEXT_COLLECTION = "context_blocks"
DEFAULT_EMBEDDING_COLLECTION = "embedding_blocks"


def to_point_uuid(raw_id: Any, namespace: str) -> str:
    if raw_id is None:
        return str(uuid.uuid4())

    raw = str(raw_id)
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{raw}"))


def ensure_collection_exists(client: QdrantClient, collection_name: str) -> None:
    available = {collection.name for collection in client.get_collections().collections}
    if collection_name not in available:
        available_text = ", ".join(sorted(available)) if available else "<none>"
        raise ValueError(f"Collection {collection_name} not found. Available collections: {available_text}")


def ensure_embedding_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    available = {collection.name for collection in client.get_collections().collections}
    if collection_name in available:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True),
        hnsw_config=HnswConfigDiff(on_disk=True),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=0),
        on_disk_payload=True,
    )


def get_all_context_ids(
    client: QdrantClient,
    context_collection: str,
) -> list[str]:
    context_ids: list[str] = []
    offset = None
    limit = 100

    while True:
        points, next_offset = client.scroll(
            collection_name=context_collection,
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            context_id = payload.get("context_id")
            if context_id:
                context_ids.append(str(context_id))

        if next_offset is None:
            break
        offset = next_offset

    return context_ids


def get_existing_embedding_context_ids(
    client: QdrantClient,
    embedding_collection: str,
) -> set[str]:
    existing: set[str] = set()
    # If the embedding collection doesn't exist yet, return empty set instead
    # of calling `scroll` (which raises 404 when the collection is missing).
    available = {collection.name for collection in client.get_collections().collections}
    if embedding_collection not in available:
        return existing
    offset = None
    limit = 100

    while True:
        points, next_offset = client.scroll(
            collection_name=embedding_collection,
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            context_id = payload.get("context_id")
            if context_id:
                existing.add(str(context_id))

        if next_offset is None:
            break
        offset = next_offset

    return existing


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


def embed_text(client: genai.Client, text: str) -> list[float]:
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    if not response.embeddings:
        raise RuntimeError("No embeddings returned by Gemini")
    values = response.embeddings[0].values or []
    return list(values)


def embed_missing_contexts(
    qdrant_url: str = DEFAULT_QDRANT_URL,
    context_collection: str = DEFAULT_CONTEXT_COLLECTION,
    embedding_collection: str = DEFAULT_EMBEDDING_COLLECTION,
    api_key: str | None = None,
    chunks_per_minute: int = 30,
    limit: int | None = None,
):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not provided and not found in environment")
        sys.exit(1)

    genai_client = genai.Client(api_key=api_key)
    print(f"Using Qdrant server: {qdrant_url}")
    qdrant_client = QdrantClient(url=qdrant_url)

    # Compute rate limiting interval.
    min_interval = 0.0
    if chunks_per_minute is not None:
        if chunks_per_minute <= 0:
            print("Error: chunks_per_minute must be greater than 0")
            sys.exit(1)
        min_interval = 60.0 / float(chunks_per_minute)

    print(f"Retrieving context IDs from '{context_collection}'...")
    ensure_collection_exists(qdrant_client, context_collection)
    all_context_ids = get_all_context_ids(qdrant_client, context_collection)
    print(f"  Found {len(all_context_ids)} total context blocks")

    print(f"Retrieving existing embeddings from '{embedding_collection}'...")
    existing_ids = get_existing_embedding_context_ids(qdrant_client, embedding_collection)
    print(f"  Found {len(existing_ids)} existing embeddings")

    # Compute missing context IDs.
    missing_ids = [cid for cid in all_context_ids if cid not in existing_ids]
    print(f"  Missing: {len(missing_ids)} embeddings to generate")

    if not missing_ids:
        print("All context blocks already have embeddings")
        return

    # Apply optional limit.
    if limit is not None and limit > 0:
        missing_ids = missing_ids[:limit]
        print(f"  Limited to first {len(missing_ids)} missing items")

    # Generate embeddings with rate limiting.
    embeddings_to_insert: list[dict[str, Any]] = []
    processed = 0
    next_allowed_request_time = None

    print(f"\nGenerating embeddings ({chunks_per_minute} chunks/minute)...")
    if min_interval > 0:
        print(f"Rate limit: {min_interval:.2f}s between requests\n")

    for idx, context_id in enumerate(missing_ids, start=1):
        payload = get_context_payload(qdrant_client, context_collection, context_id)
        if not payload:
            print(f"[{idx}/{len(missing_ids)}] Skipping {context_id}: no payload found")
            continue

        text = build_combined_text(payload)
        if not text:
            print(f"[{idx}/{len(missing_ids)}] Skipping {context_id}: no text found")
            continue

        # Respect rate limiting.
        now = time.monotonic()
        if next_allowed_request_time is not None and min_interval > 0:
            remaining = next_allowed_request_time - now
            if remaining > 0:
                print(f"[{idx}/{len(missing_ids)}] Waiting {remaining:.2f}s to respect rate limit...", flush=True)
                time.sleep(remaining)

        # Reserve the next slot before making the request so failures still count.
        next_allowed_request_time = time.monotonic() + min_interval

        print(f"[{idx}/{len(missing_ids)}] Embedding {context_id}...", end=" ", flush=True)

        try:
            embedding_vector = embed_text(genai_client, text)
            embeddings_to_insert.append({
                "context_id": context_id,
                "embedding": embedding_vector,
                "chunk_id": context_id,  # Keep chunk_id for reference
            })
            processed += 1
            print("Success")
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Insert new embeddings into Qdrant.
    if embeddings_to_insert:
        print(f"\nUpserting {len(embeddings_to_insert)} embeddings into '{embedding_collection}'...")
        ensure_embedding_collection(
            qdrant_client,
            embedding_collection,
            vector_size=len(embeddings_to_insert[0]["embedding"]),
        )
        points: list[PointStruct] = []

        for emb in embeddings_to_insert:
            embedding_id = str(uuid.uuid4())
            point_id = to_point_uuid(embedding_id, namespace="embedding")

            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb["embedding"],
                    payload={
                        "embedding_id": embedding_id,
                        "context_id": emb["context_id"],
                        "chunk_id": emb["chunk_id"],
                        "model": MODEL_NAME,
                        "source": "new_embed.py",
                    },
                )
            )

        qdrant_client.upsert(collection_name=embedding_collection, points=points)
        print(f"Upserted {len(points)} embeddings")

    print(f"\nProcessed {processed} new embeddings")


def main() -> None:
    load_dotenv()

    try:
        embed_missing_contexts(
            qdrant_url=DEFAULT_QDRANT_URL,
            context_collection=DEFAULT_CONTEXT_COLLECTION,
            embedding_collection=DEFAULT_EMBEDDING_COLLECTION,
            api_key=None,
            chunks_per_minute=30,
            limit=None,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise exc

if __name__ == "__main__":
    main()

