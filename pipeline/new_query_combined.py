from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

import new_query_vector as vec_module
import new_query_traditional as trad_module

DEFAULT_RRF_K = 60.0
DEFAULT_TOP_K_VECTOR = 100
DEFAULT_TOP_K_TRADITIONAL = 20
DEFAULT_HNSW_EF = 128
DEFAULT_SCORE_THRESHOLD_VECTOR = 0.7
DEFAULT_SCORE_THRESHOLD_TRADITIONAL = 0.7

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_HISTORY_FILE = "query_history_combined.json"

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

def fetch_context_payload(client: QdrantClient, context_collection: str, context_id: str) -> Dict[str, Any] | None:
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


def rrf_fuse(lists: List[List[Dict[str, Any]]], k: float = DEFAULT_RRF_K) -> list[tuple[Any, float]]:
    # Each input list is ordered by rank (best-first). Compute RRF score per chunk_id.
    scores = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst, start=1):
            cid = item.get("chunk_id") or item.get("context_id")
            if not cid:
                continue
            scores[cid] += 1.0 / (k + float(rank))
    # Produce sorted list of (cid, score)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ordered


def run_combined(
    prompt: str,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    vector_top_k: int = DEFAULT_TOP_K_VECTOR,
    traditional_top_k: int = DEFAULT_TOP_K_TRADITIONAL,
    rrf_k: float = DEFAULT_RRF_K,
    hnsw_ef: int = DEFAULT_HNSW_EF,
    score_threshold_traditional: float = DEFAULT_SCORE_THRESHOLD_TRADITIONAL,
    score_threshold_vector: float = DEFAULT_SCORE_THRESHOLD_VECTOR,
    api_key: str | None = None,
    context_collection: str = "context_blocks",
    embedding_collection="embedding_blocks",
    history_file: str | Path = DEFAULT_HISTORY_FILE,
) -> List[Dict[str, Any]]:

    # Run both queries
    vec_results = vec_module.run_query(
        prompt,
        qdrant_url=qdrant_url,
        embedding_collection=embedding_collection,
        context_collection=context_collection,
        top_k=vector_top_k,
        hnsw_ef=hnsw_ef,
        api_key=api_key,
        score_threshold=score_threshold_vector,
    )

    trad_results = trad_module.run_query(
        prompt,
        qdrant_url=qdrant_url,
        context_collection=context_collection,
        top_k=traditional_top_k,
        score_threshold=score_threshold_traditional,
    )

    # Fuse with RRF
    fused = rrf_fuse([vec_results, trad_results], k=rrf_k)

    # Fetch payloads for top fused results
    client = QdrantClient(url=qdrant_url)
    final: List[Dict[str, Any]] = []
    for cid, score in fused:
        payload = fetch_context_payload(client, context_collection, cid) or {}
        final.append(
            {
                "chunk_id": payload.get("chunk_id") or cid,
                "context_id": cid,
                "rrf_score": float(score),
                "doc_title": payload.get("doc_title"),
                "text": payload.get("text"),
                "section_path": payload.get("section_path"),
            }
        )

    # Append to history
    history = []
    hf = Path(history_file)
    if hf.exists():
        raw = hf.read_text(encoding="utf-8").strip()
        if raw:
            history = json.loads(raw)
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "vector_count": len(vec_results),
        "traditional_count": len(trad_results),
        "fused_count": len(final),
    })
    hf.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    return final


def prompt_user() -> str:
    prompt = input("Stel je vraag: ").strip()
    if not prompt:
        raise ValueError("Vraag mag niet leeg zijn")
    return prompt


def main() -> None:
    load_dotenv()

    try:
        prompt = prompt_user()
        results = run_combined(
            prompt=prompt,
            qdrant_url=DEFAULT_QDRANT_URL,
            vector_top_k=DEFAULT_TOP_K_VECTOR,
            traditional_top_k=DEFAULT_TOP_K_TRADITIONAL,
            rrf_k=DEFAULT_RRF_K,
            hnsw_ef=DEFAULT_HNSW_EF,
            score_threshold_vector=DEFAULT_SCORE_THRESHOLD_VECTOR,
            score_threshold_traditional=DEFAULT_SCORE_THRESHOLD_TRADITIONAL,
            api_key=None,
            context_collection='context_blocks',
            history_file=DEFAULT_HISTORY_FILE,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        raise exc

    print("\nTop fused results:")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] {r.get('chunk_id')} (rrf_score={r.get('rrf_score'):.6f})")
        if r.get("doc_title"):
            print(f"Title: {r.get('doc_title')}")
        text = build_combined_text(r)
        print(text if text else "<no text available>")


if __name__ == "__main__":
    main()

