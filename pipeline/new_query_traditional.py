from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient

DEFAULT_TOP_K = 100
DEFAULT_SCORE_THRESHOLD = 5

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_CONTEXT_COLLECTION = "context_blocks"
DEFAULT_HISTORY_FILE = "query_history_traditional.json"
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]


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


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.docs: List[Dict[str, Any]] = []  # list of payload dicts
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.df: Dict[str, int] = defaultdict(int)
        self.tf: List[Counter] = []

    def add_doc(self, payload: Dict[str, Any]):
        text = build_combined_text(payload)
        tokens = tokenize(text)
        tf = Counter(tokens)
        self.docs.append(payload)
        self.tf.append(tf)
        self.doc_len.append(len(tokens))
        for term in tf.keys():
            self.df[term] += 1
        self.N += 1

    def finalize(self):
        if self.N > 0:
            self.avgdl = sum(self.doc_len) / float(self.N)
        else:
            self.avgdl = 0.0

    def idf(self, term: str) -> float:
        # IDF with add-0.5 smoothing (Okapi-BM25 style)
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_terms: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for term in set(query_terms):
            idf_t = self.idf(term)
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0))
                score_term = idf_t * (f * (self.k1 + 1.0)) / denom
                scores[i] += score_term
        return scores


def fetch_all_contexts(client: QdrantClient, context_collection: str) -> List[Dict[str, Any]]:
    points = []
    offset = None
    limit = 256
    while True:
        batch, offset = client.scroll(
            collection_name=context_collection,
            offset=offset,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for p in batch:
            payload = p.payload or {}
            points.append(payload)
        if offset is None:
            break
    return points


def build_index_from_qdrant(qdrant_url: str, context_collection: str) -> BM25Index:
    client = QdrantClient(url=qdrant_url)
    index = BM25Index()
    docs = fetch_all_contexts(client, context_collection)
    for doc in docs:
        index.add_doc(doc)
    index.finalize()
    return index


def append_history(history_file: Path, prompt: str, retrieved: List[Dict[str, Any]]) -> None:
    history = []
    if history_file.exists():
        raw = history_file.read_text(encoding="utf-8").strip()
        if raw:
            history = json.loads(raw)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "results": retrieved,
    }
    history.append(record)
    history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def run_query(
    prompt: str,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    context_collection: str = DEFAULT_CONTEXT_COLLECTION,
    top_k: int = DEFAULT_TOP_K,
    history_file: str | Path = DEFAULT_HISTORY_FILE,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> List[Dict[str, Any]]:
    index = build_index_from_qdrant(qdrant_url, context_collection)
    query_terms = tokenize(prompt)
    if not query_terms:
        return []
    scores = index.score(query_terms)
    # rank all documents by score (may be zero or negative depending on idf)
    ranked = sorted([(i, scores[i]) for i in range(len(scores))], key=lambda x: x[1], reverse=True)
    # filter by the configured score threshold
    ranked = [(i, s) for (i, s) in ranked if s > score_threshold]
    results: List[Dict[str, Any]] = []
    for rank, (doc_idx, score) in enumerate(ranked[:top_k], start=1):
        payload = index.docs[doc_idx]
        results.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "doc_title": payload.get("doc_title"),
                "score": float(score),
                "section_path": payload.get("section_path"),
                "text": payload.get("text"),
            }
        )
    append_history(Path(history_file), prompt, results)
    return results


def prompt_user() -> str:
    prompt = input("Stel je vraag: ").strip()
    if not prompt:
        raise ValueError("Vraag mag niet leeg zijn")
    return prompt


def main() -> None:
    load_dotenv()

    try:
        prompt = prompt_user()
        results = run_query(
            prompt,
            qdrant_url=DEFAULT_QDRANT_URL,
            context_collection=DEFAULT_CONTEXT_COLLECTION,
            top_k=DEFAULT_TOP_K,
            history_file=DEFAULT_HISTORY_FILE,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )
    except Exception as exc:
        print(f"Something went wrong: {exc}")
        raise exc

    print("\nTop results (traditional):")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] {r.get('chunk_id')} (score={r.get('score'):.4f})")
        if r.get("doc_title"):
            print(f"Title: {r.get('doc_title')}")
        text = build_combined_text(r)
        print(text if text else "<no text available>")

if __name__ == "__main__":
    main()

