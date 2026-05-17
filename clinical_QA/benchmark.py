import numpy as np
from google import genai
from google.genai import types

import json
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GEMINI_MODEL = "gemini-embedding-2"


def embed_query(prompt, api_key=None):
    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
    response = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(response.embeddings[0].values)


def cosine_similarity(query_vector, vectors):
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    if query_norm == 0:
        return np.zeros(len(vectors))
    denom = vector_norms * query_norm
    scores = np.zeros(len(vectors))
    valid = denom > 0
    scores[valid] = np.dot(vectors[valid], query_vector) / denom[valid]
    return scores


def run_query(prompt, embeddings_file="embeddings.json", top_k=5, api_key=None):
    with open(embeddings_file, encoding="utf-8") as f:
        index = json.load(f)

    query_emb = embed_query(prompt, api_key=api_key)
    query_vec = np.array(query_emb)
    vectors = np.array([item["embedding"] for item in index])
    scores = cosine_similarity(query_vec, vectors)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [index[i]["chunk_id"] for i in top_indices]


# Load benchmark
with open("results/QA.json", encoding="utf-8") as f:
    benchmark = json.load(f)

# Load chunk text lookup
chunks = {}
with open("../data/nhg_subset_guidelines.jsonl", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        chunks[chunk["chunk_id"]] = chunk

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ragchecker_input = {"results": []}

for item in benchmark["results"]:
    # 1. Retrieve chunks
    retrieved_ids = run_query(item["query"], embeddings_file="../pipeline/embeddings.json", top_k=5)
    retrieved_chunks = [chunks[cid] for cid in retrieved_ids if cid in chunks]

    # 2. Build context string for the LLM
    context_text = "\n\n".join(c["text"] for c in retrieved_chunks)

    # 3. Generate a response using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap and good; swap for gpt-4o if you want higher quality
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Beantwoord de vraag op basis van de volgende context.\n\nContext:\n{context_text}\n\nVraag: {item['query']}"
        }]
    )
    response_text = response.choices[0].message.content

    # 4. Format for RAGChecker
    ragchecker_input["results"].append({
        "query_id": item["query_id"],
        "query": item["query"],
        "gt_answer": item["gt_answer"],
        "response": response_text,
        "retrieved_context": [
            {"doc_id": c["doc_id"], "text": c["text"]}
            for c in retrieved_chunks
        ]
    })

# Save the prepared file
with open("ragchecker_input.json", "w", encoding="utf-8") as f:
    json.dump(ragchecker_input, f, ensure_ascii=False, indent=2)

print("Saved ragchecker_input.json")