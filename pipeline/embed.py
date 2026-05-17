import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types


def embed_chunks(
    input_file: str = "nhg_subset_guidelines.jsonl",
    output_file: str = "embeddings.json",
    api_key: str | None = None,
    chunks_per_minute: int = 30,
):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GEMINI_API_KEY not provided and not found in environment")
        print("Usage: python embed.py <api_key> [input_file] [output_file]")
        print("Or set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    min_interval = 0.0
    if chunks_per_minute is not None:
        if chunks_per_minute <= 0:
            print("Error: chunks_per_minute must be greater than 0")
            sys.exit(1)
        min_interval = 60.0 / float(chunks_per_minute)

    embeddings = []
    processed = 0
    next_allowed_request_time = None

    print(f"Reading chunks from {input_file}...")
    if min_interval > 0:
        print(f"Rate limit enabled: {chunks_per_minute} chunks/minute ({min_interval:.2f}s between requests)")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line)
                chunk_id = chunk.get('chunk_id')
                text = chunk.get('text')

                if not chunk_id or not text:
                    print(f"Warning: Line {line_num} missing chunk_id or text, skipping")
                    continue

                now = time.monotonic()
                if next_allowed_request_time is not None and min_interval > 0:
                    remaining = next_allowed_request_time - now
                    if remaining > 0:
                        print(f"Sleeping {remaining:.2f}s to respect rate limit...", flush=True)
                        time.sleep(remaining)

                # Reserve the next slot before making the request so failures still count.
                next_allowed_request_time = time.monotonic() + min_interval

                print(f"[{line_num}] Embedding {chunk_id}...", end=" ", flush=True)

                response = client.models.embed_content(
                    # model="gemini-embedding-001",
                    model="gemini-embedding-2",
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )

                if not response.embeddings:
                    raise RuntimeError("No embeddings returned by Gemini")

                embedding_vector = response.embeddings[0].values

                # Save chunk_id and embedding
                embeddings.append({
                    "chunk_id": chunk_id,
                    "embedding": embedding_vector
                })

                processed += 1
                print("✓")

            except json.JSONDecodeError:
                print(f"Error: Line {line_num} is not valid JSON")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Save embeddings
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False)

    print(f"✓ Processed {processed} chunks")
    print(f"✓ Saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        input_file = sys.argv[2] if len(sys.argv) > 2 else "nhg_subset_guidelines.jsonl"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "embeddings.json"
        chunks_per_minute = 1
        if len(sys.argv) > 4:
            chunks_per_minute = int(sys.argv[4])
        embed_chunks(input_file, output_file, api_key, chunks_per_minute)
    else:
        load_dotenv()
        # chunks_per_minute = int(os.getenv("CHUNKS_PER_MINUTE", "1"))
        embed_chunks()

