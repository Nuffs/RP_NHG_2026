import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from factual_benchmark.select_chunks import chunk_document, sample_chunks_per_guideline
from factual_benchmark.generate_dataset import generate_qa_pairs
from scraping.ten_guidelines.main import run_scraping

load_dotenv()
client = OpenAI()

RESULTS_DIR = "factual_benchmark/results"


def run_benchmarking_pipeline(output_path="data/benchmark_chunks.jsonl"):
    raw_chunks = run_scraping()

    all_chunks = []
    for raw_chunk in raw_chunks:
        chunks = chunk_document(raw_chunk)
        all_chunks.extend(chunks)

    sampled = sample_chunks_per_guideline(all_chunks)
    
    print(f"Sampled {len(sampled)} chunks across guidelines.")
    
    # save sampled chunks to data/benchmark_chunks.jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved benchmark chunks to {output_path}")
    return output_path


def run_qa_generation(chunks_path="data/benchmark_chunks.jsonl", prompt_name="zero_shot"):
    """
    Generate QA pairs from benchmark chunks using specified prompt config.
    
    Args:
        chunks_path: Path to benchmark chunks JSONL file
        prompt_name: Name of prompt config (without .json extension)
    
    Returns:
        List of QA pairs
    """
    print(f"\n{'='*60}")
    print(f"QA GENERATION: {prompt_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load benchmark chunks
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"  -> Loaded {len(chunks)} chunks")
    
    # Load prompt config
    prompt_path = os.path.join("factual_benchmark/prompts", f"{prompt_name}.json")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt config not found: {prompt_path}")
    
    print(f"Loading prompt config from {prompt_path}...")
    with open(prompt_path, "r") as f:
        prompt_config = json.load(f)
    
    # Generate QA pairs
    print(f"\nGenerating QA pairs using {prompt_config['model']}...\n")
    qa_pairs = generate_qa_pairs(chunks, prompt_config)
    
    return qa_pairs


if __name__ == "__main__":
    # Run benchmarking pipeline (scrape + chunk + sample)
    chunks_path = run_benchmarking_pipeline()
    
    # Generate QA pairs
    qa_pairs = run_qa_generation(chunks_path, prompt_name="zero_shot")
    
    print(f"\n✅ Complete! Generated {len(qa_pairs)} QA pairs")

