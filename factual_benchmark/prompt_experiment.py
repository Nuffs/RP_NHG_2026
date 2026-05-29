from generate_dataset import generate_qa_pairs
from validate_dataset import evaluate_qa_pairs_grounding
import json, os

PROMPTS_DIR = "factual_benchmark/prompts"
RESULTS_DIR = "factual_benchmark/results"

def run_prompt_experiment():
    with open("data/benchmark_prompt_experiment_chunks.jsonl", "r") as f:
        scraped_chunks = [json.loads(line) for line in f]
    
    prompt_files = [f for f in os.listdir(PROMPTS_DIR)]
    
    all_results = []
    
    for prompt_file in prompt_files:
        with open(os.path.join(PROMPTS_DIR, prompt_file), "r") as f:
            prompt_config = json.load(f)
        
        print(f"Running experiment for prompt: {prompt_config['name']}")
        
        # Generate QA pairs
        qa_pairs = generate_qa_pairs(scraped_chunks, prompt_config)
        
        # Evaluate QA pairs
        results = evaluate_qa_pairs_grounding(scraped_chunks, qa_pairs)
        
        all_results.append({
            "prompt_name": prompt_config["name"],
            "results": results
        })
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "prompt_experiment_scores.jsonl"), "w") as f:
        for result in all_results:
            print(f"Saving results for prompt: {result['prompt_name']}")
            f.write(json.dumps(result) + "\n")
        print(f"Results saved to {os.path.join(RESULTS_DIR, 'prompt_experiment_scores.jsonl')}")

    return all_results


if __name__ == "__main__":
    results = run_prompt_experiment()
    print(f"\nGenerated results for {len(results)} prompts")