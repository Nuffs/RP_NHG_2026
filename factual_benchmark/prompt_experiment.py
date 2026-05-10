from generate_dataset import generate_qa_pairs
from validate_dataset import evaluate_qa_pairs
import json, os

PROMPTS_DIR = "factual_benchmark/prompts"
RESULTS_DIR = "factual_benchmark/results"

def run_prompt_experiment():
    with open("data/benchmark_chunks.jsonl", "r") as f:
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
        results = evaluate_qa_pairs(scraped_chunks, qa_pairs)
        
        # Store results
        all_results.append({
            "prompt_name": prompt_config["name"],
            "results": results
        })
    
    # Save all results to a JSONL file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "prompt_experiment_scores.jsonl"), "w") as f:
        for result in all_results:
            print(f"Saving results for prompt: {result['prompt_name']}")
            print(f"  Overall BERTScore - Precision: {result['results']['overall']['precision']:.4f}, Recall: {result['results']['overall']['recall']:.4f}, F1: {result['results']['overall']['f1']:.4f}")
            f.write(json.dumps(result) + "\n")
        print(f"\n✅ Experiment complete! Results saved to {os.path.join(RESULTS_DIR, 'prompt_experiment_scores.jsonl')}")

    return all_results


if __name__ == "__main__":
    try:
        print("Starting prompt experiment...\n")
        results = run_prompt_experiment()
        print(f"\nGenerated results for {len(results)} prompts")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the following files exist:")
        print("  - data/benchmark_chunks.jsonl")
        print("  - factual_benchmark/prompts/*.json")
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()