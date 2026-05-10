from dotenv import load_dotenv
import json, os
from openai import OpenAI

load_dotenv()
client = OpenAI()

RESULTS_DIR = "factual_benchmark/results"

def generate_qa_pairs(scraped_chunks, prompt_config):
    qa_pairs = []

    for chunk in scraped_chunks:
        print(f"Generating QA pair for chunk {chunk['chunk_id']} with {chunk['tokens']} tokens")
        try:
            prompt = prompt_config["prompt_template"].format(chunk=chunk["text"])

            response = client.chat.completions.create(
                model=prompt_config["model"],
                messages=[{"role": "system", "content": prompt_config["system_prompt"]},
                            {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            content = json.loads(response.choices[0].message.content)
            qa_pairs.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "section_path": chunk["section_path"],
                "source_text": chunk["text"],
                "question": content.get("question"),
                "answer": content.get("answer"),
                "prompt_technique": prompt_config["name"]
            })
        except Exception as e:
            print(f"  ERROR generating QA pair for chunk {chunk['chunk_id']}: {e}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"qa_{prompt_config["name"]}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"  Saved {len(qa_pairs)} Q&A pairs to {output_path}")

    return qa_pairs

if __name__ == "__main__":
    with open("data/benchmark_chunks.jsonl", "r") as f:
        scraped_chunks = [json.loads(line) for line in f]
    
    # just test zero-shot for now
    with open(os.path.join("factual_benchmark/prompts", "zero_shot.json"), "r") as f:
        prompt_config = json.load(f)
    
    qa_pairs = generate_qa_pairs(scraped_chunks, prompt_config)


