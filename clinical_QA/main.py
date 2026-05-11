import json
import uuid
from openai import OpenAI

client = OpenAI(api_key="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")

# CONFIGURATION
N_QUESTIONS = 5  # how many questions per chunk
MODEL = "gpt-4o-mini"  # to test

# INSTRUCTION SHEET (RealMedQA-style)
INSTRUCTION_SHEET = """
You generate realistic, PICO‑structured clinical questions that a Dutch general practitioner might ask during patient care.

Rules:
1. Each question MUST be fully answerable using ONLY the provided NHG guideline text.
2. Questions MUST follow a PICO‑like structure, even if implicit:
- P: Patient / Problem
- I: Intervention / Investigation / Action
- C: Comparison (optional; e.g., alternatief beleid, afwachten, verwijzen)
- O: Outcome / doel van beleid (e.g., diagnose bevestigen, risico verlagen)
3. Avoid comprehension or definition questions (e.g., “Wat zegt deze richtlijn over X”).
4. Write realistic clinical questions about diagnosis, beleid, behandeling, verwijzing, follow‑up.
5. Use natural Dutch clinical language.
6. Do NOT invent facts not present in the text.
7. For each question provide a strictly text‑grounded answer.

Output format (strict JSON)
Code
[
  {
    "query": "...",
    "gt_answer": "..."
  }
]
"""


# FUNCTION: Generate QA pairs
def generate_questions(text, n_questions=N_QUESTIONS):
    prompt = f"""
NHG guideline text:
\"\"\"{text}\"\"\"

Using the instruction sheet below, generate {n_questions} high-quality clinical QA pairs.

Instruction sheet:
{INSTRUCTION_SHEET}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    # Parse JSON returned by the model
    try:
        qa_list = json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Error parsing model output:", e)
        print("Raw output:", response.choices[0].message.content)
        return []

    return qa_list


# MAIN PIPELINE
def process_jsonl(input_path, output_path):
    results = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]

            qa_pairs = generate_questions(text)

            for qa in qa_pairs:
                results.append({
                    "query_id": str(uuid.uuid4())[:8],
                    "query": qa["query"],
                    "gt_answer": qa["gt_answer"],
                })

    # Save in RAGChecker format
    output = {"results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} QA pairs to {output_path}")


# RUN
if __name__ == "__main__":
    process_jsonl("C:/Users/annes/OneDrive/Documenten/00_TUD/Y3/4_Thesis/code/RP_NHG_2026/scraping/astma/nhg_astma.jsonl", "ragchecker_output.json")
