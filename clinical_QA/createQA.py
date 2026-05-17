import json
import uuid
import re
import time
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIGURATION
N_QUESTIONS = 1  # Number of questions per chunk
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
MIN_TOKENS = 50  # Skip chunks with little to no information
MAX_CHUNKS = 5  # set to None to process all chunks

SYSTEM_PROMPT = """
Je bent een ervaren Nederlandse huisarts die toetsvragen schrijft voor huisartsen in opleiding.

Gegeven een fragment uit een NHG-richtlijn, schrijf je een realistische klinische casus 
in het format van een Key Feature Question (KFQ).

## Wat is een Key Feature Question?
Een KFQ beschrijft een concrete patiëntsituatie en stelt één kritische klinische vraag 
over een beslismoment. De vraag test of de arts de juiste klinische redenering toepast, 
niet of hij de richtlijn kan opzoeken.

## Structuur van de casus
Beschrijf een concrete patiënt met:
- Naam, leeftijd, geslacht
- Aanleiding voor het consult
- Relevante klachten, duur en beloop
- Voorgeschiedenis, medicatie, leefstijl, allergieën waar relevant
- Bevindingen uit anamnese of lichamelijk onderzoek waar relevant
- Eventuele voorkeuren of zorgen van de patiënt

Sluit af met één concrete klinische vraag over:
- Diagnose ("Wat is de meest waarschijnlijke diagnose?")
- Beleid ("Welk beleid is het meest aangewezen?")
- Behandeling ("Welk medicament heeft de voorkeur?")
- Verwijzing ("Is verwijzing geïndiceerd, en zo ja, met welke urgentie?")
- Follow-up ("Wanneer en hoe volgt u deze patiënt op?")

## Regels
1. De casus is REALISTISCH en SPECIFIEK — gebruik een echte naam, leeftijd en concrete details.
2. Verwerk bewust klinisch relevante details die de beslissing beïnvloeden (bijv. contra-indicaties, comorbiditeit, leefstijl).
3. Het antwoord moet volledig onderbouwd kunnen worden vanuit de gegeven NHG-tekst.
4. Vermijd vragen die beginnen met "Wat zegt de richtlijn over..." of "Noem de criteria voor...".
5. De casus moet de arts dwingen tot redeneren, niet tot het opzoeken van een definitie.
6. Schrijf in natuurlijk Nederlands, zoals een ervaren huisarts zou communiceren.
7. Verzin GEEN feiten die niet in de tekst staan.

#3 Regels antwoord
1. Het antwoord (gt_answer) bestaat uit maximaal 2-3 zinnen. 
2. Geef alleen de klinische conclusie en de directe onderbouwing vanuit de tekst. 
3. Geen opsommingen, geen uitgebreide uitleg.

Geef je uitvoer ALLEEN als JSON-array, zonder markdown of uitleg:
[
  {
    "query": "Volledige casusbeschrijving gevolgd door de klinische vraag.",
    "gt_answer": "Het antwoord op de vraag.",
    "source_span": "Verbatim fragment uit de tekst dat het antwoord onderbouwt."
  }
]
"""

def strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()

def validate_qa(qa: dict, source_text: str) -> bool:
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False
    if qa["source_span"] not in source_text:
        print(f"  ⚠ source_span not found in chunk, skipping: {qa['source_span'][:60]}...")
        return False
    return True

def generate_questions(chunk: dict, n_questions: int = N_QUESTIONS) -> list:
    text = chunk["text"]
    prompt = f"""NHG-richtlijntekst ({chunk.get('section_path', 'onbekend')}):

\"\"\"{text}\"\"\"

Genereer {n_questions} hoogwaardige klinische QA-paren op basis van bovenstaande tekst."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            raw = response.choices[0].message.content
            cleaned = strip_markdown_fences(raw)
            qa_list = json.loads(cleaned)

            valid = [qa for qa in qa_list if validate_qa(qa, text)]
            print(f"  ✓ {len(valid)}/{len(qa_list)} valid QA pairs generated")
            return valid

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} — JSON parse error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} — Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    print(f"  ✗ Failed after {MAX_RETRIES} attempts, skipping chunk.")
    return []

def process_jsonl(input_path: str, output_path: str):
    results = []
    failed_chunks = []
    processed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # Filter out short chunks
    chunks = [c for c in chunks if c.get("tokens", len(c["text"].split())) >= MIN_TOKENS]
    print(f"Processing {len(chunks)} chunks after filtering (min {MIN_TOKENS} tokens)...")

    if MAX_CHUNKS is not None:
        print(f"Limiting to first {MAX_CHUNKS} chunks for testing.")

    for i, chunk in enumerate(chunks):
        if MAX_CHUNKS is not None and processed >= MAX_CHUNKS:
            print(f"\nReached MAX_CHUNKS limit of {MAX_CHUNKS}, stopping.")
            break

        print(f"[{processed+1}/{MAX_CHUNKS or len(chunks)}] chunk_id={chunk.get('chunk_id', '?')} — {chunk.get('section_path', '?')}")
        qa_pairs = generate_questions(chunk)

        if not qa_pairs:
            failed_chunks.append(chunk.get("chunk_id", f"index_{i}"))
        else:
            for qa in qa_pairs:
                results.append({
                    "query_id": str(uuid.uuid4())[:8],
                    "query": qa["query"],
                    "gt_answer": qa["gt_answer"],
                    "source_span": qa.get("source_span", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_path": chunk.get("section_path", ""),
                })

        processed += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(results)} QA pairs to {output_path}")
    if failed_chunks:
        print(f"Failed chunks ({len(failed_chunks)}): {failed_chunks}")

if __name__ == "__main__":
    process_jsonl(
        "../data/nhg_subset_guidelines.jsonl",
        "results/QA.json"
    )