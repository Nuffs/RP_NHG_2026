"""
Benchmark generation script for clinical vignette QA dataset.
Compares zero-shot and few-shot prompt strategies and selects the best
using round-trip retrieval hit rate and BERTScore against source chunk.
"""

import json
import uuid
import re
import time
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from bert_score import score as bert_score

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
MIN_TOKENS = 50  # Skip chunks with too little content
MAX_CHUNKS = 10  # Set to an int (e.g. 10) for testing; None = all chunks
N_QUESTIONS = 1  # Questions per chunk
TOP_K = 5  # Chunks to retrieve during round-trip validation

INPUT_PATH = "../data/nhg_subset_guidelines.jsonl"
EMBEDDINGS_PATH = "../pipeline/embeddings.json"
OUTPUT_DIR = "results"

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """
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
8. source_span MOET een exacte kopie zijn uit de gegeven tekst.
9. Je mag de tekst NIET parafraseren.
10. Kies alleen een letterlijke zin of zinsdeel uit de bron.

## Regels antwoord
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

# One high-quality manually written example to anchor few-shot generation.
# Based on the astma_bij_volwassenen guideline, section Richtlijnen diagnostiek.
FEW_SHOT_EXAMPLE = """
Hieronder volgt een voorbeeld van een hoogwaardige KFQ zoals bedoeld:

Tekst: "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten. 
Voer spirometrie uit bij patiënten met klachten die passen bij astma of COPD."

Gewenste output:
[
  {
    "query": "Mevrouw De Vries, 52 jaar, komt op uw spreekuur met al drie maanden aanhoudende hoestklachten en kortademigheid bij inspanning. Ze rookt 10 jaar, een half pakje per dag. Haar longauscultatie is normaal. U overweegt astma of COPD. Welk aanvullend onderzoek is als eerste aangewezen om obstructie aan te tonen of uit te sluiten?",
    "gt_answer": "Spirometrie is aangewezen om obstructie aan te tonen of uit te sluiten bij klachten die passen bij astma of COPD. Dit is de geëigende methode volgens de richtlijn.",
    "source_span": "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten."
  }
]

Genereer nu een vergelijkbare KFQ op basis van de onderstaande tekst.
"""

SYSTEM_PROMPT_ZERO_SHOT = SYSTEM_PROMPT_BASE
SYSTEM_PROMPT_FEW_SHOT = SYSTEM_PROMPT_BASE + FEW_SHOT_EXAMPLE
SYSTEM_PROMPT_COT = SYSTEM_PROMPT_BASE + """

## INTERNE REDENEERSTAPPEN (niet tonen):
1. Zoek het beslismoment
2. Selecteer relevante klinische factoren
3. Kies relevante zin uit de tekst (letterlijk)
4. Bouw KFQ rond deze exacte zin

## OUTPUT REGELS (strikt):
- Alleen JSON output
- source_span = exact citaat uit brontekst
- geen uitleg, geen tussenstappen
"""


# ---------------------------------------------------------------------------
# EMBEDDING + RETRIEVAL (mirrors query.py, no import dependency)
# ---------------------------------------------------------------------------

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-embedding-2"


def embed_query(prompt: str) -> list[float]:
    gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = gclient.models.embed_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(response.embeddings[0].values)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec)
    m_norms = np.linalg.norm(matrix, axis=1)
    if q_norm == 0:
        return np.zeros(len(matrix))
    denom = m_norms * q_norm
    scores = np.zeros(len(matrix))
    valid = denom > 0
    scores[valid] = np.dot(matrix[valid], query_vec) / denom[valid]
    return scores


def run_query(prompt: str, embeddings_file: str, top_k: int = TOP_K) -> list[str]:
    with open(embeddings_file, encoding="utf-8") as f:
        index = json.load(f)
    query_vec = np.array(embed_query(prompt))
    vectors = np.array([item["embedding"] for item in index])
    scores = cosine_similarity(query_vec, vectors)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [index[i]["chunk_id"] for i in top_idx]


# ---------------------------------------------------------------------------
# QA GENERATION
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()

def critique_and_refine(chunk: dict, draft_qa: dict) -> dict:
    """
    Second-pass refinement: model critiques and improves its own KFQ.
    """

    critique_prompt = f"""
    Je beoordeelt een KFQ.

    ## BRONTEKST:
    \"\"\"{chunk['text']}\"\"\"

    ## KFQ:
    {json.dumps(draft_qa, ensure_ascii=False)}

    ## CRITICAL RULE:
    Je mag GEEN inhoud uit de bron herschrijven.

    Je mag alleen:
    - formulering verbeteren
    - structuur verbeteren
    - maar source_span MOET EXACT ONGEWIJZIGD blijven uit de bron

    ## OUTPUT:
    Return exact JSON:
    [
      {{
        "query": "...",
        "gt_answer": "...",
        "source_span": "EXACTE TEKST UIT BRON (niet wijzigen!)"
      }}
    ]
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Je bent een kritische medische onderwijsassistent."},
            {"role": "user", "content": critique_prompt},
        ],
        temperature=0.3,
    )

    raw = strip_markdown_fences(response.choices[0].message.content)

    try:
        return json.loads(raw)[0]
    except:
        return draft_qa  # fallback

def validate_qa(qa: dict, source_text: str) -> bool:
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False

    def normalize(text):
        return " ".join(text.lower().split())

    if normalize(qa["source_span"]) not in normalize(source_text):
        print(f"    ⚠ source_span not found in chunk: {qa['source_span'][:60]}...")
        return False
    return True

def generate_questions_with_strategy(chunk, system_prompt, strategy_name):
    if strategy_name == "self_critique":
        # Step 1: draft
        drafts = generate_questions(chunk, SYSTEM_PROMPT_BASE)

        refined = []
        for draft in drafts:
            refined_qa = critique_and_refine(chunk, draft)
            refined.append(refined_qa)

        return refined

    else:
        return generate_questions(chunk, system_prompt)

def generate_questions(chunk: dict, system_prompt: str, n: int = N_QUESTIONS) -> list[dict]:
    text = chunk["text"]
    user_prompt = f"""NHG-richtlijntekst ({chunk.get('section_path', 'onbekend')}):

\"\"\"{text}\"\"\"

Genereer {n} hoogwaardige klinische QA-paren op basis van bovenstaande tekst."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            raw = response.choices[0].message.content
            cleaned = strip_markdown_fences(raw)
            qa_list = json.loads(cleaned)
            valid = [qa for qa in qa_list if validate_qa(qa, text)]
            return valid

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} — JSON parse error: {e}")
        except Exception as e:
            print(f"    Attempt {attempt}/{MAX_RETRIES} — Error: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(2)

    return []


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def round_trip_hit_rate(results: list[dict], embeddings_file: str) -> float:
    """
    Round-trip validation: re-retrieve using the generated query and check
    whether the source chunk_id appears in the top-K results.
    A high hit rate indicates that questions are well-grounded in their source.
    """
    hits = 0
    for item in results:
        try:
            retrieved = run_query(item["query"], embeddings_file, top_k=TOP_K)
            if item["chunk_id"] in retrieved:
                hits += 1
        except Exception as e:
            print(f"  Round-trip error for {item['query_id']}: {e}")
    return hits / len(results) if results else 0.0


def bertscore_vs_source(results: list[dict], chunks: dict[str, dict]) -> float:
    """
    BERTScore recall of gt_answer against source chunk text.
    High recall means the answer content is semantically covered by the source,
    confirming the answer is grounded in the guideline text.
    """
    hypotheses = [item["gt_answer"] for item in results]
    references = [chunks[item["chunk_id"]]["text"] for item in results if
                  item["chunk_id"] in chunks]

    if not hypotheses or not references:
        return 0.0

    _, R, _ = bert_score(hypotheses, references, lang="nl", verbose=False)
    return float(R.mean())


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_strategy(
        strategy_name: str,
        system_prompt: str,
        chunks: list[dict],
        chunks_lookup: dict[str, dict],
) -> tuple[list[dict], dict]:
    print(f"\n{'=' * 60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'=' * 60}")

    results = []
    failed = []
    limit = MAX_CHUNKS if MAX_CHUNKS is not None else len(chunks)

    for i, chunk in enumerate(chunks[:limit]):
        print(
            f"[{i + 1}/{limit}] chunk_id={chunk.get('chunk_id', '?')} — {chunk.get('section_path', '?')}")
        qa_pairs = generate_questions_with_strategy(chunk, system_prompt, strategy_name)

        if not qa_pairs:
            failed.append(chunk.get("chunk_id", f"index_{i}"))
            print("  ✗ No valid QA pairs generated")
        else:
            print(f"  ✓ {len(qa_pairs)} valid QA pair(s)")
            for qa in qa_pairs:
                results.append({
                    "query_id": str(uuid.uuid4())[:8],
                    "query": qa["query"],
                    "gt_answer": qa["gt_answer"],
                    "source_span": qa.get("source_span", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_path": chunk.get("section_path", ""),
                    "strategy": strategy_name,
                })

    # Validate
    print(f"\nValidating {len(results)} QA pairs...")
    rt_score = round_trip_hit_rate(results, EMBEDDINGS_PATH)
    bs_score = bertscore_vs_source(results, chunks_lookup)

    metrics = {
        "strategy": strategy_name,
        "n_generated": len(results),
        "n_failed_chunks": len(failed),
        "round_trip_hit_rate": round(rt_score, 4),
        "bertscore_recall_vs_source": round(bs_score, 4),
        # Combined score used for strategy selection
        "combined_score": round((rt_score + bs_score) / 2, 4),
    }

    print(f"\nResults for [{strategy_name}]:")
    print(f"  Round-trip hit rate (@{TOP_K}): {rt_score:.2%}")
    print(f"  BERTScore recall vs source:    {bs_score:.4f}")
    print(f"  Combined score:                {metrics['combined_score']:.4f}")

    return results, metrics


def main():
    # Load chunks
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    all_chunks = [c for c in all_chunks if c.get("tokens", len(c["text"].split())) >= MIN_TOKENS]
    chunks_lookup = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks (after filtering < {MIN_TOKENS} tokens)")

    strategies = [
        ("zero_shot", SYSTEM_PROMPT_ZERO_SHOT),
        ("few_shot", SYSTEM_PROMPT_FEW_SHOT),
        ("cot", SYSTEM_PROMPT_COT),
        ("self_critique", SYSTEM_PROMPT_BASE),
    ]

    all_metrics = []
    strategy_results = {}

    for name, prompt in strategies:
        results, metrics = run_strategy(name, prompt, all_chunks, chunks_lookup)
        strategy_results[name] = results
        all_metrics.append(metrics)

    # Save metrics comparison
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, "strategy_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nStrategy comparison saved to {metrics_path}")

    # Pick best strategy by combined score
    best = max(all_metrics, key=lambda m: m["combined_score"])
    print(f"\nBest strategy: {best['strategy']} (combined score: {best['combined_score']})")

    # Save best results as the final benchmark
    best_results = strategy_results[best["strategy"]]
    output = {"results": best_results, "strategy_comparison": all_metrics}
    qa_path = os.path.join(OUTPUT_DIR, "QA.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Final benchmark ({len(best_results)} QA pairs) saved to {qa_path}")


if __name__ == "__main__":
    main()
