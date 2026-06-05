"""
NHG Clinical KFQ Dataset Generator
====================================
Uses the best-performing strategy from the benchmark experiment:
few_shot_cot (Few-Shot + Chain-of-Thought).

Outputs:
  - results/qa_dataset_<timestamp>.json   — all QA pairs
  - results/qa_dataset_<timestamp>.jsonl  — newline-delimited, one pair per line
  - results/generation_log_<timestamp>.json — per-chunk generation stats

Usage:
  pip install openai python-dotenv
  python generate_qa_dataset.py

Configuration (top of file):
  INPUT_PATH        — path to your NHG JSONL chunks
  OUTPUT_DIR        — where outputs are saved
  N_QUESTIONS       — questions per chunk (default: 3)
  MAX_CHUNKS        — set to None for full dataset
  MIN_TOKENS        — skip chunks shorter than this (default: 50)
  SEED              — random seed for reproducibility
"""

import json
import os
import re
import random
import time
import uuid
import logging
from datetime import datetime
from difflib import SequenceMatcher

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust these to your needs
# ---------------------------------------------------------------------------

INPUT_PATH   = "../data/nhg_subset_guidelines.jsonl"
OUTPUT_DIR   = "results"
N_QUESTIONS  = 3      # questions per chunk; see README for sizing guidance
MAX_CHUNKS   = None    # set to an integer (e.g. 100) for a dry run
MIN_TOKENS   = 50      # skip chunks with fewer tokens
SEED         = 1
MAX_RETRIES  = 3       # retries per chunk on parse/API failure
MODEL        = "gpt-4o"  # gpt-4o for production

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PROMPTS — best-performing configuration (few_shot_cot)
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
2. Verwerk bewust klinisch relevante details die de beslissing beïnvloeden.
3. Het antwoord moet volledig onderbouwd kunnen worden vanuit de gegeven NHG-tekst.
4. Vermijd vragen die beginnen met "Wat zegt de richtlijn over..." of "Noem de criteria voor...".
5. De casus moet de arts dwingen tot redeneren, niet tot het opzoeken van een definitie.
6. Schrijf in natuurlijk Nederlands, zoals een ervaren huisarts zou communiceren.
7. Verzin GEEN feiten die niet in de tekst staan.
8. source_span is een LETTERLIJK CITAAT — kopieer het woord voor woord uit de tekst.
9. Controleer: plak source_span terug in de tekst. Als je het niet kunt vinden, kies een andere zin.
10. Kies ALLEEN volledige zinnen die eindigen op een punt.

## Regels antwoord
1. Het antwoord (gt_answer) bestaat uit maximaal 2-3 zinnen.
2. Geef alleen de klinische conclusie en de directe onderbouwing vanuit de tekst.
3. Geen opsommingen, geen uitgebreide uitleg.

Geef je uitvoer ALLEEN als JSON-array, zonder markdown of uitleg.
"""

FEW_SHOT_COT_EXAMPLE = """
Hieronder volgt een voorbeeld van een hoogwaardige KFQ die gebruikmaakt van een expliciet, 
intern klinisch redeneerproces (Chain-of-Thought):

Tekst: "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten. 
Voer spirometrie uit bij patiënten met klachten die passen bij astma of COPD."

Gewenste output:
[
  {
    "explanation": "Beslismoment: Aantonen of uitsluiten van obstructie bij vermoeden van astma/COPD. Relevante klinische factoren: 3 maanden hoestklachten, rookhistorie, normale longauscultatie sluit obstructie niet uit. Gekozen zin: 'Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten.'",
    "query": "Mevrouw De Vries, 52 jaar, komt op uw spreekuur met al drie maanden aanhoudende hoestklachten en kortademigheid bij inspanning. Ze rookt 10 jaar, een half pakje per dag. Haar longauscultatie is normaal. U overweegt astma of COPD. Welk aanvullend onderzoek is als eerste aangewezen om obstructie aan te tonen of uit te sluiten?",
    "gt_answer": "Spirometrie is aangewezen om obstructie aan te tonen of uit te sluiten bij klachten die passen bij astma of COPD. Dit is de geëigende methode volgens de richtlijn.",
    "source_span": "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten.",
    "retrieval_query": "spirometrie obstructie aantonen uitsluiten astma COPD aanvullend onderzoek"
  }
]

Genereer nu vergelijkbare KFQs inclusief de 'explanation' stap op basis van de onderstaande tekst.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + FEW_SHOT_COT_EXAMPLE

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()


def validate_source_span(qa: dict, source_text: str) -> bool:
    """
    Verifies that source_span is genuinely grounded in the source chunk.
    Uses three increasingly lenient checks to handle minor whitespace/
    punctuation variation while still catching hallucinated spans.
    """
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False

    norm_span   = normalize(qa["source_span"])
    norm_source = normalize(source_text)

    # 1. Exact substring match
    if norm_span in norm_source:
        return True

    # 2. Full-string fuzzy match
    if SequenceMatcher(None, norm_span, norm_source).ratio() > 0.82:
        return True

    # 3. Sliding-window fuzzy match over the source
    span_words   = norm_span.split()
    source_words = norm_source.split()
    window_size  = len(span_words)
    for i in range(max(0, len(source_words) - window_size + 1)):
        window = " ".join(source_words[i: i + window_size])
        if SequenceMatcher(None, norm_span, window).ratio() > 0.88:
            return True

    log.warning("  ⚠ source_span not found: %s…", qa["source_span"][:60])
    return False


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def generate_for_chunk(
    chunk: dict,
    client: OpenAI,
    n: int = N_QUESTIONS,
) -> list[dict]:
    """
    Calls the OpenAI API and returns a list of validated QA dicts.
    Retries up to MAX_RETRIES times on JSON/API errors.
    """
    text = chunk["text"]
    user_prompt = f"""NHG-richtlijntekst ({chunk.get('section_path', 'onbekend')}):

--- BEGIN BRONTEKST ---
{text}
--- EINDE BRONTEKST ---

INSTRUCTIE: source_span MOET een aaneengesloten reeks woorden zijn die letterlijk voorkomt
in de bovenstaande brontekst. Kopieer woord voor woord, niet uit geheugen.

Genereer {n} hoogwaardige klinische QA-paren op basis van bovenstaande tekst.
Geef uitvoer als JSON-array."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.7,
            )
            raw     = response.choices[0].message.content
            cleaned = strip_markdown_fences(raw)
            qa_list = json.loads(cleaned)

            valid = [qa for qa in qa_list if validate_source_span(qa, text)]

            if valid:
                return valid
            else:
                log.warning("  Attempt %d/%d — 0 valid spans, retrying…", attempt, MAX_RETRIES)

        except json.JSONDecodeError as e:
            log.warning("  Attempt %d/%d — JSON error: %s", attempt, MAX_RETRIES, e)
        except Exception as e:
            log.warning("  Attempt %d/%d — API error: %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)   # exponential back-off

    return []


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Load & filter chunks ──────────────────────────────────────────────
    log.info("Loading chunks from %s", INPUT_PATH)
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    all_chunks = [
        c for c in all_chunks
        if c.get("tokens", len(c["text"].split())) >= MIN_TOKENS
    ]
    log.info("Loaded %d chunks (≥ %d tokens)", len(all_chunks), MIN_TOKENS)

    random.seed(SEED)
    random.shuffle(all_chunks)

    limit  = MAX_CHUNKS if MAX_CHUNKS is not None else len(all_chunks)
    chunks = all_chunks[:limit]
    log.info("Using %d chunks (MAX_CHUNKS=%s)", len(chunks), MAX_CHUNKS)

    # ── Generate ──────────────────────────────────────────────────────────
    results  : list[dict] = []
    chunk_log: list[dict] = []
    failed   : list[str]  = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"index_{i}")
        log.info(
            "[%d/%d] %s — %s",
            i + 1, len(chunks), chunk_id, chunk.get("section_path", "?")
        )

        qa_pairs = generate_for_chunk(chunk, client, n=N_QUESTIONS)

        if not qa_pairs:
            failed.append(chunk_id)
            log.warning("  ✗ No valid QA pairs for %s", chunk_id)
            chunk_log.append({"chunk_id": chunk_id, "n_generated": 0, "status": "failed"})
            continue

        log.info("  ✓ %d valid pair(s)", len(qa_pairs))
        chunk_log.append({"chunk_id": chunk_id, "n_generated": len(qa_pairs), "status": "ok"})

        for qa in qa_pairs:
            results.append({
                "query_id"       : str(uuid.uuid4())[:8],
                "strategy"       : "few_shot_cot",
                "model"          : MODEL,
                "doc_id"         : chunk.get("doc_id", ""),
                "chunk_id"       : chunk_id,
                "section_path"   : chunk.get("section_path", ""),
                # CoT intermediate step — kept for interpretability / audit
                "explanation"    : qa.get("explanation", ""),
                # Core QA fields
                "query"          : qa["query"],
                "gt_answer"      : qa["gt_answer"],
                "source_span"    : qa.get("source_span", ""),
                "retrieval_query": qa.get("retrieval_query", ""),
            })

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_path = os.path.join(OUTPUT_DIR, f"qa_dataset_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved %d QA pairs → %s", len(results), json_path)

    # JSONL (one record per line — convenient for streaming / fine-tuning)
    jsonl_path = os.path.join(OUTPUT_DIR, f"qa_dataset_{ts}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved JSONL → %s", jsonl_path)

    # Generation log
    log_path = os.path.join(OUTPUT_DIR, f"generation_log_{ts}.json")
    summary  = {
        "timestamp"      : ts,
        "model"          : MODEL,
        "strategy"       : "few_shot_cot",
        "n_chunks"       : len(chunks),
        "n_generated"    : len(results),
        "n_failed_chunks": len(failed),
        "failed_chunk_ids": failed,
        "chunk_log"      : chunk_log,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Generation log → %s", log_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Dataset generation complete")
    print(f"  Chunks processed : {len(chunks)}")
    print(f"  QA pairs saved   : {len(results)}")
    print(f"  Failed chunks    : {len(failed)}")
    print(f"  Output (JSON)    : {json_path}")
    print(f"  Output (JSONL)   : {jsonl_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()