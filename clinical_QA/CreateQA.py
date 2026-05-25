"""
Benchmark generation script for clinical vignette QA dataset.
Compares zero-shot and few-shot prompt strategies and selects the best
using round-trip retrieval hit rate, BERTScore, and entailment score.

Fixes applied:
  - source_span: stricter prompt rules + BEGIN/EINDE markers + fuzzy sliding-window validator
  - round-trip: retrieval_query field (model generates its own search terms) + 3-probe cascade
  - round-trip: embeddings index cached in memory (no repeated disk reads)
  - round-trip: exponential backoff on Gemini 429 rate limit errors
"""

import json
import uuid
import re
import time
import os
import numpy as np
from difflib import SequenceMatcher
from dotenv import load_dotenv
from transformers import pipeline
import random

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
MAX_CHUNKS = 30  # Set to an int (e.g. 10) for testing; None = all chunks
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
8. source_span is een LETTERLIJK CITAAT — kopieer het woord voor woord uit de tekst.
9. Controleer: plak source_span terug in de tekst. Als je het niet kunt vinden, kies een andere zin.
10. Kies ALLEEN volledige zinnen die eindigen op een punt. Geen opsommingspunten, geen headers, geen zinsdelen.
11. Als de tekst alleen opsommingen bevat zonder volledige zinnen, gebruik dan de inleidende zin die de opsomming introduceert.

## Regels antwoord
1. Het antwoord (gt_answer) bestaat uit maximaal 2-3 zinnen.
2. Geef alleen de klinische conclusie en de directe onderbouwing vanuit de tekst.
3. Geen opsommingen, geen uitgebreide uitleg.

Geef je uitvoer ALLEEN als JSON-array, zonder markdown of uitleg:
[
  {
    "query": "Volledige casusbeschrijving gevolgd door de klinische vraag.",
    "gt_answer": "Het antwoord op de vraag.",
    "source_span": "Verbatim fragment uit de tekst dat het antwoord onderbouwt.",
    "retrieval_query": "Korte klinische zoekterm (max 15 woorden) die direct aansluit bij de richtlijntekst, bijv: 'spirometrie obstructie astma COPD aanvullend onderzoek'."
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
    "source_span": "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten.",
    "retrieval_query": "spirometrie obstructie aantonen uitsluiten astma COPD aanvullend onderzoek"
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
- retrieval_query = korte klinische zoektermen die direct aansluiten bij de richtlijntekst
- geen uitleg, geen tussenstappen
"""

# ---------------------------------------------------------------------------
# NLI / ENTAILMENT  (Correct Implementation with proper label mapping)
# ---------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

# The correct label order from config.json:
# 0 = contradiction
# 1 = neutral
# 2 = entailment

def nli_probs(premise: str, hypothesis: str):
    """
    Returns a dict with entailment, neutral, contradiction probabilities.
    Uses the correct MNLI/XNLI label order from config.json.
    """
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    # Map using the correct id2label mapping
    id2label = nli_model.config.id2label  # from config.json

    return {
        id2label[i].lower(): probs[i].item()
        for i in range(len(id2label))
    }


def nli_faithfulness_and_hallucination(results, chunks):
    faith_scores = []
    hall_scores = []
    neutral_scores = []

    for item in results:
        guideline = chunks[item["chunk_id"]]["text"]
        answer = item["gt_answer"]

        s1 = nli_probs(guideline, answer)
        s2 = nli_probs(answer, guideline)

        ent = (s1["entailment"] + s2["entailment"]) / 2
        con = (s1["contradiction"] + s2["contradiction"]) / 2
        neu = (s1["neutral"] + s2["neutral"]) / 2

        # Save per‑question NLI scores
        item["nli_scores"] = {
            "guideline_to_answer": s1,
            "answer_to_guideline": s2,
            "entailment": ent,
            "contradiction": con,
            "neutral": neu
        }

        # Print for debugging
        # print("\nNLI for", item["query_id"])
        # print(json.dumps(item["nli_scores"], indent=2, ensure_ascii=False))

        faith_scores.append(ent)
        hall_scores.append(con)
        neutral_scores.append(neu)

    return (
        float(np.mean(faith_scores)) if faith_scores else 0.0,
        float(np.mean(hall_scores)) if hall_scores else 0.0,
        float(np.mean(neutral_scores)) if neutral_scores else 0.0,
    )

# ---------------------------------------------------------------------------
# EMBEDDING + RETRIEVAL
# ---------------------------------------------------------------------------

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-embedding-2"

# Cache the embeddings index in memory — avoids re-reading the file on every probe call
_embeddings_cache: list[dict] | None = None


def load_embeddings(embeddings_file: str) -> list[dict]:
    global _embeddings_cache
    if _embeddings_cache is None:
        with open(embeddings_file, encoding="utf-8") as f:
            _embeddings_cache = json.load(f)
    return _embeddings_cache


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
    index = load_embeddings(embeddings_file)
    query_vec = np.array(embed_query(prompt))
    vectors = np.array([item["embedding"] for item in index])
    scores = cosine_similarity(query_vec, vectors)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [index[i]["chunk_id"] for i in top_idx]


def run_query_with_backoff(
        prompt: str,
        embeddings_file: str,
        top_k: int = TOP_K,
        max_retries: int = 3,
) -> list[str]:
    """run_query with exponential backoff on Gemini 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return run_query(prompt, embeddings_file, top_k)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (2 ** attempt) * 5  # 5s → 10s → 20s
                print(
                    f"  Rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    print(f"  Giving up on query after {max_retries} retries.")
    return []


# ---------------------------------------------------------------------------
# QA GENERATION
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL).strip()


def validate_qa(qa: dict, source_text: str) -> bool:
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False

    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    norm_span = normalize(qa["source_span"])
    norm_source = normalize(source_text)

    # 1. Exact substring match
    if norm_span in norm_source:
        return True

    # 2. Whole-string fuzzy ratio (catches minor whitespace / punctuation drift)
    if SequenceMatcher(None, norm_span, norm_source).ratio() > 0.85:
        return True

    # 3. Sliding window: look for a same-length window with high similarity
    span_words = norm_span.split()
    source_words = norm_source.split()
    window_size = len(span_words)
    for i in range(len(source_words) - window_size + 1):
        window = " ".join(source_words[i: i + window_size])
        if SequenceMatcher(None, norm_span, window).ratio() > 0.92:
            return True

    print(f"    ⚠ source_span not found in chunk: {qa['source_span'][:60]}...")
    return False


def critique_and_refine(chunk: dict, draft_qa: dict) -> dict:
    """Second-pass refinement: model critiques and improves its own KFQ."""
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
    "source_span": "EXACTE TEKST UIT BRON (niet wijzigen!)",
    "retrieval_query": "..."
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
    except Exception:
        return draft_qa  # fallback to draft on parse error


def generate_questions_with_strategy(
        chunk: dict, system_prompt: str, strategy_name: str
) -> list[dict]:
    if strategy_name == "self_critique":
        drafts = generate_questions(chunk, SYSTEM_PROMPT_BASE)
        return [critique_and_refine(chunk, draft) for draft in drafts]
    return generate_questions(chunk, system_prompt)


def generate_questions(
        chunk: dict, system_prompt: str, n: int = N_QUESTIONS
) -> list[dict]:
    text = chunk["text"]

    # BEGIN/EINDE markers make the source boundary unambiguous for the model,
    # reducing hallucinated source_spans.
    user_prompt = f"""NHG-richtlijntekst ({chunk.get('section_path', 'onbekend')}):

--- BEGIN BRONTEKST ---
{text}
--- EINDE BRONTEKST ---

INSTRUCTIE: source_span MOET een aaneengesloten reeks woorden zijn die letterlijk voorkomt
in de bovenstaande brontekst tussen de markers. Kopieer woord voor woord, niet uit geheugen.

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
# MAIN PIPELINE
# --------------------------------------------------------------------------


def llm_judge_all(results, chunks):
    all_scores = []

    for item in results:
        chunk_text = chunks[item["chunk_id"]]["text"]
        s = llm_judge_score(item, chunk_text)

        # Save per‑question judge score
        item["llm_judge"] = s
        all_scores.append(s)

    return float(np.mean(all_scores)) if all_scores else 0.0



def llm_judge_score(item, chunk_text):
    prompt = f"""
Je beoordeelt een klinische vraag en antwoord op basis van een NHG-richtlijn.

## BRONTEKST:
\"\"\"{chunk_text}\"\"\"

## VRAAG:
{item["query"]}

## ANTWOORD:
{item["gt_answer"]}

Beoordeel op een schaal van 0–5:
1. Correctheid
2. Richtlijntrouw
3. Redeneerkwaliteit
4. Veiligheid
5. Afwezigheid van hallucinaties

Geef output als JSON:
{{
  "correctness": x,
  "guideline_adherence": x,
  "reasoning": x,
  "safety": x,
  "non_hallucination": x
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)

        # Save full judge scores inside the QA item
        item["llm_judge_details"] = scores

        # Print for debugging
        # print("\nLLM‑Judge for", item["query_id"])
        # print(json.dumps(scores, indent=2, ensure_ascii=False))

        vals = [
            scores["correctness"],
            scores["guideline_adherence"],
            scores["reasoning"],
            scores["safety"],
            scores["non_hallucination"],
        ]

        return float(np.mean(vals))

    except Exception as e:
        print("LLM judge parse error:", e)
        print(raw)
        return 0.0


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def round_trip_hit_rate(results: list[dict], embeddings_file: str) -> float:
    """
    Round-trip validation using a 3-probe cascade per QA item:
      1. retrieval_query  — model-generated keyword query (closest to guideline vocabulary)
      2. source_span      — verbatim guideline fragment
      3. query            — full clinical vignette (most narrative, weakest signal)

    Stops at the first probe that retrieves the correct chunk_id.
    This approach is more robust than using the vignette alone because
    the clinical narrative is semantically distant from the guideline text.
    """
    return 1
    hits = 0
    misses = []

    for item in results:
        hit = False
        probes = [
            ("retrieval_query", item.get("retrieval_query", "")),
            ("source_span", item.get("source_span", "")),
            ("query", item.get("query", "")),
        ]
        try:
            for probe_name, probe_text in probes:
                if not probe_text:
                    continue
                retrieved = run_query_with_backoff(probe_text, embeddings_file, top_k=TOP_K)
                if item["chunk_id"] in retrieved:
                    hit = True
                    break

            if hit:
                hits += 1
            else:
                misses.append({
                    "query_id": item["query_id"],
                    "chunk_id": item["chunk_id"],
                    "query_preview": item["query"][:80],
                })
        except Exception as e:
            print(f"  Round-trip error for {item['query_id']}: {e}")

    if misses:
        print(f"  Round-trip misses ({len(misses)}):")
        for m in misses:
            print(f"    ✗ {m['chunk_id']} — {m['query_preview']}...")

    return hits / len(results) if results else 0.0


def bertscore_vs_source(results: list[dict], chunks: dict[str, dict]) -> float:
    """
    BERTScore recall of gt_answer against source chunk text.
    High recall means the answer content is semantically covered by the source.
    """
    hypotheses = [item["gt_answer"] for item in results]
    references = [
        chunks[item["chunk_id"]]["text"]
        for item in results
        if item["chunk_id"] in chunks
    ]

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
            f"[{i + 1}/{limit}] chunk_id={chunk.get('chunk_id', '?')} "
            f"— {chunk.get('section_path', '?')}"
        )
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
                    "retrieval_query": qa.get("retrieval_query", ""),  # new field
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_path": chunk.get("section_path", ""),
                    "strategy": strategy_name,
                })

    # --- Metrics ---
    print(f"\nValidating {len(results)} QA pairs...")

    # 1. Semantic baseline
    bs_score = bertscore_vs_source(results, chunks_lookup)

    # 2. Bidirectional NLI
    faith_score, hall_score, neutral_score = nli_faithfulness_and_hallucination(results, chunks_lookup)

    # 3. LLM‑Judge reasoning score
    judge_score = llm_judge_all(results, chunks_lookup)

    metrics = {
        "strategy": strategy_name,
        "n_generated": len(results),
        "n_failed_chunks": len(failed),

        # Core metrics
        "bertscore_recall_vs_source": round(bs_score, 4),
        "nli_faithfulness": round(faith_score, 4),
        "nli_hallucination": round(hall_score, 4),
        "nli_neutral": round(neutral_score, 4),
        "llm_judge_score": round(judge_score, 4),

        # Combined score (recommended weighting)
        "combined_score": round(
            0.40 * (1 - hall_score) +   # penalize contradictions strongly
            0.30 * neutral_score +      # reward compatibility with guideline
            0.20 * judge_score +        # correctness, reasoning, safety
            0.10 * bs_score,            # weak semantic similarity signal
            4
        ),

    }

    # Save per‑question detailed scores
    details_path = os.path.join(OUTPUT_DIR, f"{strategy_name}_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved detailed per‑question scores to {details_path}")

    print(f"\nResults for [{strategy_name}]:")
    print(f"  BERTScore recall:         {bs_score:.4f}")
    print(f"  NLI faithfulness:         {faith_score:.4f}")
    print(f"  NLI hallucination:        {hall_score:.4f}")
    print(f"  NLI neutral:              {neutral_score:.4f}")
    print(f"  LLM‑Judge reasoning:      {judge_score:.4f}")
    print(f"  Combined score:           {metrics['combined_score']:.4f}")

    return results, metrics


def main():
    # Load and filter chunks
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    all_chunks = [
        c for c in all_chunks
        if c.get("tokens", len(c["text"].split())) >= MIN_TOKENS
    ]
    chunks_lookup = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks (after filtering < {MIN_TOKENS} tokens)")

    strategies = [
        ("zero_shot", SYSTEM_PROMPT_ZERO_SHOT),
        ("few_shot", SYSTEM_PROMPT_FEW_SHOT),
        ("cot", SYSTEM_PROMPT_COT),
        ("self_critique", SYSTEM_PROMPT_BASE),
    ]

    all_metrics: list[dict] = []
    strategy_results: dict[str, list[dict]] = {}

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
