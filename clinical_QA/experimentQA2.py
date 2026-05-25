"""
Benchmark generation script for clinical vignette QA dataset.
Compares zero-shot, few-shot, CoT, and self-critique prompt strategies.

Metrics:
  - Round-trip retrieval hit rate (direct span + embedding cascade)
  - BERTScore recall vs source chunk
  - Bidirectional NLI faithfulness / hallucination
  - LLM-as-Judge (GPT-4o-mini, 5 dimensions, 0-5 scale)
  - Repeated-measures ANOVA across strategies
"""

import json
import uuid
import re
import time
import os
import random
import numpy as np
from difflib import SequenceMatcher
from dotenv import load_dotenv

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

import pandas as pd
from statsmodels.stats.anova import AnovaRM

load_dotenv()

from openai import OpenAI
from bert_score import score as bert_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-mini"
MAX_RETRIES = 1
MIN_TOKENS = 50
MAX_CHUNKS = 30  # Set to None for full run
N_QUESTIONS = 1
TOP_K = 10
SEED = 1

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
# NLI / ENTAILMENT
# ---------------------------------------------------------------------------

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print(f"Loading NLI model: {NLI_MODEL_NAME}")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
nli_model.eval()

_id2label = nli_model.config.id2label
print(f"NLI label mapping: {_id2label}")


def nli_probs(premise: str, hypothesis: str) -> dict[str, float]:
    """
    MNLI sentence-pair inference.
    Returns {"entailment": float, "neutral": float, "contradiction": float}.

    Uses truncation="only_first" to ensure the hypothesis is never truncated —
    only the premise is shortened if the combined length exceeds 512 tokens.
    """
    try:
        inputs = nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
        )
    except Exception:
        # fallback: truncate both sides if HF tokenizer still complains
        inputs = nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    return {_id2label[i].lower(): probs[i].item() for i in range(len(_id2label))}


def sentence_level_nli(chunk_text, answer):
    sentences = sent_tokenize(chunk_text)

    ent, con, neu = [], [], []

    for s in sentences:
        probs = nli_probs(s, answer)
        ent.append(probs["entailment"])
        con.append(probs["contradiction"])
        neu.append(probs["neutral"])

    return {
        "entailment": np.mean(ent),
        "contradiction": np.mean(con),
        "neutral": np.mean(neu),
    }


def _nli_sanity_check():
    """
    Verify the NLI model produces sensible outputs before the main pipeline runs.
    Prints self-entailment and clear-contradiction scores.
    Expected: self-entailment ~1.0, contradiction ~1.0 for opposite sentences.
    If self-entailment < 0.5 the model or label mapping is broken.
    """
    self_result = nli_probs("De patiënt heeft astma.", "De patiënt heeft astma.")
    contra_result = nli_probs("De patiënt heeft astma.", "De patiënt heeft geen astma.")

    print(f"\nNLI sanity check:")
    print(f"  Self-entailment   (expect ~1.0): {self_result.get('entailment', 0):.3f}")
    print(f"  Contradiction     (expect ~1.0): {contra_result.get('contradiction', 0):.3f}")
    print(f"  Full self result:   {self_result}")
    print(f"  Full contra result: {contra_result}")

    if self_result.get("entailment", 0) < 0.5:
        print("  ⚠ WARNING: NLI self-entailment is low — check model and label mapping!")
    else:
        print("  ✓ NLI model looks correct.")


def nli_faithfulness_and_hallucination(results, chunks):
    """
    Evaluates generated answers by matching each sentence against the single
    best supporting sentence in the source text using maximum entailment.
    """
    faith_scores, hall_scores, neutral_scores = [], [], []

    for item in results:
        chunk = chunks.get(item["chunk_id"])
        source_context = item.get("source_span", "").strip()

        if (not source_context or len(source_context) < 10) and chunk:
            source_context = chunk["text"]

        if not source_context:
            item["nli_faithfulness"] = 0.0
            item["nli_hallucination"] = 0.0
            item["nli_neutral"] = 1.0
            continue

        source_sentences = sent_tokenize(source_context)
        answer_sentences = sent_tokenize(item["gt_answer"])

        item_faithfulness = []
        item_hallucination = []
        item_neutral = []

        for a_sent in answer_sentences:
            best_ent = -1.0
            best_con = 0.0
            best_neu = 1.0

            for s_sent in source_sentences:
                probs = nli_probs(s_sent, a_sent)

                if probs["entailment"] > best_ent:
                    best_ent = probs["entailment"]
                    best_con = probs["contradiction"]
                    best_neu = probs["neutral"]

            item_faithfulness.append(best_ent)
            item_hallucination.append(best_con)
            item_neutral.append(best_neu)

        ent = float(np.mean(item_faithfulness)) if item_faithfulness else 0.0
        con = float(np.mean(item_hallucination)) if item_hallucination else 0.0
        neu = float(np.mean(item_neutral)) if item_neutral else 0.0

        item["nli_faithfulness"] = ent
        item["nli_hallucination"] = con
        item["nli_neutral"] = neu

        faith_scores.append(ent)
        hall_scores.append(con)
        neutral_scores.append(neu)

    return (
        float(np.mean(faith_scores)) if faith_scores else 0.0,
        float(np.mean(hall_scores)) if hall_scores else 0.0,
        float(np.mean(neutral_scores)) if neutral_scores else 0.0,
    )

# ---------------------------------------------------------------------------
# LLM-JUDGE
# ---------------------------------------------------------------------------

def llm_judge_score(item: dict, chunk_text: str) -> float:
    prompt = f"""
Je beoordeelt een klinische vraag en antwoord op basis van een NHG-richtlijn.

## BRONTEKST:
\"\"\"{chunk_text}\"\"\"

## VRAAG:
{item["query"]}

## ANTWOORD:
{item["gt_answer"]}

Beoordeel streng op een schaal van 0-5 continuous:
1. Correctheid — Is het antwoord feitelijk juist volgens de richtlijn? 
2. Richtlijntrouw — Sluit het antwoord nauw aan op de letterlijke inhoud van de richtlijn?
3. Redeneerkwaliteit — Is de redenering logisch en volledig onderbouwd?
4. Veiligheid — Is het antwoord klinisch veilig volgens de richtlijn?
5. Afwezigheid van hallucinaties — Bevat het antwoord GEEN informatie die niet in de richtlijn staat?

Geef niet zomaar alles een 4 of 5
Wees extra streng op richtlijntrouw en hallucinaties. Als het antwoord niet expliciet in de richtlijn staat, moet de score laag zijn (0-2).

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
        item["llm_judge_details"] = scores
        mean_score = float(np.mean([
            scores["correctness"],
            scores["guideline_adherence"],
            scores["reasoning"],
            scores["safety"],
            scores["non_hallucination"],
        ]))
        item["llm_judge"] = round(mean_score, 6)
        return mean_score
    except Exception as e:
        print(f"  LLM judge parse error for {item.get('query_id', '?')}: {e}")
        item["llm_judge"] = 0.0
        return 0.0


def llm_judge_all(results: list[dict], chunks: dict[str, dict]) -> float:
    scores = []
    for item in results:
        chunk = chunks.get(item["chunk_id"])
        if not chunk:
            item["llm_judge"] = 0.0
            continue
        scores.append(llm_judge_score(item, chunk["text"]))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# EMBEDDING + RETRIEVAL
# ---------------------------------------------------------------------------

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-embedding-2"
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
    for attempt in range(max_retries):
        try:
            return run_query(prompt, embeddings_file, top_k)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (2 ** attempt) * 5
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


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def validate_qa(qa: dict, source_text: str) -> bool:
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False

    norm_span = normalize(qa["source_span"])
    norm_source = normalize(source_text)

    if norm_span in norm_source:
        return True

    if SequenceMatcher(None, norm_span, norm_source).ratio() > 0.82:
        return True

    span_words = norm_span.split()
    source_words = norm_source.split()
    window_size = len(span_words)

    for i in range(len(source_words) - window_size + 1):
        window = " ".join(source_words[i: i + window_size])
        if SequenceMatcher(None, norm_span, window).ratio() > 0.88:  # Balanced step-down boundary
            return True

    print(f"    ⚠ source_span not found in chunk: {qa['source_span'][:60]}...")
    return False


def critique_and_refine(chunk: dict, draft_qa: dict) -> dict:
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
        return draft_qa

def generate_questions(
        chunk: dict, system_prompt: str, n: int = N_QUESTIONS
) -> list[dict]:
    text = chunk["text"]
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
# VALIDATION METRICS
# ---------------------------------------------------------------------------

def round_trip_hit_rate(
        results: list[dict],
        embeddings_file: str,
        chunks_lookup: dict[str, dict],
) -> dict[str, float]:
    """
    Evaluates both text grounding and dense retriever accuracy independently.
    Fixes the 0.00% embedding hit rate bug caused by early loop short-circuiting.
    """
    hits_direct = 0
    hits_embedding = 0
    misses = []

    for item in results:
        source_span = item.get("source_span", "")
        chunk = chunks_lookup.get(item["chunk_id"])

        # Track grounding status independently
        is_grounded = False
        if source_span and chunk:
            if normalize(source_span) in normalize(chunk["text"]):
                hits_direct += 1
                is_grounded = True
                item["round_trip_probe"] = "direct_span"

        # Probes 2-4: Run embedding retrieval regardless of grounding status
        probes = [
            ("retrieval_query", item.get("retrieval_query", "")),
            ("source_span", source_span),
            ("query", item.get("query", "")),
        ]

        hit_retriever = False
        try:
            for probe_name, probe_text in probes:
                if not probe_text:
                    continue
                retrieved = run_query_with_backoff(probe_text, embeddings_file, top_k=TOP_K)

                if item["chunk_id"] in retrieved:
                    hits_embedding += 1
                    hit_retriever = True
                    # Only update the tracking string if it wasn't already caught by direct span
                    if not is_grounded:
                        item["round_trip_probe"] = f"embedding_{probe_name}"
                    break

            if hit_retriever:
                if is_grounded:
                    item["round_trip_probe"] = "both_span_and_embedding"
            else:
                if not is_grounded:
                    item["round_trip_probe"] = "miss"
                    misses.append({
                        "query_id": item["query_id"],
                        "chunk_id": item["chunk_id"],
                        "query_preview": item["query"][:80],
                    })
                else:
                    item["round_trip_probe"] = "grounded_but_retrieval_miss"

        except Exception as e:
            print(f"  Round-trip error for {item['query_id']}: {e}")
            item["round_trip_probe"] = "error"

    total = len(results)
    print(f"  Round-trip breakdown:")
    print(f"    Direct span hits (grounding quality): {hits_direct}/{total}")
    print(f"    Embedding hits (retriever index accuracy): {hits_embedding}/{total}")
    print(f"    Total true system misses:             {len(misses)}/{total}")

    if misses:
        print(f"  Embedding misses:")
        for m in misses:
            print(f"    ✗ {m['chunk_id']} — {m['query_preview']}...")

    return {
        "span_grounding_rate": hits_direct / total if total > 0 else 0.0,
        "retrieval_hit_rate": hits_embedding / total if total > 0 else 0.0
    }


def bertscore_vs_source(results: list[dict], chunks: dict[str, dict]) -> float:
    hypotheses = [item["gt_answer"] for item in results]

    # FIX: Compare against the complete guideline text chunk, not just the tiny snippet
    references = [chunks[item["chunk_id"]]["text"] for item in results if
                  item["chunk_id"] in chunks]

    if not hypotheses or len(hypotheses) != len(references):
        return 0.0

    _, R, _ = bert_score(hypotheses, references, lang="nl", verbose=False)
    return float(R.mean())


# ---------------------------------------------------------------------------
# REPEATED-MEASURES ANOVA
# ---------------------------------------------------------------------------

def repeated_measures_anova(
        strategy_results: dict[str, list[dict]], metric: str
) -> tuple[pd.DataFrame, object]:
    rows = []
    for strategy, results in strategy_results.items():
        for r in results:
            if metric not in r:
                raise KeyError(
                    f"Metric '{metric}' not found in result {r.get('query_id')}. "
                    f"Available keys: {list(r.keys())}"
                )
            rows.append({
                "chunk_id": r["chunk_id"],
                "strategy": strategy,
                "score": r[metric],
            })

    df = pd.DataFrame(rows)
    df = df.groupby(["chunk_id", "strategy"], as_index=False)["score"].mean()

    strategy_counts = df.groupby("chunk_id")["strategy"].nunique()
    n_strategies = df["strategy"].nunique()
    balanced_chunks = strategy_counts[strategy_counts == n_strategies].index
    df = df[df["chunk_id"].isin(balanced_chunks)]

    if df["chunk_id"].nunique() < 2:
        print(f"  ⚠ Not enough balanced subjects for ANOVA on '{metric}' — skipping.")
        return df, None

    anova = AnovaRM(
        data=df,
        depvar="score",
        subject="chunk_id",
        within=["strategy"],
    ).fit()

    return df, anova


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
                    "retrieval_query": qa.get("retrieval_query", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "section_path": chunk.get("section_path", ""),
                    "strategy": strategy_name,
                })

    print(f"\nValidating {len(results)} QA pairs...")

    rt_scores = round_trip_hit_rate(results, EMBEDDINGS_PATH, chunks_lookup)
    rt_score = rt_scores["retrieval_hit_rate"]
    bs_score = bertscore_vs_source(results, chunks_lookup)
    faith_score, hall_score, neutral_score = nli_faithfulness_and_hallucination(results,
                                                                                chunks_lookup)
    judge_score = llm_judge_all(results, chunks_lookup)

    # Inside run_strategy()...

    # Non-linear penalty calculation: heavily punish clear hallucinations
    hallucination_penalty = np.mean([
        1.5 if item["nli_hallucination"] > 0.35 else item["nli_hallucination"]
        for item in results
    ]) if results else 0.0

    metrics = {
        "strategy": strategy_name,
        "n_generated": len(results),
        "n_failed_chunks": len(failed),
        "round_trip_hit_rate": round(rt_score, 4),
        "bertscore_recall_vs_source": round(bs_score, 4),
        "nli_faithfulness": round(faith_score, 4),
        "nli_hallucination": round(hall_score, 4),
        "nli_neutral": round(neutral_score, 4),
        "llm_judge_score": round(judge_score, 4),

        # CALIBRATED OBJECTIVE FUNCTION:
        "combined_score": round(
            (0.40 * faith_score) -
            (0.25 * hallucination_penalty) +
            (0.25 * (judge_score / 5.0)) +
            (0.10 * bs_score),
            4,
        ),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    details_path = os.path.join(OUTPUT_DIR, f"{strategy_name}_details2.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved detailed per-question scores to {details_path}")

    print(f"\nResults for [{strategy_name}]:")
    print(f"  Round-trip hit rate (@{TOP_K}): {rt_score:.2%}")
    print(f"  BERTScore recall:              {bs_score:.4f}")
    print(f"  NLI faithfulness:              {faith_score:.4f}")
    print(f"  NLI hallucination:             {hall_score:.4f}")
    print(f"  NLI neutral:                   {neutral_score:.4f}")
    print(f"  LLM-Judge score (0-5):         {judge_score:.4f}")
    print(f"  Combined score:                {metrics['combined_score']:.4f}")

    return results, metrics


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    all_chunks = [
        c for c in all_chunks
        if c.get("tokens", len(c["text"].split())) >= MIN_TOKENS
    ]

    random.seed(1)
    random.shuffle(all_chunks)

    chunks_lookup = {c["chunk_id"]: c for c in all_chunks}
    print(f"Loaded {len(all_chunks)} chunks (after filtering < {MIN_TOKENS} tokens)")

    # Sanity-check NLI before running any strategies
    # _nli_sanity_check()

    strategies = [
        ("few_shot", SYSTEM_PROMPT_FEW_SHOT),
        ("cot", SYSTEM_PROMPT_COT),

        # New testing groups
        ("few_shot_cot", SYSTEM_PROMPT_FEW_SHOT_COT),
        ("few_shot_refine", SYSTEM_PROMPT_FEW_SHOT),
        ("few_shot_cot_refine", SYSTEM_PROMPT_FEW_SHOT_COT)
    ]

    all_metrics: list[dict] = []
    strategy_results: dict[str, list[dict]] = {}

    for name, prompt in strategies:
        results, metrics = run_strategy(name, prompt, all_chunks, chunks_lookup)
        strategy_results[name] = results
        all_metrics.append(metrics)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    anova_metrics = ["nli_faithfulness", "nli_hallucination", "llm_judge"]
    anova_results = {}

    for metric in anova_metrics:
        print(f"\n===== ANOVA for {metric} =====")
        df, anova = repeated_measures_anova(strategy_results, metric)
        df.to_csv(os.path.join(OUTPUT_DIR, f"anova_{metric}_data.csv"), index=False)
        if anova is not None:
            print(anova)
            anova_results[metric] = {"anova_table": str(anova), "n_samples": len(df)}
        else:
            anova_results[metric] = {"anova_table": "skipped — insufficient balanced subjects"}

    metrics_path = os.path.join(OUTPUT_DIR, "strategy_comparison2.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nStrategy comparison saved to {metrics_path}")

    best = max(all_metrics, key=lambda m: m["combined_score"])
    print(f"\nBest strategy: {best['strategy']} (combined score: {best['combined_score']})")

    best_results = strategy_results[best["strategy"]]
    output = {"results": best_results, "strategy_comparison": all_metrics}
    qa_path = os.path.join(OUTPUT_DIR, "Experiment2QA.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Final benchmark ({len(best_results)} QA pairs) saved to {qa_path}")

    anova_path = os.path.join(OUTPUT_DIR, "anova_results2.json")
    with open(anova_path, "w", encoding="utf-8") as f:
        json.dump(anova_results, f, indent=2, ensure_ascii=False)
    print(f"ANOVA results saved to {anova_path}")


# ---------------------------------------------------------------------------
# EXPERIMENT 2: HYBRID CONFIGURATIONS & PROMPTS
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FEW_SHOT_COT = SYSTEM_PROMPT_BASE + """
Hieronder volgt een voorbeeld van een hoogwaardige KFQ die gebruikmaakt van een expliciet, intern klinisch redeneerproces (Chain-of-Thought):

Tekst: "Spirometrie is de aangewezen methode om obstructie aan te tonen of uit te sluiten. Voer spirometrie uit bij patiënten met klachten die passen bij astma of COPD."

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

Genereer nu een vergelijkbare KFQ inclusief de 'explanation' stap op basis van de onderstaande tekst. Volg de JSON structuur nauwkeurig.
"""


# Update validation function to bypass the temporary "explanation" key during structure checks
def validate_qa_hybrid(qa: dict, source_text: str) -> bool:
    if not all(k in qa for k in ("query", "gt_answer", "source_span")):
        return False

    norm_span = normalize(qa["source_span"])
    norm_source = normalize(source_text)

    if norm_span in norm_source:
        return True

    if SequenceMatcher(None, norm_span, norm_source).ratio() > 0.82:
        return True
    return False


def few_shot_critique_and_refine(chunk: dict, draft_qa: dict) -> dict:
    """
    Advanced multi-turn refiner guided by an explicit few-shot critique exemplar
    to restrict structural hallucination behaviors observed in Exp 1.
    """
    few_shot_refine_prompt = f"""
Je bent een kritische medische onderwijsassistent. Je taak is om een concept-KFQ te verfijnen op basis van de brontekst. Je verbetert uitsluitend de formulering en de structuur van de JSON objecten. Je mag NOOIT nieuwe klinische claims verzinnen of de letterlijke 'source_span' aanpassen.

### VOORBEELD VAN EEN GOEDE VERFIJNING
BRONTEKST: "Spirometrie is de aangewezen methode om obstructie aan te tonen."
CONCEPT-JSON: {{"query": "Vraag over spirometrie?", "gt_answer": "Spirometrie doen en bloedprikken.", "source_span": "Spirometrie is de aangewezen methode"}}
VERFIJND-JSON: [{{ "query": "Mevrouw De Vries, 52 jaar... Welk aanvullend onderzoek zet u in?", "gt_answer": "Spirometrie is de aangewezen methode om obstructie aan te tonen.", "source_span": "Spirometrie is de aangewezen methode om obstructie aan te tonen." }}]

Pas deze strikte methodiek nu toe op het onderstaande element:
BRONTEKST: \"\"\"{chunk['text']}\"\"\"
CONCEPT-JSON: {json.dumps(draft_qa, ensure_ascii=False)}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"} if "json_object" in str(
                dir(client.chat.completions)) else None,
            messages=[{"role": "user", "content": few_shot_refine_prompt}],
            temperature=0.2,
        )
        raw = strip_markdown_fences(response.choices[0].message.content)
        data = json.loads(raw)
        return data[0] if isinstance(data, list) else data.get("output", data)
    except Exception:
        return draft_qa


# ---------------------------------------------------------------------------
# ROUTING CONTROLLER MODIFICATION
# ---------------------------------------------------------------------------

def generate_questions_with_strategy(
        chunk: dict, system_prompt: str, strategy_name: str
) -> list[dict]:
    # Experiment 1 Routes
    if strategy_name == "self_critique":
        drafts = generate_questions(chunk, SYSTEM_PROMPT_BASE)
        return [critique_and_refine(chunk, draft) for draft in drafts]

    # Experiment 2: Hybrid Routes
    elif strategy_name == "few_shot_cot":
        return generate_questions(chunk, SYSTEM_PROMPT_FEW_SHOT_COT)

    elif strategy_name == "few_shot_refine":
        drafts = generate_questions(chunk, SYSTEM_PROMPT_FEW_SHOT)
        return [few_shot_critique_and_refine(chunk, draft) for draft in drafts]

    elif strategy_name == "few_shot_cot_refine":
        drafts = generate_questions(chunk, SYSTEM_PROMPT_FEW_SHOT_COT)
        return [few_shot_critique_and_refine(chunk, draft) for draft in drafts]

    return generate_questions(chunk, system_prompt)

if __name__ == "__main__":
    main()


