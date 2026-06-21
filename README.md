# RP_NHG_2026 — RAG & Benchmarking for NHG Guidelines

A thesis research project exploring Retrieval-Augmented Generation (RAG) on Dutch medical guidelines (NHG).
This work is part of the Research Project course 2025/2026 at TU Delft. 

---

## Project structure

```
scraping/            — NHG guideline scrapers (Selenium + BeautifulSoup)
pipeline/            — RAG retrieval pipeline (vector, BM25, combined RRF)
factual_benchmark/   — Factual QA benchmark for the RAG pipeline
clinical_benchmark/  — Clinical QA benchmark for the RAG pipeline
model_benchmarking/  — Multi-LLM benchmarking over the RAG pipeline (factual + clinical)
data/                — Scraped NHG guidelines (JSONL)
ragchecker/          — RAGChecker evaluation framework
```

Each module is self-contained and can be run independently. Intermediate outputs are checkpointed as JSON/JSONL files.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download nl_core_news_sm
```

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=...        # query embeddings for vector retrieval
ANTHROPIC_API_KEY=...     # required for model_benchmarking (Claude)
# Additional provider keys (Moonshot, DeepSeek, Z.ai) for model_benchmarking
```

---

## Running the modules

**Scraping**
```bash
python scraping/main.py
```

**Retrieval pipeline**
```bash
python pipeline/new_query_vector.py      # vector retrieval
python pipeline/new_query_traditional.py # BM25 retrieval
python pipeline/new_query_combined.py    # RRF fusion
```

**Factual benchmark** (run in order, each step checkpoints to `factual_benchmark/results/`)
```bash
python factual_benchmark/select_chunks.py
python factual_benchmark/generate_dataset.py
python factual_benchmark/validate_dataset.py
python factual_benchmark/run_benchmark.py
python factual_benchmark/convert_to_ragchecker.py
python factual_benchmark/run_ragchecker.py
```

Final factual qa benchmark dataset: `factual_benchmark/results/qa_final_dataset.json`

**Model benchmarking** (compares 6 LLMs over the RAG pipeline on both the factual and clinical QA sets)

```bash
python model_benchmarking/run_models_factual.py    # factual set  (qa_final_dataset.json)
python model_benchmarking/run_models_clinical.py   # clinical set (qa_subset_200.json)
python model_benchmarking/generate_plots.py        # aggregate metrics + plots + LaTeX table
```

Models benchmarked: `gpt-5.5`, `gpt-5.4`, `claude-opus-4-7`, `kimi-k2.6`, `deepseek-v4-pro`, `glm-5.1`.

Each runner retrieves context via the vector pipeline (`pipeline/new_query_vector.py`), queries every
model concurrently, and incrementally checkpoints results so interrupted runs can resume:

```
model_benchmarking/datasets/                 — benchmark input datasets (factual + clinical)
model_benchmarking/results/factual/          — per-model answers, metrics.csv/xlsx, RAGChecker output
model_benchmarking/results/clinical/         — same layout for the clinical set
model_benchmarking/plots/                    — efficiency, factual-vs-clinical, and RAG safety plots
```

Per-question metrics captured: latency, input/output/reasoning tokens, cost (USD), and inference
speed (tokens/s). RAGChecker scores (F1, faithfulness, context precision/utilization) are produced
per model under each `results/.../ragchecker/` folder.

---

## Data format

Scraped guidelines are stored as JSONL in `data/`. Each line:

```json
{
  "doc_id": "astma_bij_volwassenen",
  "doc_title": "Astma bij volwassenen",
  "url": "https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen",
  "chunk_id": "astma_bij_volwassenen_0001",
  "section_path": ["Diagnostiek", "Anamnese"],
  "text": "Vraag naar ...",
  "tokens": 245
}
```

---

## Dependencies

Key packages: `selenium`, `beautifulsoup4`, `spacy`, `transformers`, `BERTScore`, `ragchecker`, `qdrant-client`, `openai`, `anthropic`, `google-genai`, `tenacity`, `pandas`, `matplotlib`, `seaborn`. See `requirements.txt` for the full pinned list.

---

## RAGChecker

This project uses [RAGChecker](https://github.com/amazon-science/RAGChecker) for fine-grained RAG evaluation. See the [tutorial](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md) for usage details.

Additional setup required:
```bash
pip install ragchecker
python -m spacy download en_core_web_sm
```
