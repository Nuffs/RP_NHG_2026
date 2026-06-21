# RP_NHG_2026 — RAG & Benchmarking for NHG Guidelines

A thesis research project exploring Retrieval-Augmented Generation (RAG) on Dutch medical guidelines (NHG).
This work is part of the Research Project course 2025/2026 at TU Delft. 

---

## Project structure

```
scraping/           — NHG guideline scrapers (Selenium + BeautifulSoup)
pipeline/           — RAG retrieval pipeline (vector, BM25, combined RRF)
factual_benchmark/  — Factual QA benchmark for the RAG pipeline
clinical_QA/        — Clinical QA benchmark for the RAG pipeline
data/               — Scraped NHG guidelines (JSONL)
ragchecker/         — RAGChecker evaluation framework
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

Key packages: `selenium`, `beautifulsoup4`, `spacy`, `transformers`, `BERTScore`, `ragchecker`, `qdrant-client`, `openai`. See `requirements.txt` for the full pinned list.

---

## RAGChecker

This project uses [RAGChecker](https://github.com/amazon-science/RAGChecker) for fine-grained RAG evaluation. See the [tutorial](https://github.com/amazon-science/RAGChecker/blob/main/tutorial/ragchecker_tutorial_en.md) for usage details.

Additional setup required:
```bash
pip install ragchecker
python -m spacy download en_core_web_sm
```
