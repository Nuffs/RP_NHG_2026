# RP_NHG_2026 - Benchmarking RAG for NHG-Guidelines

This project constructs and evaluates a RAG-based Q/A system for the NHG-guidelines.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Components](#project-components)
- [Output Files](#output-files)

## Features

## Project Structure

## Setup Instructions

### 1. Prerequisites

- **Python 3.13+** (recommended)
- **Git** (for cloning the repository)
- **Chrome/Chromium** browser (required for Selenium web scraping)
- **Virtual Environment** (highly recommended)

### 2. Clone Repository

```bash
git clone <repository-url>
cd RP_NHG_2026
```

### 3. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage

### Run All Guidelines Scraper

Scrapes all NHG guidelines from the sitemap:

```bash
cd scraping/all
python main.py
```

**Output**: `data/nhg_all_guidelines.jsonl`

### Run 10 Guidelines Scraper

Scrapes a specific subset of 10 NHG guidelines:

```bash
cd scraping/10_guidelines
python main.py
```

**Output**: `data/nhg_subset_guidelines.jsonl`

### Run Specific Guideline Scraper (Example: Astma)

```bash
cd scraping/astma
python main.py
```

**Output**: `data/nhg_astma.jsonl`


## Project Components

### Scrapers

#### `scraping/all/main.py`

#### `scraping/10_guidelines/main.py`

#### `scraping/astma/main.py`

### Benchmark

## Output Files

All scraped data is saved to the `data/` folder as JSONL files.

### JSONL Format

Each line is a JSON object with the following structure:

```json
{
  "doc_id": "acne",
  "doc_title": "Acne",
  "url": "https://richtlijnen.nhg.org/standaarden/acne",
  "chunk_id": "acne_0001",
  "section_path": ["Richtlijnen diagnostiek", "Anamnese"],
  "text": "Vraag naar: duur en lokalisatie; ...",
  "tokens": 245
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `doc_id` | Unique identifier for the guideline (derived from URL slug) |
| `doc_title` | Full title of the guideline |
| `url` | URL of the guideline page |
| `chunk_id` | Unique identifier for this chunk within the document |
| `section_path` | Array of heading hierarchy (h2, h3, etc.) |
| `text` | Cleaned content text for this section |
| `tokens` | Word count of the text |

## Dependencies

The project uses the following main dependencies:

- **beautifulsoup4** (4.14.3) - HTML/XML parsing
- **lxml** (6.1.0) - Fast XML/HTML parser
- **selenium** (4.43.0) - Web browser automation
- **webdriver-manager** (4.0.2) - Automatic ChromeDriver management
- **requests** (2.33.1) - HTTP library
- **spacy** (3.8.14) - NLP library with Dutch support
- **python-dotenv** (1.2.2) - Environment variable management

## Troubleshooting

### ChromeDriver Issues

If Selenium can't find Chrome:
1. Ensure Chrome/Chromium is installed
2. `webdriver-manager` should auto-download the correct ChromeDriver
3. On Linux: Install Chromium: `sudo apt-get install chromium-browser`

## Contributers

## License

## Contact
