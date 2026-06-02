import json
import os
import time
import re
import tiktoken
import uuid
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, HnswConfigDiff, OptimizersConfigDiff, PointStruct, VectorParams

DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Input limit is 30.000 so keep it below that
MAX_TOKENS = 15000
TOKEN_ENCODER = None

def get_token_encoder():
    global TOKEN_ENCODER
    if TOKEN_ENCODER is None:
        TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return TOKEN_ENCODER


def count_tokens(text):
    encoder = get_token_encoder()
    return len(encoder.encode(text))


def to_point_uuid(raw_id: Any, namespace: str) -> str:
    if raw_id is None:
        return str(uuid.uuid4())

    raw = str(raw_id)
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{raw}"))


guidelines = [
    {"id": "astma_bij_volwassenen", "url": "https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen"},
    # -------------- Above done, below still to do ---------------
    # {"id": "diabetes", "url": "https://richtlijnen.nhg.org/standaarden/diabetes-mellitus-type-2"},
    # {"id": "dementie", "url": "https://richtlijnen.nhg.org/standaarden/dementie"},
    # {"id": "depressie", "url": "https://richtlijnen.nhg.org/standaarden/depressie"},
    # {"id": "COPD", "url": "https://richtlijnen.nhg.org/standaarden/copd"},
    # {"id": "angst", "url": "https://richtlijnen.nhg.org/standaarden/angst"},
    # {"id": "chronische nierschade", "url": "https://richtlijnen.nhg.org/standaarden/chronische-nierschade"},
    # {"id": "influenza", "url": "https://richtlijnen.nhg.org/behandelrichtlijnen/influenza"},
    # {"id": "hartfalen", "url": "https://richtlijnen.nhg.org/standaarden/hartfalen"},
    # {"id": "hand-en-polsklachten", "url": "https://richtlijnen.nhg.org/standaarden/hand-en-polsklachten"},
]


class SectionPath:
    def __init__(self):
        self.path = {}  # {level: heading_text}

    def push(self, level, text):
        # Remove all levels deeper than current
        levels_to_remove = [k for k in self.path.keys() if k > level]
        for k in levels_to_remove:
            del self.path[k]
        self.path[level] = text

    def get_path(self):
        return [self.path[k] for k in sorted(self.path.keys())]

    def reset(self):
        self.path = {}


def get_heading_level(tag_name):
    if tag_name.startswith('h') and len(tag_name) == 2 and tag_name[1].isdigit():
        return int(tag_name[1])
    return None


def convert_element_to_markdown(element):
    if isinstance(element, str):
        text = element.strip()
        return text if text else ""

    tag_name = element.name
    if not tag_name:
        return ""

    # Handle paragraphs
    if tag_name == 'p':
        text = element.get_text(strip=True)
        return f"{text}\n" if text else ""

    # Handle emphasis and strong
    if tag_name in ['em', 'i']:
        text = element.get_text(strip=True)
        return f"*{text}*" if text else ""
    if tag_name in ['strong', 'b']:
        text = element.get_text(strip=True)
        return f"**{text}**" if text else ""

    # Handle links
    if tag_name == 'a':
        text = element.get_text(strip=True)
        href = element.get('href', '')
        return f"[{text}]({href})" if text else ""

    # Handle images
    if tag_name == 'img':
        alt_text = element.get('alt', 'Image')
        return f"\n[IMAGE PLACEHOLDER: {alt_text}]\n"

    # Handle unordered lists
    if tag_name == 'ul':
        items = []
        for li in element.find_all('li'):
            li_text = li.get_text(strip=True)
            if li_text:
                items.append(f"- {li_text}")
        return "\n".join(items) + "\n" if items else ""

    # Handle ordered lists
    if tag_name == 'ol':
        items = []
        for i, li in enumerate(element.find_all('li'), 1):
            li_text = li.get_text(strip=True)
            if li_text:
                items.append(f"{i}. {li_text}")
        return "\n".join(items) + "\n" if items else ""

    # Handle tables - preserve as text representation
    if tag_name == 'table':
        trs = element.find_all('tr')
        if not trs:
            return ""

        max_rows = len(trs)
        max_cols = 0
        grid = [[None for _ in range(100)] for _ in range(max_rows)]
        filled = [[False for _ in range(100)] for _ in range(max_rows)]

        for row_idx, tr in enumerate(trs):
            cells = tr.find_all(['th', 'td'])
            col_idx = 0

            for cell in cells:
                while col_idx < 100 and filled[row_idx][col_idx]:
                    col_idx += 1

                cell_text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))

                for r in range(row_idx, min(row_idx + rowspan, max_rows)):
                    for c in range(col_idx, col_idx + colspan):
                        grid[r][c] = cell_text
                        filled[r][c] = True

                col_idx += colspan
                max_cols = max(max_cols, col_idx)

        grid = [row[:max_cols] for row in grid]
        if grid:
            rows_md = []
            for row in grid:
                row_cells = [str(cell) if cell is not None else "" for cell in row]
                rows_md.append(' | '.join(row_cells))

            md = rows_md[0] + "\n"
            md += ' | '.join(['----'] * max_cols) + "\n"
            for row in rows_md[1:]:
                md += row + "\n"
            return md + "\n"
        return ""

    # Handle headings (pre-blocks)
    if tag_name == 'pre':
        text = element.get_text()
        return f"```\n{text}\n```\n"

    # Handle blockquotes
    if tag_name == 'blockquote':
        text = element.get_text(strip=True)
        if text:
            return "\n".join([f"> {line}" for line in text.split("\n")]) + "\n"
        return ""

    # Default: get text content
    text = element.get_text(strip=True)
    return text + "\n" if text else ""


def build_markdown_content(elements):
    markdown = ""
    for element in elements:
        markdown += convert_element_to_markdown(element)

    # Clean up multiple consecutive newlines
    markdown = re.sub(r'\n\n\n+', '\n\n', markdown).strip()
    return markdown


def clean_heading(text):
    text = re.sub(r"Naar\s+[Ss]amenvatting", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Naar\s+[Vv]olledige\s+tekst", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[Vv]olledige\s+tekst", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[Ss]amenvatting", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Kopieer\s+ankerlink", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Kopieer\s+anker", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNaar\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(text):
    # Remove literature references
    text = re.sub(r'\d+\s*\(#literature-ref-\d+\)', '', text)
    # Remove tips sections
    text = re.sub(r'Tips voor gebruik:.*?scrollen\.', '', text, flags=re.DOTALL)
    # Preserve markdown formatting while normalizing whitespace
    # Only collapse multiple spaces (not newlines which are important for markdown)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Clean up multiple spaces in each line but preserve the line
        line = re.sub(r' +', ' ', line).strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    return text


def setup_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    return driver


def expand_collapsibles(driver):
    buttons = driver.find_elements("css selector", "button")
    for b in buttons:
        try:
            if "Open submenu" in b.text:
                b.click()
                time.sleep(0.2)
        except Exception:
            pass


def scrape_single_guideline(driver, url, doc_id):
    driver.get(url)

    wait = WebDriverWait(driver, 15)

    try:
        wait.until(lambda d: d.find_element(By.TAG_NAME, "h1").text.strip() != ""
                             and "beveiligingscontrole" not in d.find_element(By.TAG_NAME, "h1").text.lower())
        doc_title = driver.find_element(By.TAG_NAME, "h1").text.strip()
    except:
        doc_title = doc_id

    try:
        button = wait.until(ec.presence_of_element_located((By.ID, "volledige-tekst-trigger")))
        driver.execute_script("arguments[0].click();", button)
        print("  Clicked 'Volledige tekst' tab.")
    except Exception as e:
        print(f"  Could not click tab: {e}")

    try:
        wait.until(lambda d: (
            lambda el: el is not None and el.is_displayed()
        )(d.find_element(By.ID, "volledige-tekst")))
        print("  Panel #volledige-tekst is visible.")
    except Exception as e:
        print(f"  WARNING: panel did not become visible: {e}")

    time.sleep(2)

    expand_collapsibles(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    main_content = soup.find(id="volledige-tekst")
    if not main_content:
        print(f"  WARNING: #volledige-tekst not found in parsed HTML for {url}")
        main_content = soup.find("div", class_="content--main") or soup

    # Process all content dynamically
    output = []
    chunk_counter = 1

    # Get all elements (headings and content) in document order
    all_elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'table', 'pre', 'blockquote'])
    print(f"  Found {len(all_elements)} content elements")

    section_path = SectionPath()
    current_content_blocks = []

    for element in all_elements:
        tag_name = element.name

        # Check if this is a heading
        heading_level = get_heading_level(tag_name)

        if heading_level:
            # If we have accumulated content, process it
            if current_content_blocks and section_path.get_path():
                markdown_content = build_markdown_content(current_content_blocks)
                # markdown_content = clean_text(markdown_content)

                if markdown_content:
                    token_count = count_tokens(markdown_content)

                    output.append({
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "url": url,
                        "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                        "section_path": section_path.get_path(),
                        "text": markdown_content,
                        "tokens": token_count
                    })
                    chunk_counter += 1

                current_content_blocks = []

            # Update section path with new heading
            heading_text = clean_heading(element.get_text(strip=True))
            if heading_text:  # Only update if heading has text
                section_path.push(heading_level, heading_text)

        else:
            # Accumulate content blocks
            current_content_blocks.append(element)

    # Process any remaining content
    if current_content_blocks and section_path.get_path():
        markdown_content = build_markdown_content(current_content_blocks)
        # markdown_content = clean_text(markdown_content)

        if markdown_content:
            token_count = count_tokens(markdown_content)

            output.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "url": url,
                "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                "section_path": section_path.get_path(),
                "text": markdown_content,
                "tokens": token_count
            })
            chunk_counter += 1

    return output


def upsert_context_blocks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    blocks: list[dict[str, Any]],
) -> int:
    points: list[PointStruct] = []

    for block in blocks:
        context_id = str(block.get("chunk_id") or block.get("id") or uuid.uuid4())
        point_id = to_point_uuid(context_id, namespace="context")

        # Context data lives in payload; a tiny placeholder vector keeps the collection valid.
        points.append(
            PointStruct(
                id=point_id,
                vector=[0.0],
                payload={
                    "context_id": context_id,
                    "doc_id": block.get("doc_id"),
                    "doc_title": block.get("doc_title"),
                    "url": block.get("url"),
                    "chunk_id": block.get("chunk_id"),
                    "section_path": block.get("section_path"),
                    "text": block.get("text"),
                    "tokens": block.get("tokens"),
                },
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)

    return len(points)


def ensure_context_collection(client: QdrantClient, collection_name: str, recreate: bool = False) -> None:
    existing = {collection.name for collection in client.get_collections().collections}
    if recreate and collection_name in existing:
        client.delete_collection(collection_name=collection_name)

    if collection_name in existing and not recreate:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE, on_disk=True),
        hnsw_config=HnswConfigDiff(on_disk=True),
        optimizers_config=OptimizersConfigDiff(memmap_threshold=0),
        on_disk_payload=True,
    )


def run_scraping_and_ingest(
    qdrant_url: str = DEFAULT_QDRANT_URL,
    context_collection: str = "context_blocks",
    save_jsonl: bool = True,
    jsonl_path: str | Path = Path(__file__).resolve().parent / "nhg_subset_guidelines.jsonl",
):
    driver = setup_driver()
    all_chunks = []

    try:
        for i, g in enumerate(guidelines, start=1):
            print(f"[{i}/{len(guidelines)}]")
            try:
                chunks = scrape_single_guideline(driver, g["url"], g["id"])
                all_chunks.extend(chunks)
                print(f"  -> {len(chunks)} chunks")
            except Exception as e:
                print(f"  ERROR scraping {g['url']}: {e}")
    finally:
        driver.quit()

    json.dump(all_chunks, open('test.json', "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")


    # Initialize Qdrant client and ingest into the server-backed dashboard storage.
    client = QdrantClient(url=qdrant_url)
    ensure_context_collection(client, context_collection)
    print(f"Using collection: {context_collection}")

    # Upsert context blocks into the database.
    inserted = upsert_context_blocks_to_qdrant(client, context_collection, all_chunks)
    print(f"Ingested {inserted} context blocks into '{context_collection}'.")

    # Optionally save to JSONL for backup/reference.
    if save_jsonl:
        jsonl_path = Path(jsonl_path)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in all_chunks:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved backup to {jsonl_path}")

    return all_chunks


def main():
    run_scraping_and_ingest(
        qdrant_url=DEFAULT_QDRANT_URL,
        context_collection="context_blocks",
        save_jsonl=True,
        jsonl_path="nhg_subset_guidelines.jsonl",
    )


if __name__ == "__main__":
    main()

