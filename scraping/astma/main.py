import json
import time
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

URL = "https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen"


def chunk_text(text, max_tokens=300):
    words = text.split()
    chunks, current = [], []

    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def scrape_nhg():
    # --- Selenium setup ---
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get(URL)
    time.sleep(3)

    # Expand collapsible sections
    for b in driver.find_elements("css selector", "button"):
        try:
            if "Open submenu" in b.text:
                b.click()
                time.sleep(0.2)
        except:
            pass

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    doc_title = soup.find("h1").get_text(strip=True)
    doc_id = "asthma_volwassenen"

    # All headings in order
    headings = soup.find_all(["h2", "h3", "h4"], class_="section-heading")
    if not headings:
        raise ValueError("No section headings found")

    output = []
    chunk_counter = 1

    for i, heading_tag in enumerate(headings):
        # Clean heading text (remove copy button)
        for btn in heading_tag.find_all("button", class_="btn-copy-anchor-link"):
            btn.decompose()
        heading_text = heading_tag.get_text(strip=True)

        # Collect all content blocks between this heading and the next heading
        content_blocks = []
        for el in heading_tag.next_elements:
            if not isinstance(el, Tag):
                continue

            # Stop when we hit the next heading
            if el.name in ["h2", "h3", "h4"] and "section-heading" in (el.get("class") or []):
                break

            # Real content lives in div.field--name-text
            classes = el.get("class") or []
            if "field--name-text" in classes:
                content_blocks.append(el)

        if not content_blocks:
            continue

        # Merge all text from these content blocks
        merged_text = ""
        for block in content_blocks:
            # Inside each block, take p/ul/ol/table
            inner_blocks = block.find_all(["p", "ul", "ol", "table"], recursive=True)
            if not inner_blocks:
                inner_blocks = [block]

            for ib in inner_blocks:
                # Replace links with "text (URL)"
                for a in ib.find_all("a"):
                    href = a.get("href")
                    if href:
                        a.replace_with(f"{a.get_text(strip=True)} ({href})")

                text = ib.get_text(" ", strip=True)
                if text:
                    merged_text += " " + text

        merged_text = merged_text.strip()
        if not merged_text:
            continue

        # Chunk per heading
        for chunk in chunk_text(merged_text):
            output.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "url": URL,
                "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                "section_path": [heading_text],
                "text": chunk,
                "tokens": len(chunk.split())
            })
            chunk_counter += 1

    return output


if __name__ == "__main__":
    data = scrape_nhg()

    print("\n--- Preview of scraped chunks ---")
    for item in data[:10]:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    with open("nhg_astma.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(data)} chunks to nhg_astma.jsonl")
