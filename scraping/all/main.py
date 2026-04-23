import json
import time
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

SITEMAP_URL = "https://richtlijnen.nhg.org/sitemap.xml"
MAX_TOKENS = 300  # ~words


def chunk_text(text, max_tokens=MAX_TOKENS):
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


def get_doc_id_from_url(url: str) -> str:
    """
    Turn https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen
    into 'astma_bij_volwassenen'
    """
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1]
    slug = slug.replace("-", "_")
    return slug


def fetch_guideline_urls():
    print("Fetching sitemap...")
    xml = requests.get(SITEMAP_URL).text
    soup = BeautifulSoup(xml, "xml")

    guideline_links = sorted({
        loc.text.strip()
        for loc in soup.find_all("loc")
        if "/standaarden/" in loc.text
    })

    print(f"Found {len(guideline_links)} guideline URLs.")
    return guideline_links


def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    return driver


def expand_collapsibles(driver):
    # Open all "Open submenu" buttons so full text is in DOM
    buttons = driver.find_elements("css selector", "button")
    for b in buttons:
        try:
            if "Open submenu" in b.text:
                b.click()
                time.sleep(0.2)
        except Exception:
            pass


def scrape_single_guideline(driver, url):
    print(f"Scraping: {url}")
    driver.get(url)
    time.sleep(3)

    expand_collapsibles(driver)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Title
    h1 = soup.find("h1")
    doc_title = h1.get_text(strip=True) if h1 else "Untitled"
    doc_id = get_doc_id_from_url(url)

    # All headings in order
    headings = soup.find_all(["h2", "h3", "h4"], class_="section-heading")
    if not headings:
        print(f"  WARNING: no section headings found for {url}")
        return []

    output = []
    chunk_counter = 1

    for heading_tag in headings:
        # Clean heading text (remove copy button)
        for btn in heading_tag.find_all("button", class_="btn-copy-anchor-link"):
            btn.decompose()
        heading_text = heading_tag.get_text(strip=True)

        # Collect all content blocks between this heading and the next heading
        content_blocks = []
        for el in heading_tag.next_elements:
            if not isinstance(el, Tag):
                continue

            # Stop when we hit the next section heading
            if el.name in ["h2", "h3", "h4"] and "section-heading" in (el.get("class") or []):
                break

            # Real guideline text is inside div.field--name-text
            classes = el.get("class") or []
            if "field--name-text" in classes:
                content_blocks.append(el)

        if not content_blocks:
            continue

        merged_text = ""
        for block in content_blocks:
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
                "url": url,
                "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                "section_path": [heading_text],
                "text": chunk,
                "tokens": len(chunk.split())
            })
            chunk_counter += 1

    return output


def main():
    guideline_urls = fetch_guideline_urls()
    driver = setup_driver()

    all_chunks = []

    try:
        for i, url in enumerate(guideline_urls, start=1):
            print(f"[{i}/{len(guideline_urls)}]")
            try:
                chunks = scrape_single_guideline(driver, url)
                all_chunks.extend(chunks)
                print(f"  -> {len(chunks)} chunks")
            except Exception as e:
                print(f"  ERROR scraping {url}: {e}")
    finally:
        driver.quit()

    print(f"\nTotal chunks: {len(all_chunks)}")

    with open("nhg_all_guidelines.jsonl", "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved to nhg_all_guidelines.jsonl")


if __name__ == "__main__":
    main()
