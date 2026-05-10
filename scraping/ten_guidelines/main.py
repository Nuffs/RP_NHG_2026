import json
import time
import re
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

MAX_TOKENS = 300

guidelines = [
    {"id": "astma_bij_volwassenen", "url": "https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen"},
    {"id": "diabetes", "url": "https://richtlijnen.nhg.org/standaarden/diabetes-mellitus-type-2"},
    {"id": "dementie", "url": "https://richtlijnen.nhg.org/standaarden/dementie"},
    {"id": "depressie", "url": "https://richtlijnen.nhg.org/standaarden/depressie"},
    {"id": "COPD", "url": "https://richtlijnen.nhg.org/standaarden/copd"},
    {"id": "angst", "url": "https://richtlijnen.nhg.org/standaarden/angst"},
    {"id": "chronische nierschade", "url": "https://richtlijnen.nhg.org/standaarden/chronische-nierschade"},
    {"id": "influenza", "url": "https://richtlijnen.nhg.org/behandelrichtlijnen/influenza"},
    {"id": "hartfalen", "url": "https://richtlijnen.nhg.org/standaarden/hartfalen"},
    {"id": "hand-en-polsklachten", "url": "https://richtlijnen.nhg.org/standaarden/hand-en-polsklachten"},
]

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
    text = re.sub(r'\d+\s*\(#literature-ref-\d+\)', '', text)
    text = re.sub(r'Tips voor gebruik:.*?scrollen\.', '', text, flags=re.DOTALL)
    text = deduplicate_text(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def deduplicate_text(text):
    sentences = text.split(". ")
    seen = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            seen.append(s_clean)
    return ". ".join(seen)


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
        button = wait.until(EC.presence_of_element_located((By.ID, "volledige-tekst-trigger")))
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


    h2_headings = main_content.find_all("h2")
    desired_h2 = [h for h in h2_headings 
            if "diagnostiek" in clean_heading(h.get_text()).lower()
            or "beleid" in clean_heading(h.get_text()).lower()
            or "spoed" in clean_heading(h.get_text()).lower()
            or "acuut" in clean_heading(h.get_text()).lower()
            or "acute" in clean_heading(h.get_text()).lower()]
    
    if not desired_h2:
        print(f"  WARNING: no 'diagnostiek' or 'beleid' h2 headings found for {url}")
        return []
        
    tag_list = list(main_content.find_all(["h2", "h3"]))
    
    output = []
    chunk_counter = 1

    for h2_tag in desired_h2:
        h2_pos = tag_list.index(h2_tag)

        h2_end_pos = len(tag_list)
        for i in range(h2_pos + 1, len(tag_list)):
            if tag_list[i].name == "h2":
                h2_end_pos = i
                break
        h3_tags = [t for t in tag_list[h2_pos + 1:h2_end_pos] if t.name == "h3"]

        if not h3_tags:
            content_blocks = []
            for el in h2_tag.find_all_next(["h2", "p", "ul", "ol", "table"]):
                if el.name == "h2":
                    break
            content_blocks.append(el)

            if not content_blocks:
                continue

            merged_text = ""
            for block in content_blocks:
                text = block.get_text(" ", strip=True)
                if text:
                    merged_text += " " + text

            merged_text = clean_text(merged_text.strip())
            h2_text = clean_heading(h2_tag.get_text(strip=True))

            if merged_text:
                output.append({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "url": url,
                    "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                    "section_path": [h2_text],
                    "text": merged_text,
                    "tokens": len(merged_text.split())
                })
                chunk_counter += 1
            continue

        for heading_tag in h3_tags:
            for btn in heading_tag.find_all("button"):
                btn.decompose()
            for btn in h2_tag.find_all("button"):
                btn.decompose()
            heading_text = clean_heading(heading_tag.get_text(strip=True))

            content_blocks = []
            for el in heading_tag.find_all_next(["h2", "h3", "p", "ul", "ol", "table", "h4"]):
                if el.name in ["h2", "h3"]:
                    break
                content_blocks.append(el)

            if not content_blocks:
                continue

            merged_text = ""
            for block in content_blocks:
                for a in block.find_all("a"):
                    href = a.get("href")
                    if href:
                        a.replace_with(f"{a.get_text(strip=True)} ({href})")
                text = block.get_text(" ", strip=True)
                if text:
                    merged_text += " " + text
            
            merged_text = merged_text.strip()
            merged_text = clean_heading(merged_text)
            merged_text = clean_text(merged_text)

            if not merged_text:
                continue

            h2_text = clean_heading(h2_tag.get_text(strip=True))

            output.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "url": url,
                "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                "section_path": [h2_text, heading_text],
                "text": merged_text,
                "tokens": len(merged_text.split())
            })
            chunk_counter += 1

    return output


def run_scraping():
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

    print(f"\nTotal chunks: {len(all_chunks)}")

    output_path = "data/nhg_subset_guidelines.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    return all_chunks


if __name__ == "__main__":
    run_scraping