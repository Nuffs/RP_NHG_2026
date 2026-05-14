import json
import time
import re
import tiktoken
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

# Input limit is 30.000 so keep it below that
MAX_TOKENS = 15000
TOKEN_ENCODER = None

def get_token_encoder():
    global TOKEN_ENCODER
    if TOKEN_ENCODER is None:
        TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return TOKEN_ENCODER

def count_tokens(text):
    """Count actual tokens using tiktoken (GPT-3.5/4 encoding)"""
    encoder = get_token_encoder()
    return len(encoder.encode(text))

guidelines = [
    {"id": "astma_bij_volwassenen", "url": "https://richtlijnen.nhg.org/standaarden/astma-bij-volwassenen"},
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
        rows = []
        for tr in element.find_all('tr'):
            cells = []
            for cell in tr.find_all(['th', 'td']):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(" | ".join(cells))

        if rows:
            markdown = rows[0] + "\n"
            if element.find('th'):
                markdown += " | ".join(["----"] * len(rows[0].split(" | "))) + "\n"
            for row in rows[1:]:
                markdown += row + "\n"
            return markdown
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
                markdown_content = clean_text(markdown_content)

                if markdown_content:
                    token_count = count_tokens(markdown_content)

                    output.append({
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "url": url,
                        "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                        # "block_id": chunk_counter,
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
        markdown_content = clean_text(markdown_content)

        if markdown_content:
            token_count = count_tokens(markdown_content)

            output.append({
                "doc_id": doc_id,
                "doc_title": doc_title,
                "url": url,
                "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                # "block_id": chunk_counter,
                "section_path": section_path.get_path(),
                "text": markdown_content,
                "tokens": token_count
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

    output_path = "nhg_subset_guidelines.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")

    return all_chunks


if __name__ == "__main__":
    run_scraping()