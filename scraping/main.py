import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import time
import re

SITEMAP_URL = "https://richtlijnen.nhg.org/sitemap.xml"

print("Fetching sitemap...")
xml = requests.get(SITEMAP_URL).text
soup = BeautifulSoup(xml, "xml")

guideline_links = sorted({
    loc.text.strip()
    for loc in soup.find_all("loc")
    if "/standaarden/" in loc.text
})

print(f"Found {len(guideline_links)} guidelines.")

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver, 20)


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()


JUNK_PHRASES = [
    "Om deze website optimaal te laten functioneren",
    "Cookies die noodzakelijk zijn",
    "Bekijk alle resultaten",
    "Open submenu",
    "Alleen hoofdtekst",
    "Printen",
    "Volledig",
    "Home Handleiding NHG-Standaarden",
]


def remove_junk(text: str) -> str:
    for phrase in JUNK_PHRASES:
        text = text.replace(phrase, "")
    return clean_text(text)


def chunk_text(text, max_chars=1200):
    words = text.split()
    chunks, current = [], []

    for w in words:
        current.append(w)
        if len(" ".join(current)) > max_chars:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def extract_sections(soup: BeautifulSoup):
    container = soup.select_one("main, article, #main-content, body") or soup
    headings = container.find_all(["h2", "h3"])

    sections = {}

    for i, h in enumerate(headings):
        title = clean_text(h.get_text())
        if not title:
            continue

        texts = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) in ["h2", "h3"]:
                break
            if getattr(sib, "name", None) in ["p", "li", "ul", "ol", "div", "section"]:
                txt = clean_text(sib.get_text(" ", strip=True))
                if txt:
                    texts.append(txt)

        section_text = remove_junk(" ".join(texts))
        if section_text:
            sections[title] = chunk_text(section_text)

    return sections


def scrape_guideline(url):
    print(f"Scraping: {url}")
    driver.get(url)
    time.sleep(2)

    # Try iframe first
    iframe_found = False
    try:
        iframe = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "iframe[src*='nhgdoc'], iframe[src*='richtlijnen']")
            )
        )
        driver.switch_to.frame(iframe)
        iframe_found = True
    except:
        pass

    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Title
    title_el = soup.select_one("h1") or soup.select_one("title")
    title = clean_text(title_el.get_text()) if title_el else "Untitled"

    # Extract structured sections
    sections = extract_sections(soup)

    # Fallback: whole text
    if not sections:
        blocks = soup.select("p, li")
        text = remove_junk(" ".join(clean_text(b.get_text()) for b in blocks))
        sections = {"FULL_TEXT": chunk_text(text)}

    if iframe_found:
        driver.switch_to.default_content()

    return {
        "url": url,
        "title": title,
        "sections": sections
    }


all_guidelines = []

for i, link in enumerate(guideline_links, start=1):
    print(f"[{i}/{len(guideline_links)}]")
    try:
        data = scrape_guideline(link)
        all_guidelines.append(data)
    except Exception as e:
        print("Error scraping", link, e)

driver.quit()

with open("nhg_guidelines.json", "w", encoding="utf-8") as f:
    json.dump(all_guidelines, f, ensure_ascii=False, indent=2)

print("Done! Saved to nhg_guidelines.json")
