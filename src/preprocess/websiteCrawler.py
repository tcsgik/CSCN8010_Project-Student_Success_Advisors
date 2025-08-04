import re
from bs4 import BeautifulSoup, Comment, Tag, NavigableString
from urllib.parse import urljoin, urlparse
from collections import deque
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse

excluded_roles = {'navigation', 'banner'}
excluded_classes = {"fas", "fa-fw", "fa-chevron-right", "fa", "fontawesome", "fa-solid"}

def is_excluded(tag):
    # Skip if it's not a Tag (e.g., NavigableString or Comment)
    if not isinstance(tag, Tag):
        return False

    excluded_roles = {"navigation", "search", "complementary", "banner"}

    while tag is not None:
        if isinstance(tag, Tag):
            # Exclude <div> with role
            if tag.get('role') in excluded_roles:
                return True
            # Exclude <li> with class "navbar-right"
            if tag.name == 'li' and 'navbar-right' in tag.get('class', []):
                return True
            if tag.name == 'i':
                return True
            # Exclude known non-content tags and FontAwesome icons
            if tag.name in {'script', 'style', 'noscript', 'i'}:
                return True

        tag = tag.parent

    return False

def extract_leaf_text_blocks(tag: Tag) -> List[str]:
    if tag is None or is_excluded(tag):
        return []

    if isinstance(tag, Comment):
        return []

    if isinstance(tag, NavigableString):
        text = str(tag).strip()
        return [text] if text else []

    if isinstance(tag, Tag) and tag.name == 'i':
        return []  # ✅ completely exclude <i> tags (FontAwesome etc.)

    # Special handling for <a> tags to include text + link
    if tag.name == 'a':
        href = tag.get('href', '').strip()
        inner_blocks = []
        for child in tag.children:
            inner_blocks.extend(extract_leaf_text_blocks(child))
        text = ' '.join(inner_blocks).strip()
        if text:
            return [f"{text} [LINK: {href}]" if href else text]
        return []

    # Leaf node
    if not list(tag.children):
        text = tag.get_text(strip=True)
        return [text] if text else []

    # Traverse children
    blocks = []
    for child in tag.children:
        blocks.extend(extract_leaf_text_blocks(child))
    return blocks

def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter based on punctuation.
    # Splits on period, exclamation, question marks followed by space or end of string.
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

def split_html_by_leaf_text(
    soup: BeautifulSoup,
    url: str,
    contentId: str = "main",
    maxChar: int = 1000,
    overlap_sentences: int = 2  # Number of sentences to overlap
) -> List[Tuple[str, int, str]]:
    main_div = soup.find('div', id=contentId)
    if not main_div:
        return []

    chunks = []
    chunk_index = 0
    current_sentences = []
    current_length = 0

    def flush_chunk():
        nonlocal chunk_index, current_sentences, current_length
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            chunks.append((url, chunk_index, chunk_text))
            chunk_index += 1
            # Keep overlap sentences for next chunk
            current_sentences = current_sentences[-overlap_sentences:]
            current_length = sum(len(s) + 1 for s in current_sentences)  # plus one for spaces

    for child in main_div.find_all(recursive=False):
        if is_excluded(child):
            continue

        text_blocks = extract_leaf_text_blocks(child)
        for block in text_blocks:
            sentences = split_into_sentences(block)

            for sentence in sentences:
                sentence_len = len(sentence) + 1  # plus space

                # If sentence longer than maxChar, truncate it safely
                if sentence_len > maxChar:
                    # flush current chunk first
                    flush_chunk()
                    truncated_sentence = sentence[:maxChar].rstrip()
                    chunks.append((url, chunk_index, truncated_sentence))
                    chunk_index += 1
                    current_sentences = []
                    current_length = 0
                    continue

                if current_length + sentence_len > maxChar:
                    # flush current chunk before adding this sentence
                    flush_chunk()

                current_sentences.append(sentence)
                current_length += sentence_len

    # Flush any remaining sentences
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        chunks.append((url, chunk_index, chunk_text))

    return chunks

def normalize_url(url):
    parsed = urlparse(url)
    # Keep only scheme, netloc, and path
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

def crawl_site(driver, start_url, output_file, contentId="main", max_level=2):
    domain = urlparse(start_url).netloc
    visited = set()
    exclude_patterns = ["dashboard", "settings/", "account", "appointments", "/profile"]
    queue = deque([(start_url, 0)])

    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'chunk_number', 'content'])  # Header

        while queue:
            url, level = queue.popleft()
            normalized = normalize_url(url)
            if normalized in visited or level > max_level or any(pattern in url.lower() for pattern in exclude_patterns) :
                continue
            visited.add(normalized)

            try:
                driver.get(url)
                if "resources/search" in url.lower():
                    time.sleep(10)
                else:
                    time.sleep(1)

                html =  driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                chunks = split_html_by_leaf_text(soup, url, contentId)
                for row in chunks:
                    writer.writerow(row)
                
                print(f"[Level {level}] {url}")

                if level < max_level:
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if (not href or
                            href.startswith('mailto:') or
                            href.startswith('javascript:') or
                            href == '/students/logout'):
                            continue  # Skip unwanted links
                        
                        full_url = urljoin(url, href)
                        if urlparse(full_url).netloc == domain and full_url not in visited:
                            queue.append((full_url, level + 1))

            except Exception as e:
                print(f"❌ Error on {url}: {e}")
                continue

    print(f"\n✅ Done! Output saved to '{output_file}'")

# Setup Chrome
options = Options()
options.add_argument('--log-level=3')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument('--start-maximized')

driver = webdriver.Chrome(options=options)

start_url = "https://successportal.conestogac.on.ca/students/resources/search/?order=Relevance&topicsUseAnd=true&take=428"
driver.get(start_url)

# Wait for user to log in manually
print("⏳ Waiting for manual login...")
time.sleep(40)  # You can adjust the time
crawl_site(driver, start_url, '../../data/conestogac_successportal.csv', max_level=5)
driver.quit()

# crawl_site(driver, 'https://www.conestogac.on.ca', '../../data/conestogac.csv', 'maincontent', 4)
# driver.quit()

print(f"\n✅ Crawl completed.")

