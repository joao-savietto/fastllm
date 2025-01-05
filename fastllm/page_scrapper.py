from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
from collections import Counter
from fastllm.vectordb import VectorDB
from fastllm.decorators import run_in_thread
import re
from typing import List


def longest_repeated_substring(s: str) -> str:
    if not s:
        return ""

    def is_repeated_substring(length: int) -> str:
        substrings = set()
        for i in range(len(s) - length + 1):
            substr = s[i: i + length]
            if substr in substrings:
                return substr
            substrings.add(substr)
        return ""

    low, high = 0, len(s)
    result = ""

    while low <= high:
        mid = (low + high) // 2
        repeated_substr = is_repeated_substring(mid)
        if repeated_substr:
            result = repeated_substr
            low = mid + 1
        else:
            high = mid - 1

    return result


def is_valid_url(url: str):
    repeated = longest_repeated_substring(url)
    if repeated:
        if url.count(repeated) > 1 and len(repeated) > 4:
            return False
    if (
        url.count("www.") > 1
        or url.count("//") > 1
        or url.count("http") > 1
        or url.count(".html") > 1
    ):
        return False
    if len(url) > 1000:
        return False
    return (
        re.match(r"https?://(?:[a-z0-9-]+\.)*[a-z0-9-]+\.[a-z]+(?:/.*)?", url)
        is not None
    )


def find_urls(text: str) -> List[str]:
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    urls = re.findall(pattern, text)
    return urls if urls else []


class PageScraper:
    def __init__(
        self,
        base_url: str, page_name: str,
        vector_db: VectorDB
    ):
        self.base_url = base_url
        self.page_name = page_name
        self.visited = set()
        self.links = set()
        self.texts = []
        self.sources = []
        self.vector_db = vector_db

    def _scrap(self, url_base=None):
        url = url_base or self.base_url
        if not url.endswith("/"):
            url = f"{url}/"
        if (
            url in self.visited
            or f"{url}#" in self.visited
            or ("/." in url and "./" in url)
            or "#" in url
        ):
            return
        time.sleep(0.2)
        try:
            html = urlopen(url).read()
        except Exception:
            print(f"Failed: {url}")
            self.visited.add(url)
            return
        soup = BeautifulSoup(html, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        self.visited.add(url)
        # find all <a> tags
        new_links = set()
        for link in soup.find_all("a"):
            href = link.get("href")
            if href:
                if href.startswith("http") and (
                    not href.startswith(url) or
                        href == url or href == f"{url}#"
                ):
                    continue
                elif href.startswith("/"):
                    href = f"{url}{href.removeprefix('/')}"
                else:
                    if href.startswith("http"):
                        continue
                    href = f"{url}{href}" if\
                        not href.startswith("www.") else href
                if href not in self.visited and is_valid_url(href):
                    new_links.add(href)
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)
        self.texts.append(text)
        self.sources.append(url)
        # scrape all links
        for link in new_links:
            self._scrap(link)

    def _clean_text(self):
        texts = self.texts
        all_text = "\n".join(texts)
        lines = [x for x in all_text.split("\n") if len(x) >= 2]
        counts = Counter(lines)
        most_common = [k for k, v in counts.items() if v >= len(texts)]
        clean_pages = []
        for text in texts:
            text = "\n".join([x for x in text.split("\n") if x not in most_common])
            text = text.strip().strip("\n")
            clean_pages.append(text)
        return clean_pages

    @run_in_thread
    def run(self):
        self._scrap()
        self.texts = self._clean_text()
        for text, source in zip(self.texts, self.sources):
            chunks = [text[i: i + 300] for i in range(0, len(text), 300)]
            self.vector_db.insert(
                self.page_name, chunks, metadatas=[{"source": source} for _ in chunks]
            )
