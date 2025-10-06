import hashlib
import pickle
import os
from pathlib import Path
from seleniumbase import SB
from bs4 import BeautifulSoup
from typing import Optional

class ScraperCache:
    def __init__(self, cache_dir: str = ".scraper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, url: str) -> Optional[BeautifulSoup]:
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None

    def set(self, url: str, soup: BeautifulSoup) -> None:
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(soup, f)
        except:
            pass

_cache = ScraperCache() # glob

def scrape_url(url: str, use_cache: bool = True) -> BeautifulSoup:
    if use_cache:
        cached_soup = _cache.get(url)
        if cached_soup:
            return cached_soup

    with SB(test=True, uc=True) as sb:
        sb.open(url)
        html = sb.get_page_source()
        soup = BeautifulSoup(html, 'html.parser')

        if use_cache:
            _cache.set(url, soup)

        return soup



if __name__ == "__main__":
    url = "https://httpbin.org/html"
    print("Testing scraper...")
    soup = scrape_url(url)
    print(f"Title: {soup.title.text if soup.title else 'No title'}")
    print(f"Found {len(soup.find_all('p'))} paragraphs")
    print("\nTesting cache...")
    soup2 = scrape_url(url)
    print("Cache test completed")
