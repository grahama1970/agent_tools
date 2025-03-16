#!/usr/bin/env python3
"""
page_downloader.py

Official Documentation:
- wget: https://www.gnu.org/software/wget/manual/wget.html
- requests: https://docs.python-requests.org/
- Playwright: https://playwright.dev/python/docs/intro
- subprocess: https://docs.python.org/3/library/subprocess.html
- pathlib: https://docs.python.org/3/library/pathlib.html

This module provides a function to download a web page.
It first attempts to fetch the page using wget (via subprocess).
If the downloaded content appears to be rendered by JavaScript (e.g. shows a "You need to enable JavaScript" message or a 404 indicator),
it falls back to using Playwright to fetch the fully rendered page.
The fetched page is cached based on the URL hash.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from loguru import logger  # Documentation: https://loguru.readthedocs.io/
import asyncio
from playwright.async_api import async_playwright  # Documentation: https://playwright.dev/python/docs/intro

# Cache directory for downloaded pages
CACHE_DIR = Path(__file__).parent / "cache" / "pages"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_path(url: str) -> Path:
    """Generate a cache file path for a given URL using its hash."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{url_hash}.html"

def is_page_rendered_by_js(content: str) -> bool:
    """
    Heuristically determine if the downloaded content is only a skeleton due to JavaScript rendering.
    
    Returns True if:
      - Common indicators of unrendered pages are present (e.g., "You need to enable JavaScript", "404 Page not found", "<noscript>")
      - OR if the content length is very short (less than 500 characters).
    """
    indicators = [
        "You need to enable JavaScript",
        "404 Page not found",
        "<noscript>",
    ]
    if any(indicator in content for indicator in indicators):
        return True

    if len(content.strip()) < 500:
        return True

    return False
    

def fetch_with_wget(url: str) -> str:
    """
    Use wget to download the page to a temporary cache file and return its content.
    
    Raises:
        subprocess.CalledProcessError if wget fails.
    """
    cache_file = get_cache_path(url)
    # Only download if cache file doesn't exist
    if not cache_file.exists():
        wget_command = [
            "wget",
            "--quiet",
            "--no-clobber",          # Do not overwrite existing files
            "--output-document", str(cache_file),
            url
        ]
        logger.info(f"Running wget command: {' '.join(wget_command)}")
        subprocess.run(wget_command, check=True)
    else:
        logger.info(f"Using cached file: {cache_file}")

    return cache_file.read_text(encoding="utf-8")

async def fetch_with_playwright(url: str) -> str:
    """
    Use Playwright to fetch a fully rendered page.
    """
    logger.info(f"Fetching with Playwright: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        await browser.close()
    return content

async def fetch_page(url: str) -> str:
    """
    Fetch a web page by first trying wget, and if the content appears incomplete,
    fallback to using Playwright.

    Returns:
        The HTML content of the page.
    """
    try:
        content = fetch_with_wget(url)
    except subprocess.CalledProcessError as e:
        logger.error(f"wget failed: {e}; falling back to Playwright")
        content = await fetch_with_playwright(url)

    # Check if the content is insufficient (e.g., indicates JS is required)
    if is_page_rendered_by_js(content):
        logger.warning("Downloaded content appears to be rendered by JavaScript; falling back to Playwright")
        content = await fetch_with_playwright(url)
        # Update the cache with the fully rendered page.
        cache_file = get_cache_path(url)
        cache_file.write_text(content, encoding="utf-8")
    else:
        logger.info("Content downloaded successfully with wget.")

    return content

# -------------------------
# For Testing/Usage
# -------------------------
if __name__ == "__main__":
    import asyncio
    if len(sys.argv) != 2:
        print("Usage: python page_downloader.py <url>")
        sys.exit(1)
    url = sys.argv[1]
    content = asyncio.run(fetch_page(url))
    print(content)
