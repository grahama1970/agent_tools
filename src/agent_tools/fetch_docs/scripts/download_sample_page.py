#!/usr/bin/env python3
"""
download_dynamic_sample_data.py

Official Documentation:
- Playwright: https://playwright.dev/python/docs/intro
- asyncio: https://docs.python.org/3/library/asyncio.html
- pathlib: https://docs.python.org/3/library/pathlib.html

This script downloads the fully rendered HTML of the ArangoDB AQL syntax page
using Playwright (via our fetch_page function) and saves it as "syntax.html" in the sample data folder:
  tests/fetch_docs/sample_data/arangodb_aql/syntax.html

Usage:
    python scripts/download_dynamic_sample_data.py
"""

import asyncio
from pathlib import Path
from agent_tools.fetch_docs.page_downloader import fetch_page

async def main():
    # URL for the ArangoDB AQL syntax page.
    url = "https://docs.arangodb.com/stable/aql/fundamentals/syntax/"
    
    # Define the output directory for sample data in the tests folder.
    # This places the file in: tests/fetch_docs/sample_data/arangodb_aql/
    output_dir = Path(__file__).resolve().parent.parent / "tests" / "fetch_docs" / "sample_data" / "arangodb_aql"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the output file path.
    output_file = output_dir / "syntax.html"
    
    # Fetch the page (using wget first, falling back to Playwright if necessary).
    content = await fetch_page(url)
    
    # Save the fully rendered HTML content to the output file.
    output_file.write_text(content, encoding="utf-8")
    print(f"Downloaded page saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
