#!/usr/bin/env python3
"""
test_single_page_download.py

Official Documentation:
- pytest: https://docs.pytest.org/
- pathlib: https://docs.python.org/3/library/pathlib.html
- subprocess: https://docs.python.org/3/library/subprocess.html

This test verifies that when the single-page download mode is enabled,
only the specified page (and its assets) are downloaded, and no additional pages are fetched.
"""

import subprocess
from pathlib import Path
import pytest

from agent_tools.fetch_docs.download_site import download_site

# Use a known static page for testing, for example "https://example.com"
TEST_URL = "https://example.com"
# Use a temporary directory for testing purposes.
TEST_OUTPUT_DIR = Path("tests/temp_single_page_download")

@pytest.fixture(autouse=True)
def cleanup_temp_dir():
    """Ensure the temporary directory is removed before and after tests."""
    if TEST_OUTPUT_DIR.exists():
        for child in TEST_OUTPUT_DIR.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                for sub in child.glob("**/*"):
                    if sub.is_file():
                        sub.unlink()
                child.rmdir()
    yield
    if TEST_OUTPUT_DIR.exists():
        for child in TEST_OUTPUT_DIR.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                for sub in child.glob("**/*"):
                    if sub.is_file():
                        sub.unlink()
                child.rmdir()
        TEST_OUTPUT_DIR.rmdir()

def test_single_page_download():
    """
    Test that when single-page mode is used, only the target page and its immediate assets are downloaded.
    
    This test calls download_site with recursive=False and then checks that the output directory contains a limited number
    of HTML files (ideally, only one main page file). Adjust the expected count based on the actual assets downloaded.
    """
    # Download only the single page using our function.
    download_site(TEST_URL, str(TEST_OUTPUT_DIR), recursive=False)
    
    # Find all .html files in the output directory.
    html_files = list(TEST_OUTPUT_DIR.rglob("*.html"))
    # For a single-page download, we expect at least one file (the main page).
    # Depending on the page, there might be a few asset files; however, we expect that the number
    # of main pages (i.e., HTML files in subdirectories corresponding to additional pages) is one.
    # Here, we assume that if more than 3 HTML files are present, recursion might have occurred.
    assert len(html_files) <= 3, f"Expected a single page download, found {len(html_files)} HTML files."
    
    # Optionally, check that the main downloaded file corresponds to the test URL (e.g., "index.html" for example.com)
    main_file = None
    for file in html_files:
        if file.name.lower() in ["index.html", "example.html"]:
            main_file = file
            break
    assert main_file is not None, "Main page file not found in single-page download."
    