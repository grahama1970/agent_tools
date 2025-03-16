#!/usr/bin/env python3
"""
main.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- json (Python standard library): https://docs.python.org/3/library/json.html
- loguru: https://loguru.readthedocs.io/
- tqdm: https://tqdm.github.io/

This script integrates the downloading, HTML cleaning, section extraction, and metadata enrichment into one pipeline.
It processes files from a downloaded site directory and produces a JSON output.
A tqdm progress bar is used to indicate progress during file processing.
"""

import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm  # Added for progress indication
from clean_html import clean_html, convert_to_markdown
from extract_sections import extract_sections_from_html

def process_file(file_path: Path) -> dict:
    """
    Process a single HTML file: clean, convert to markdown, extract sections.
    
    Args:
        file_path (Path): Path to the HTML file.
    
    Returns:
        dict: A dictionary representing the file's extracted content and metadata.
    """
    with file_path.open("r", encoding="utf-8") as f:
        raw_html = f.read()
    
    cleaned = clean_html(raw_html)
    # Optionally convert to markdown (here we keep the HTML for section extraction)
    # md_text = convert_to_markdown(cleaned)
    
    sections = extract_sections_from_html(cleaned, file_path)
    
    return {
        "file": str(file_path),
        "sections": sections
    }

def process_directory(root_dir: Path) -> list:
    """
    Recursively process all HTML files in a directory.
    
    Args:
        root_dir (Path): The root directory containing downloaded HTML files.
    
    Returns:
        list: A list of processed file dictionaries.
    """
    results = []
    html_files = list(root_dir.rglob("*.html"))
    
    # Use tqdm to provide progress feedback during file processing
    for html_file in tqdm(html_files, desc="Processing HTML files"):
        logger.info(f"Processing file: {html_file}")
        result = process_file(html_file)
        results.append(result)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <downloaded_site_directory> <output_json_file>")
        sys.exit(1)
    
    site_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    logger.info(f"Starting processing for site directory: {site_dir}")
    processed_data = process_directory(site_dir)
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Output written to: {output_file}")
