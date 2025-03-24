#!/usr/bin/env python3
"""
link_detector.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- re (Python standard library): https://docs.python.org/3/library/re.html
- loguru: https://github.com/Delgan/loguru

This module provides functionality for detecting documentation links in a repository.
It scans markdown files for URLs that match documentation site patterns.

Input/Output Specifications:

detect_documentation_links(repo_path: Path) -> List[str]:
    Input:
        - repo_path: Path to repository to scan
    Output:
        - List of documentation URLs found in the repository

Example usage:
    from agent_tools.fetch_docs.link_detector import detect_documentation_links
    from pathlib import Path
    
    # Detect documentation links in a repository
    repo_path = Path("/path/to/repo")
    links = detect_documentation_links(repo_path)
    
    print(f"Found {len(links)} documentation links:")
    for link in links:
        print(f"  - {link}")
"""

import re
from pathlib import Path
from typing import List, Set
import logging

# Configure logging
logger = logging.getLogger("fetch_docs.link_detector")

# Regular expressions for detecting documentation links
DOC_PATTERNS = [
    # Read the Docs
    r'https?://[a-zA-Z0-9-]+\.readthedocs\.io/[^\s)"\']+',  # Standard RTD domain
    r'https?://readthedocs\.org/projects/[a-zA-Z0-9-]+[^\s)"\']*',  # Project pages
    
    # ArangoDB Documentation
    r'https?://docs\.arangodb\.com/[^\s)"\']+',  # ArangoDB docs
    
    # Generic documentation patterns (can be expanded)
    r'https?://docs\.[a-zA-Z0-9-]+\.[a-zA-Z]+/[^\s)"\']+',  # Generic docs.* pattern
]

def detect_documentation_links(repo_path: Path) -> List[str]:
    """
    Scan repository for documentation links in markdown and other text files.
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        List of detected documentation URLs (deduplicated)
    """
    doc_links = set()
    
    # Find all markdown files
    md_files = list(repo_path.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {repo_path}")
    
    # Process each markdown file
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Search for documentation links using regex patterns
            for pattern in DOC_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    doc_links.add(match.group(0))
        except Exception as e:
            logger.error(f"Error processing {md_file}: {e}")
    
    # Also check README.txt, docs.txt, and similar files
    additional_files = []
    for ext in [".txt", ".rst"]:
        additional_files.extend(repo_path.glob(f"**/README{ext}"))
        additional_files.extend(repo_path.glob(f"**/DOCS{ext}"))
        additional_files.extend(repo_path.glob(f"**/docs{ext}"))
        additional_files.extend(repo_path.glob(f"**/documentation{ext}"))
    
    for text_file in additional_files:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Search for documentation links using regex patterns
            for pattern in DOC_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    doc_links.add(match.group(0))
        except Exception as e:
            logger.error(f"Error processing {text_file}: {e}")
    
    # Convert to list and return
    links_list = list(doc_links)
    logger.info(f"Found {len(links_list)} unique documentation links")
    
    return links_list

if __name__ == "__main__":
    import sys
    import json
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Detect documentation links in a repository")
    parser.add_argument("repo_path", help="Path to the repository to scan")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger("fetch_docs").setLevel(logging.DEBUG)
    
    # Detect documentation links
    links = detect_documentation_links(Path(args.repo_path))
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(links, f, indent=2)
        print(f"Detected {len(links)} documentation links. Results saved to {args.output}")
    else:
        print("\nDetected documentation links:")
        for link in links:
            print(f"  - {link}")
        print(f"\nTotal: {len(links)} documentation links")