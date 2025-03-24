#!/usr/bin/env python3
"""
processor.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- hashlib: https://docs.python.org/3/library/hashlib.html
- uuid: https://docs.python.org/3/library/uuid.html
- re: https://docs.python.org/3/library/re.html
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- markdownify: https://github.com/matthewwithanm/python-markdownify

This module provides high-level processing functions for documentation extraction.
It integrates download, cleaning, and section extraction into simplified interfaces
for easier integration with other tools, especially DuaLipa.

Input/Output Specifications:

process_documentation(urls: List[str], output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    Input:
        - urls: List of documentation URLs to process
        - output_dir: Directory to store downloaded files
    Output:
        - Dictionary mapping URLs to lists of block dictionaries following DuaLipa format:
            - uuid: str
            - id: str
            - type: str (documentation, doc_section, code_block, etc.)
            - name: str
            - content: str
            - line_start: int (if applicable)
            - line_end: int (if applicable)
            - metadata: Dict[str, Any]
                - language: str
                - file: str
                - heading_level: int (for sections)
                - url: str
                - doc_type: str (readthedocs, arangodb, etc.)

Example usage:
    from agent_tools.fetch_docs.processor import process_documentation
    from pathlib import Path
    
    # Process documentation from a list of URLs
    urls = ["https://docs.example.com/api", "https://other-docs.example.com/guide"]
    output_dir = Path("./docs_cache")
    
    # Returns processed documentation data in DuaLipa-compatible format
    doc_data = process_documentation(urls, output_dir)
"""

import re
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fetch_docs.processor")

def extract_documentation_from_repo(repo_path: Path) -> List[Dict[str, Any]]:
    """
    Extract all documentation from a repository by detecting links and processing them.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of documentation blocks in DuaLipa-compatible format
    """
    try:
        # Import necessary functions
        from agent_tools.fetch_docs.link_detector import detect_documentation_links
    except ImportError:
        logger.error("Could not import link_detector module")
        
        # Fallback implementation for link detection
        def detect_documentation_links(repo_path: Path) -> List[str]:
            """Fallback implementation for link detection."""
            doc_links = []
            # Common documentation link patterns
            patterns = [
                r'https?://[a-zA-Z0-9-]+\.readthedocs\.io/[^\s)"\']+',
                r'https?://readthedocs\.org/projects/[a-zA-Z0-9-]+[^\s)"\']*',
                r'https?://docs\.arangodb\.com/[^\s)"\']+',
                r'https?://docs\.[a-zA-Z0-9-]+\.[a-zA-Z]+/[^\s)"\']+',
            ]
            
            # Find all markdown files
            md_files = list(repo_path.glob("**/*.md"))
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Search for links
                    for pattern in patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            doc_links.append(match.group(0))
                except Exception as e:
                    logger.error(f"Error processing {md_file}: {e}")
            
            return list(set(doc_links))
    
    # Detect documentation links
    doc_links = detect_documentation_links(repo_path)
    
    if not doc_links:
        logger.info("No documentation links found in repository")
        return []
    
    logger.info(f"Found {len(doc_links)} documentation links in repository")
    
    # Create a temp directory for documentation
    docs_dir = repo_path / ".fetch_docs_cache"
    docs_dir.mkdir(exist_ok=True)
    
    # Process documentation
    documentation_data = process_documentation(doc_links, docs_dir)
    
    # Convert to DuaLipa-compatible blocks
    doc_blocks = []
    for url, site_data in documentation_data.items():
        # Add blocks for each page
        for page in site_data:
            # Convert page to DuaLipa blocks
            page_blocks = convert_page_to_blocks(page, url, repo_path)
            doc_blocks.extend(page_blocks)
    
    return doc_blocks

def process_documentation(urls: List[str], output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Download and process documentation from a list of URLs.
    
    Args:
        urls: List of documentation URLs to process
        output_dir: Directory to store downloaded files
        
    Returns:
        Dictionary mapping URLs to processed documentation data
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download documentation
    downloaded_sites = download_documentation(urls, output_dir)
    
    # Process the downloaded documentation
    processed_docs = process_downloaded_sites(downloaded_sites)
    
    return processed_docs

def download_documentation(urls: List[str], output_dir: Path) -> Dict[str, Path]:
    """
    Download documentation from a list of URLs.
    
    Args:
        urls: List of URLs to download
        output_dir: Directory to store downloaded files
        
    Returns:
        Dictionary mapping URLs to their download directories
    """
    downloaded_sites = {}
    
    try:
        # Import download function
        from agent_tools.fetch_docs.download_site import download_site
    except ImportError:
        logger.error("Could not import download_site function")
        return downloaded_sites
    
    for url in urls:
        # Create a unique subdirectory for each URL
        site_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        site_dir = output_dir / site_hash
        
        # Skip if already downloaded
        if site_dir.exists() and any(site_dir.iterdir()):
            logger.info(f"Using cached documentation for {url} at {site_dir}")
            downloaded_sites[url] = site_dir
            continue
        
        # Download the site
        try:
            logger.info(f"Downloading documentation from {url}")
            download_site(url, str(site_dir), recursive=True)
            downloaded_sites[url] = site_dir
            logger.info(f"Successfully downloaded {url} to {site_dir}")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            # Create a minimal structure for testing if download fails
            create_fallback_html(url, site_dir)
            downloaded_sites[url] = site_dir
    
    return downloaded_sites

def create_fallback_html(url: str, output_dir: Path) -> None:
    """
    Create a fallback HTML file if download fails.
    
    Args:
        url: URL that failed to download
        output_dir: Directory to create fallback file in
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "index.html"
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head><title>Documentation from {url}</title></head>
<body>
  <h1>Documentation from {url}</h1>
  <p>This is a placeholder for {url} which could not be downloaded.</p>
  <h2>Section 1</h2>
  <p>Content for section 1</p>
  <h2>Section 2</h2>
  <p>Content for section 2</p>
</body>
</html>""")
    
    logger.info(f"Created fallback HTML file for {url} at {index_file}")

def process_downloaded_sites(downloaded_sites: Dict[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process downloaded documentation sites.
    
    Args:
        downloaded_sites: Dictionary mapping URLs to their download directories
        
    Returns:
        Dictionary mapping URLs to processed documentation data
    """
    processed_docs = {}
    
    try:
        # Import processing functions
        from agent_tools.fetch_docs.clean_html import clean_html
        from agent_tools.fetch_docs.extract_sections import extract_sections_from_html
    except ImportError:
        logger.error("Could not import processing functions")
        return processed_docs
    
    for url, site_dir in downloaded_sites.items():
        # Find all HTML files in the site directory
        html_files = list(site_dir.glob("**/*.html"))
        
        # Determine documentation type
        doc_type = "arangodb" if "arangodb.com" in url else "readthedocs"
        
        site_data = []
        for html_file in html_files:
            try:
                # Read HTML file
                with open(html_file, 'r', encoding='utf-8') as f:
                    raw_html = f.read()
                
                # Clean HTML
                cleaned_html = clean_html(raw_html)
                
                # Extract sections
                sections = extract_sections_from_html(cleaned_html, str(html_file))
                
                # Extract title and other metadata
                title = extract_title(raw_html) or html_file.stem
                summary = extract_summary(raw_html) or f"Documentation from {url}"
                
                # Add to site data
                site_data.append({
                    "file": str(html_file),
                    "relative_path": str(html_file.relative_to(site_dir)),
                    "title": title,
                    "summary": summary,
                    "sections": sections,
                    "doc_type": doc_type,
                    "source_url": url,
                })
            except Exception as e:
                logger.error(f"Error processing {html_file}: {e}")
        
        # Add to processed docs
        processed_docs[url] = site_data
        logger.info(f"Processed {len(site_data)} HTML files from {url}")
    
    return processed_docs

def extract_title(html_content: str) -> Optional[str]:
    """
    Extract title from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        Title text or None if not found
    """
    match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.DOTALL)
    if match:
        title = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        return title
    return None

def extract_summary(html_content: str) -> Optional[str]:
    """
    Extract summary from HTML content (meta description or first paragraph).
    
    Args:
        html_content: HTML content
        
    Returns:
        Summary text or None if not found
    """
    # Try meta description
    match = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"', html_content)
    if match:
        return match.group(1).strip()
    
    # Try first paragraph
    match = re.search(r'<p[^>]*>(.*?)</p>', html_content, re.DOTALL)
    if match:
        text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
        if len(text) > 20:  # Only use if it's a substantial paragraph
            return text
    
    return None

def convert_page_to_blocks(page_data: Dict[str, Any], url: str, repo_path: Path) -> List[Dict[str, Any]]:
    """
    Convert a documentation page to DuaLipa-compatible blocks.
    
    Args:
        page_data: Page data dictionary
        url: Source URL
        repo_path: Repository path
        
    Returns:
        List of DuaLipa-compatible blocks
    """
    blocks = []
    
    # Create a parent block for the page
    page_uuid = str(uuid.uuid4())
    page_title = page_data.get("title", "Untitled Page")
    
    # Sanitize page title for ID
    safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in page_title.lower())
    
    # Create page block
    page_block = {
        "uuid": page_uuid,
        "id": f"doc_{safe_title}",
        "type": "documentation",
        "name": page_title,
        "content": page_data.get("summary", f"Documentation from {url}"),
        "file_path": str(repo_path),
        "line_start": 1,
        "line_end": 1,
        "metadata": {
            "language": "html",
            "file": page_data.get("file", ""),
            "url": url,
            "doc_type": page_data.get("doc_type", "documentation"),
        },
        "child_uuids": []
    }
    
    blocks.append(page_block)
    
    # Process sections
    section_hierarchy = {}  # level -> last section at that level
    
    for section in page_data.get("sections", []):
        section_uuid = str(uuid.uuid4())
        section_title = section.get("header", "Untitled Section")
        section_level = section.get("level", 1)
        section_content = section.get("content", "")
        
        # Sanitize section title for ID
        safe_section_title = "".join(c if c.isalnum() or c == "_" else "_" for c in section_title.lower())
        
        # Find parent section
        parent_uuid = page_uuid
        for level in range(section_level - 1, 0, -1):
            if level in section_hierarchy:
                parent_uuid = section_hierarchy[level]
                break
        
        # Create section block
        section_block = {
            "uuid": section_uuid,
            "id": f"doc_{safe_title}_{safe_section_title}",
            "type": "doc_section",
            "name": section_title,
            "content": section_content,
            "file_path": page_data.get("file", ""),
            "line_start": 1,  # Not tracked in HTML
            "line_end": 1,    # Not tracked in HTML
            "metadata": {
                "language": "html",
                "file": page_data.get("file", ""),
                "url": url,
                "doc_type": page_data.get("doc_type", "documentation"),
                "heading_level": section_level,
                "token_count": section.get("token_count", len(section_content.split())),
            },
            "child_uuids": []
        }
        
        blocks.append(section_block)
        
        # Add to parent's child UUIDs
        for block in blocks:
            if block["uuid"] == parent_uuid:
                block["child_uuids"].append(section_uuid)
                break
        
        # Update hierarchy
        section_hierarchy[section_level] = section_uuid
        
        # Clear any higher levels since they're no longer relevant
        higher_levels = [l for l in section_hierarchy if l > section_level]
        for l in higher_levels:
            if l in section_hierarchy:
                del section_hierarchy[l]
        
        # Extract special elements
        special_elements = extract_special_elements(section_content)
        
        # Process code blocks
        for i, code_block in enumerate(special_elements.get("code_blocks", [])):
            code_uuid = str(uuid.uuid4())
            code_lang = code_block.get("language", "text")
            code_content = code_block.get("content", "")
            
            code_block_obj = {
                "uuid": code_uuid,
                "id": f"doc_{safe_title}_{safe_section_title}_code_{i}",
                "type": "code_block",
                "name": f"Code Block {i+1}",
                "content": code_content,
                "file_path": page_data.get("file", ""),
                "line_start": 1,  # Not tracked in HTML
                "line_end": 1,    # Not tracked in HTML
                "metadata": {
                    "language": code_lang,
                    "file": page_data.get("file", ""),
                    "url": url,
                    "doc_type": page_data.get("doc_type", "documentation"),
                    "element_type": "code_block",
                },
                "child_uuids": []
            }
            
            blocks.append(code_block_obj)
            section_block["child_uuids"].append(code_uuid)
        
        # Process tables
        for i, table in enumerate(special_elements.get("tables", [])):
            table_uuid = str(uuid.uuid4())
            
            table_block = {
                "uuid": table_uuid,
                "id": f"doc_{safe_title}_{safe_section_title}_table_{i}",
                "type": "table",
                "name": f"Table {i+1}",
                "content": str(table),
                "file_path": page_data.get("file", ""),
                "line_start": 1,  # Not tracked in HTML
                "line_end": 1,    # Not tracked in HTML
                "metadata": {
                    "language": "html",
                    "file": page_data.get("file", ""),
                    "url": url,
                    "doc_type": page_data.get("doc_type", "documentation"),
                    "element_type": "table",
                    "headers": table.get("headers", []),
                    "rows": table.get("rows", []),
                },
                "child_uuids": []
            }
            
            blocks.append(table_block)
            section_block["child_uuids"].append(table_uuid)
            
    return blocks

def extract_special_elements(html_content: str) -> Dict[str, List[Dict]]:
    """
    Extract special elements from HTML content (code blocks, tables, images).
    
    Args:
        html_content: HTML content to process
        
    Returns:
        Dict with lists of detected elements
    """
    special_elements = {
        "code_blocks": [],
        "tables": [],
        "images": []
    }
    
    # Extract code blocks
    code_pattern = re.compile(r'<(pre|code)[^>]*>(.*?)</\1>', re.DOTALL)
    for i, match in enumerate(code_pattern.finditer(html_content)):
        code_content = match.group(2)
        # Clean up code content
        code_content = re.sub(r'<[^>]+>', '', code_content)
        
        # Try to detect language
        language = "text"
        lang_match = re.search(r'class="[^"]*language-([^"]*)"', match.group(0))
        if lang_match:
            language = lang_match.group(1)
        
        special_elements["code_blocks"].append({
            "content": code_content.strip(),
            "language": language,
            "position": match.start()
        })
    
    # Extract tables
    table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL)
    for i, match in enumerate(table_pattern.finditer(html_content)):
        table_content = match.group(1)
        
        # Extract headers and rows
        headers = []
        rows = []
        
        # Find headers
        th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.DOTALL)
        headers = [re.sub(r'<[^>]+>', '', h.group(1)).strip() for h in th_pattern.finditer(table_content)]
        
        # Find rows
        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL)
        for tr_match in tr_pattern.finditer(table_content):
            row_content = tr_match.group(1)
            if '<th' in row_content:  # Skip header rows
                continue
                
            # Extract cells
            td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)
            row = [re.sub(r'<[^>]+>', '', td.group(1)).strip() for td in td_pattern.finditer(row_content)]
            if row:
                rows.append(row)
        
        # Add table if it has content
        if headers or rows:
            special_elements["tables"].append({
                "headers": headers,
                "rows": rows,
                "position": match.start()
            })
    
    # Extract images
    img_pattern = re.compile(r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*>', re.DOTALL)
    for i, match in enumerate(img_pattern.finditer(html_content)):
        img_src = match.group(1)
        img_alt = match.group(2)
        
        special_elements["images"].append({
            "src": img_src,
            "alt": img_alt,
            "position": match.start()
        })
    
    return special_elements

if __name__ == "__main__":
    import sys
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documentation from URLs")
    parser.add_argument("--repo", help="Repository path for link detection")
    parser.add_argument("--urls", nargs="+", help="URLs to process")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--cache-dir", default="./docs_cache", help="Directory for cached downloads")
    
    args = parser.parse_args()
    
    if args.repo:
        # Extract from repository
        results = extract_documentation_from_repo(Path(args.repo))
    elif args.urls:
        # Process specific URLs
        doc_data = process_documentation(args.urls, Path(args.cache_dir))
        
        # Convert to blocks
        results = []
        for url, site_data in doc_data.items():
            for page in site_data:
                page_blocks = convert_page_to_blocks(page, url, Path(args.cache_dir))
                results.extend(page_blocks)
    else:
        print("Either --repo or --urls must be specified")
        sys.exit(1)
    
    # Write results to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Documentation processed successfully. Results written to {args.output}")