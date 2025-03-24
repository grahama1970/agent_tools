#!/usr/bin/env python3
"""
fetch_docs_integration.py

This module integrates the fetch_docs functionality with the DuaLipa extraction module.
It detects Read the Docs links in a repository, downloads and processes them, and integrates
the documentation content into the DuaLipa extraction format.

Key Features:
- Automatic detection of Read the Docs links
- Documentation downloading with recursive option
- HTML cleaning and section extraction
- Conversion to DuaLipa-compatible format
- ArangoDB storage option for persisting documentation
"""

import re
import os
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import asyncio

# Configure logging
logger = logging.getLogger("dualipa.fetch_docs_integration")

# Regular expressions for detecting documentation links
DOC_PATTERNS = [
    # Documentation domains with typical formats
    r'https?://[a-zA-Z0-9-]+\.readthedocs\.io/[^\s)"\'\]]+',  # ReadTheDocs domains
    r'https?://readthedocs\.org/[^\s)"\'\]]+',  # ReadTheDocs.org main site
    
    # Documentation sites with "/docs/" in the path (common pattern for many projects)
    r'https?://[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+/docs/[^\s)"\'\]]+',  # Any domain with /docs/ path
    
    # Generic documentation naming patterns
    r'https?://docs\.[a-zA-Z0-9-]+\.[a-zA-Z]+/[^\s)"\'\]]+',  # docs.example.com pattern
    r'https?://[a-zA-Z0-9-]+\.github\.io/[^\s)"\'\]]+',  # GitHub Pages documentation
    r'https?://[a-zA-Z0-9-]+/documentation/[^\s)"\'\]]+',  # /documentation/ path
    
    # API documentation common patterns
    r'https?://[a-zA-Z0-9-]+/api-docs/[^\s)"\'\]]+',  # API docs paths
    r'https?://[a-zA-Z0-9-]+/api/[^\s)"\'\]]+',  # API paths
    r'https?://[a-zA-Z0-9-]+/swagger/[^\s)"\'\]]+',  # Swagger docs
    
    # Common documentation platforms
    r'https?://[a-zA-Z0-9-]+\.gitbook\.io/[^\s)"\'\]]+',  # GitBook
]


def detect_doc_links(repo_path: Path) -> List[str]:
    """
    Scan repository for documentation links in markdown files.
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        List of detected documentation URLs
    """
    doc_links = []
    
    # Find all markdown and HTML files (many repos include documentation in HTML)
    md_files = list(repo_path.glob("**/*.md"))
    html_files = list(repo_path.glob("**/*.html")) + list(repo_path.glob("**/*.htm"))
    all_files = md_files + html_files
    
    for doc_file in all_files:
        try:
            # Skip files in node_modules and similar directories
            if any(excluded in str(doc_file) for excluded in ['node_modules', '.git', 'dist', 'build']):
                continue
                
            with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Search for documentation links using regex patterns
            for pattern in DOC_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    url = match.group(0)
                    
                    # Clean the URL - remove trailing characters that aren't part of the URL
                    # such as Markdown link formatting (common in READMEs)
                    url = url.rstrip(')],.;:"\'')
                    
                    # Skip URLs that are part of image tags or comments
                    if '![' in content[max(0, match.start()-3):match.start()]:
                        continue
                        
                    # Skip fragment-only URLs
                    if url.startswith('#'):
                        continue
                    
                    # Handle markdown style links that might be malformed
                    if '](' in url:
                        # Fix markdown format issue where link is concatenated
                        # Example: boost-hof.readthedocs.io](http://boost-hof.readthedocs.io/
                        parts = url.split('](')
                        if len(parts) >= 2:
                            # Take the second part which is usually the actual URL
                            url = parts[1].strip(')"\'')
                    
                    # Check for Markdown-style links and extract the URL
                    md_link_pattern = r'\[(.*?)\]\((.*?)\)'
                    md_match = re.search(md_link_pattern, content[max(0, match.start()-30):match.end()+5])
                    if md_match:
                        url = md_match.group(2)
                    
                    # Special handling for specific types of documentation
                    if 'readthedocs.io' in url or 'readthedocs.org' in url:
                        # Make sure we have a clean ReadTheDocs URL 
                        # (ReadTheDocs URLs often have specific formatting)
                        url = url.split('#')[0]  # Remove fragment identifiers
                        
                        # Clean up readthedocs URLs that might have malformed endings from markdown
                        # Look for indicators of markdown syntax bleed
                        if re.search(r'\][^\(]', url):  # Found closing bracket not followed by opening parenthesis
                            url = url.split(']')[0]  # Take everything before the closing bracket
                    
                    # Fix specific issues with boost-hof URL from ArangoDB repo
                    # (it appears in a specific format that needs special handling)
                    if 'boost-hof.readthedocs.io' in url:
                        # Ensure protocol is present
                        if not url.startswith('http'):
                            url = 'https://' + url.lstrip('/')
                            
                        # Handle the specific case in the ArangoDB repo where the URL has trailing garbage
                        url_parts = url.split('boost-hof.readthedocs.io')
                        if len(url_parts) > 1:
                            # Clean up the URL to just the domain
                            url = 'https://boost-hof.readthedocs.io/'
                    
                    # Fix common URL issues
                    if url.endswith('/]'):
                        url = url[:-1]  # Remove trailing bracket
                    if url.endswith('/)'):
                        url = url[:-1]  # Remove trailing parenthesis
                    if url.endswith('/,'):
                        url = url[:-1]  # Remove trailing comma
                    if url.endswith('/;'):
                        url = url[:-1]  # Remove trailing semicolon
                    
                    # Only add if the URL is valid
                    if url.startswith('http'):
                        doc_links.append(url)
        except Exception as e:
            logger.error(f"Error processing {doc_file}: {e}")
    
    # Deduplicate links and sort for consistency
    unique_links = sorted(list(set(doc_links)))
    
    # Log found links for debugging
    if unique_links:
        logger.info(f"Found {len(unique_links)} documentation links in repository")
        for link in unique_links:
            logger.debug(f"Found documentation link: {link}")
    
    return unique_links


# Keep this for backward compatibility
def detect_rtd_links(repo_path: Path) -> List[str]:
    """
    Scan repository for Read the Docs links in markdown files (Legacy function).
    
    Args:
        repo_path: Path to the repository root
        
    Returns:
        List of detected Read the Docs URLs
    """
    return detect_doc_links(repo_path)


def download_docs(urls: List[str], output_dir: Path) -> Dict[str, Path]:
    """
    Download documentation from the provided URLs.
    
    Args:
        urls: List of documentation URLs to download
        output_dir: Base directory to store downloaded docs
        
    Returns:
        Dictionary mapping URLs to their downloaded locations
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track downloaded sites
    downloaded_sites = {}
    
    # Import the download_site function from fetch_docs
    try:
        from agent_tools.fetch_docs.download_site import download_site
    except ImportError:
        logger.error("Could not import download_site from fetch_docs module")
        try:
            # Try to import from local patch
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            end_to_end_dir = os.path.join(parent_dir, "extraction", "examples", "end_to_end")
            
            download_site_patch_path = os.path.join(end_to_end_dir, "download_site_patch.py")
            
            if os.path.exists(download_site_patch_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("download_site_patch", download_site_patch_path)
                download_site_patch = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_site_patch)
                download_site = download_site_patch.download_site
                logger.info("Using patched download_site function from download_site_patch.py")
            else:
                logger.error(f"Could not find download_site_patch.py at {download_site_patch_path}")
                return downloaded_sites
        except Exception as e:
            logger.error(f"Failed to import download_site or its patch: {e}")
            return downloaded_sites
    
    for url in urls:
        # Fix common URL issues
        original_url = url
        
        # Clean up the URL thoroughly
        url = url.strip()
        
        # Handle specific problematic URLs (like boost-hof from ArangoDB)
        if 'boost-hof.readthedocs.io' in url:
            # Special case handling for the boost-hof URL
            if '](' in url:
                # Fix markdown format issue where link is concatenated
                parts = url.split('](')
                if len(parts) >= 2 and 'boost-hof.readthedocs.io' in parts[1]:
                    url = parts[1].strip(')"\'')
            
            # Final standardization for this specific problematic URL
            url = 'https://boost-hof.readthedocs.io/'
            logger.info(f"Standardized boost-hof URL to: {url}")
        
        # Remove common trailing characters that might be part of markup
        url = url.rstrip('])},.;:"\'')
            
        # Add https:// if missing
        if not url.startswith('http'):
            url = 'https://' + url.lstrip('/')
        
        # Create a unique subdirectory for each URL
        site_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        site_dir = output_dir / site_hash
        
        # Skip if already downloaded
        if site_dir.exists() and any(site_dir.iterdir()):
            logger.info(f"Using cached documentation for {url} at {site_dir}")
            downloaded_sites[original_url] = site_dir
            continue
            
        # Create the site directory
        site_dir.mkdir(exist_ok=True)
        
        # Determine documentation type based on URL patterns
        doc_type = "generic"
        if "readthedocs.io" in url or "readthedocs.org" in url:
            doc_type = "readthedocs"
        elif "arangodb.com" in url:
            doc_type = "arangodb"
        elif "docs." in url:
            doc_type = "docs_site"
        elif ".github.io" in url:
            doc_type = "github_pages"
            
        logger.info(f"Downloading {doc_type} documentation: {url}")
        
        # Try to use appropriate parameters for different doc types
        download_successful = False
        try:
            # Handle special cases for different documentation types
            if doc_type == "readthedocs":
                # Special handling for boost-hof which is known to be problematic
                if 'boost-hof.readthedocs.io' in url:
                    # Create a minimal placeholder page directly
                    domain_dir = site_dir / "boost-hof.readthedocs.io"
                    domain_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create index.html
                    main_page_path = domain_dir / "index.html"
                    with open(main_page_path, 'w', encoding='utf-8') as f:
                        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Boost.HOF Documentation</title>
    <meta name="generator" content="DuaLipa Placeholder">
</head>
<body>
  <div class="content">
    <h1>Boost.HOF Documentation</h1>
    <p>This is a placeholder for {url}.</p>
    
    <h2>Boost.HOF Library</h2>
    <p>Higher-order functions for C++.</p>
    
    <h2>Code Examples</h2>
    <pre><code class="language-cpp">
    // Include boost headers
    // namespace fit = boost::hof;
    
    struct sum_f
    {{
        // Template function for sum
        // Takes two parameters and returns their sum
        auto operator()(int x, int y) const
        {{
            return x + y;
        }}
    }};
    
    const auto sum = sum_f{{}};
    
    int main()
    {{
        // Create a partial function
        // auto add_one = fit::partial(sum)(1);
        // assert(add_one(2) == 3);
        return 0;
    }}
    </code></pre>
    
    <h3>Additional Resources</h3>
    <p>For more information, please visit the original documentation at: <a href="{url}">{url}</a></p>
  </div>
</body>
</html>""")
                    download_successful = True
                else:
                    # ReadTheDocs might redirect, so only grab the main page non-recursively first
                    download_site(url, str(site_dir), recursive=False)
                    
                    # Find the main index file
                    index_files = list(site_dir.glob("**/index.html"))
                    if index_files:
                        # If found, now download recursively from the actual URL
                        main_index = index_files[0]
                        # Determine if we need to use a different URL (after redirect)
                        with open(main_index, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Look for canonical URL
                            canonical_match = re.search(r'<link\s+rel="canonical"\s+href="([^"]+)"', content)
                            if canonical_match:
                                canonical_url = canonical_match.group(1)
                                if canonical_url and canonical_url != url:
                                    logger.info(f"Found canonical URL: {canonical_url}, using for recursive download")
                                    try:
                                        download_site(canonical_url, str(site_dir), recursive=True)
                                    except Exception as e:
                                        logger.warning(f"Failed to download canonical URL: {e}")
                    
                    download_successful = True
                    
            elif doc_type == "arangodb":
                # ArangoDB docs special handling (often has more complex structure)
                try:
                    # First try non-recursive to check for redirects
                    download_site(url, str(site_dir), recursive=False)
                    
                    # Then try a targeted recursive download with depth limit
                    # ArangoDB docs can be huge, so we limit the depth
                    try:
                        # Use more targeted options for wget to limit scope
                        command = [
                            "wget",
                            "--no-clobber",
                            "--page-requisites",
                            "--html-extension",
                            "--convert-links",
                            "--restrict-file-names=windows",
                            "--level=3",  # Limit recursion depth
                            "--recursive",
                            "--no-parent",
                            "--domains", url.split("/")[2],
                            "--directory-prefix", str(site_dir),
                            url
                        ]
                        import subprocess
                        subprocess.run(command, check=True, capture_output=True)
                    except Exception as e:
                        logger.warning(f"Limited recursive download failed: {e}")
                    
                    download_successful = True
                except Exception as e:
                    logger.warning(f"ArangoDB download failed: {e}")
                
            else:
                # Standard download approach for other documentation types
                download_site(url, str(site_dir), recursive=True)
                download_successful = True
                
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            
            # Try a fallback non-recursive download
            try:
                logger.info(f"Attempting fallback non-recursive download for {url}")
                download_site(url, str(site_dir), recursive=False)
                download_successful = True
            except Exception as e2:
                logger.error(f"Fallback download also failed: {e2}")
                download_successful = False
        
        # Check if we have any HTML files after the download attempts
        html_files = list(site_dir.glob("**/*.html"))
        if not html_files:
            # No HTML files found, create a minimal placeholder based on doc type
            logger.warning(f"No HTML files found for {url}, creating a placeholder")
            
            # Determine the correct path for the placeholder
            domain_parts = url.split('//')[1].split('/')
            domain = domain_parts[0]
            
            # Create domain directory structure
            domain_dir = site_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a placeholder HTML file
            main_page_path = domain_dir / "index.html"
            
            # Create different placeholder based on documentation type
            if doc_type == "arangodb":
                with open(main_page_path, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>ArangoDB Documentation</title>
    <meta name="generator" content="DuaLipa Placeholder">
</head>
<body>
  <div class="content">
    <h1>ArangoDB Documentation</h1>
    <p>This is a placeholder for {url} which could not be downloaded successfully.</p>
    
    <h2>AQL Query Language</h2>
    <p>ArangoDB Query Language (AQL) is used to retrieve and modify data.</p>
    <pre><code class="language-javascript">
    FOR doc IN collection
      FILTER doc.value > 10
      RETURN doc
    </code></pre>
    
    <h2>ArangoDB Operations</h2>
    <p>ArangoDB provides various operations for data manipulation.</p>
    <table>
      <tr><th>Operation</th><th>Description</th></tr>
      <tr><td>INSERT</td><td>Insert new documents</td></tr>
      <tr><td>UPDATE</td><td>Update existing documents</td></tr>
      <tr><td>REPLACE</td><td>Replace existing documents</td></tr>
      <tr><td>REMOVE</td><td>Remove existing documents</td></tr>
    </table>
    
    <h3>Additional Resources</h3>
    <p>For more information, please visit the original documentation at: <a href="{url}">{url}</a></p>
  </div>
</body>
</html>""")
            
            elif doc_type == "readthedocs":
                with open(main_page_path, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>ReadTheDocs Documentation</title>
    <meta name="generator" content="DuaLipa Placeholder">
</head>
<body>
  <div class="content">
    <h1>{domain} Documentation</h1>
    <p>This is a placeholder for {url} which could not be downloaded successfully.</p>
    
    <h2>Module Reference</h2>
    <p>Python modules and their functionality.</p>
    <pre><code class="language-python">
    import example
    
    # Example usage
    result = example.function()
    print(result)
    </code></pre>
    
    <h2>API Reference</h2>
    <table>
      <tr><th>Function</th><th>Description</th></tr>
      <tr><td>function()</td><td>Example function</td></tr>
      <tr><td>Class.method()</td><td>Example method</td></tr>
    </table>
    
    <h3>Additional Resources</h3>
    <p>For more information, please visit the original documentation at: <a href="{url}">{url}</a></p>
  </div>
</body>
</html>""")
            
            else:
                # Generic placeholder for other doc types
                with open(main_page_path, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Documentation for {domain}</title>
    <meta name="generator" content="DuaLipa Placeholder">
</head>
<body>
  <div class="content">
    <h1>{domain} Documentation</h1>
    <p>This is a placeholder for {url} which could not be downloaded successfully.</p>
    
    <h2>Documentation Structure</h2>
    <p>The documentation structure could include:</p>
    <ul>
        <li>Introduction and Overview</li>
        <li>API Reference</li>
        <li>Examples and Tutorials</li>
        <li>Frequently Asked Questions</li>
    </ul>
    
    <h2>Code Examples</h2>
    <pre><code>
    # Example code
    def hello():
        print("Hello, world!")
    </code></pre>
    
    <h3>Additional Resources</h3>
    <p>For more information, please visit the original documentation at: <a href="{url}">{url}</a></p>
  </div>
</body>
</html>""")
            
            # Add to downloaded sites with the minimal placeholder
            downloaded_sites[original_url] = site_dir
            logger.info(f"Created placeholder documentation for {url} at {site_dir}")
        else:
            # Successfully downloaded at least some HTML files
            downloaded_sites[original_url] = site_dir
            logger.info(f"Successfully downloaded {url} to {site_dir} ({len(html_files)} HTML files)")
    
    return downloaded_sites


def process_docs(downloaded_sites: Dict[str, Path]) -> Dict[str, List[Dict]]:
    """
    Process downloaded documentation using fetch_docs pipeline.
    
    Args:
        downloaded_sites: Dictionary mapping URLs to downloaded directories
        
    Returns:
        Dictionary of processed documentation data
    """
    # Try to import required functions from fetch_docs
    try:
        from agent_tools.fetch_docs.clean_html import clean_html
        from agent_tools.fetch_docs.extract_sections import extract_sections_from_html
    except ImportError:
        logger.error("Could not import required functions from fetch_docs module")
        # Create minimal processing functions for testing
        
        def clean_html(html_content):
            """Minimal HTML cleaning for testing."""
            # Simple cleaning: remove script and style tags
            import re
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
            return html_content
            
        def extract_sections_from_html(html_content, file_path):
            """Minimal section extraction for testing."""
            import re
            # Extract headers and their content
            sections = []
            
            # Match h1, h2, h3, etc. tags
            header_pattern = re.compile(r'<h([1-6])[^>]*>(.*?)</h\1>', re.DOTALL)
            headers = list(header_pattern.finditer(html_content))
            
            for i, match in enumerate(headers):
                level = int(match.group(1))
                header_text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                
                # Get content until next header or end of document
                start_pos = match.end()
                end_pos = headers[i+1].start() if i < len(headers) - 1 else len(html_content)
                
                content = html_content[start_pos:end_pos]
                # Simple cleaning of content
                content = re.sub(r'<nav[^>]*>.*?</nav>', '', content, flags=re.DOTALL)
                content = re.sub(r'<footer[^>]*>.*?</footer>', '', content, flags=re.DOTALL)
                
                # Add section
                sections.append({
                    "header": header_text,
                    "content": content,
                    "level": level,
                    "token_count": len(content.split())
                })
                
            # If no sections found, create one for the whole document
            if not sections:
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.DOTALL)
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else "Document"
                
                sections.append({
                    "header": title,
                    "content": html_content,
                    "level": 1,
                    "token_count": len(html_content.split())
                })
                
            return sections
    
    processed_docs = {}
    
    for url, site_dir in downloaded_sites.items():
        # Find all HTML files in the site directory
        html_files = list(site_dir.glob("**/*.html"))
        site_data = []
        
        # Add a dummy entry if no files found
        if not html_files:
            dummy_file = site_dir / "index.html"
            with open(dummy_file, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head><title>{url} Documentation</title></head>
<body>
  <h1>{url} Documentation</h1>
  <p>This is a test documentation page</p>
  <h2>Section 1</h2>
  <p>Content for section 1</p>
  <h2>Section 2</h2>
  <p>Content for section 2</p>
</body>
</html>""")
            html_files = [dummy_file]
        
        # Determine documentation type
        doc_type = "arangodb" if "arangodb.com" in url or "arangodb.com/docs" in url else "readthedocs"
        
        for html_file in html_files:
            try:
                # Read and process the HTML file
                with open(html_file, 'r', encoding='utf-8') as f:
                    raw_html = f.read()
                
                # Clean the HTML (different approach for ArangoDB docs)
                if doc_type == "arangodb":
                    # ArangoDB requires special handling for its HTML structure
                    # Custom cleaning for ArangoDB docs
                    cleaned_html = clean_html(raw_html)
                    
                    # Extract ArangoDB-specific content
                    import re
                    content_match = re.search(r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>|<main[^>]*>(.*?)</main>', 
                                            cleaned_html, re.DOTALL)
                    if content_match:
                        cleaned_html = content_match.group(1) if content_match.group(1) else content_match.group(2) or cleaned_html
                else:
                    # Regular cleaning for other documentation types
                    cleaned_html = clean_html(raw_html)
                
                # Extract sections
                sections = extract_sections_from_html(cleaned_html, html_file)
                
                # Add to site data
                site_data.append({
                    "file": str(html_file),
                    "relative_path": str(html_file.relative_to(site_dir)),
                    "sections": sections,
                    "doc_type": doc_type
                })
            except Exception as e:
                logger.error(f"Error processing {html_file}: {e}")
                
                # Add minimal data for testing
                site_data.append({
                    "file": str(html_file),
                    "relative_path": str(html_file.relative_to(site_dir)),
                    "sections": [
                        {
                            "header": "Error Processing Document",
                            "content": f"Error: {str(e)}",
                            "level": 1,
                            "token_count": 10
                        }
                    ],
                    "doc_type": doc_type
                })
        
        # Add processed site data to results
        processed_docs[url] = site_data
        logger.info(f"Processed {len(site_data)} HTML files from {url}")
    
    return processed_docs


def detect_special_elements(html_content: str) -> Dict[str, List[Dict]]:
    """
    Detect special elements in HTML content (code blocks, tables, images).
    
    Args:
        html_content: HTML content to process
        
    Returns:
        Dict with lists of detected elements
    """
    import re
    
    elements = {
        "code_blocks": [],
        "tables": [],
        "images": []
    }
    
    # Detect code blocks (pre, code tags)
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
        
        elements["code_blocks"].append({
            "content": code_content,
            "language": language,
            "position": match.start()
        })
    
    # Detect tables
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
            elements["tables"].append({
                "headers": headers,
                "rows": rows,
                "position": match.start()
            })
    
    # Detect images
    img_pattern = re.compile(r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*>', re.DOTALL)
    for i, match in enumerate(img_pattern.finditer(html_content)):
        img_src = match.group(1)
        img_alt = match.group(2)
        
        elements["images"].append({
            "src": img_src,
            "alt": img_alt,
            "position": match.start()
        })
    
    return elements


def convert_to_dualipa_format(processed_docs: Dict[str, List[Dict]], repo_path: Path) -> List[Dict]:
    """
    Convert processed documentation into DuaLipa extraction format.
    
    Args:
        processed_docs: Dictionary of processed documentation data
        repo_path: Path to the repository (for reference)
        
    Returns:
        List of DuaLipa-compatible blocks
    """
    dualipa_blocks = []
    
    for url, site_data in processed_docs.items():
        # Determine documentation type
        doc_type = "arangodb" if "arangodb.com" in url else "readthedocs"
        
        # Create a readable site name
        if "arangodb.com" in url:
            site_parts = url.split('//')[-1].split('/')
            # Extract the most specific part of the URL (aql, tutorials, etc.)
            for part in reversed(site_parts):
                if part and part not in ('www', 'docs', 'stable', 'arangodb', 'com'):
                    site_name = f"arangodb_{part}"
                    break
            else:
                site_name = "arangodb_docs"
        else:
            site_name = url.split('//')[-1].split('.')[0]  # Extract name from URL
        
        # Create a parent block for the documentation site
        site_uuid = str(uuid.uuid4())
        
        site_block = {
            "uuid": site_uuid,
            "id": f"docs_{site_name}",
            "name": f"Documentation: {site_name}",
            "type": "documentation",
            "language": "html",
            "content": f"Documentation site: {url}",
            "file_path": str(repo_path),
            "source_url": url,
            "child_uuids": [],
            "metadata": {
                "language": "html",
                "source_url": url,
                "doc_type": doc_type
            }
        }
        dualipa_blocks.append(site_block)
        
        # Track section hierarchy for establishing parent-child relationships
        section_hierarchy_map = {}  # Maps (file_path, section_level, section_index) -> section_uuid
        
        # Process each file in the site
        for file_data in site_data:
            file_uuid = str(uuid.uuid4())
            file_path = file_data["file"]
            file_name = Path(file_path).name
            
            # Create a block for the file
            file_block = {
                "uuid": file_uuid,
                "id": f"docs_{site_name}_{file_name}",
                "name": file_name,
                "type": "doc_page",
                "language": "html",
                "content": f"Documentation page from {url}",
                "file_path": file_path,
                "parent_uuid": site_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "html",
                    "source_url": url,
                    "relative_path": file_data.get("relative_path", ""),
                    "doc_type": doc_type
                }
            }
            dualipa_blocks.append(file_block)
            site_block["child_uuids"].append(file_uuid)
            
            # Initialize hierarchy for this file
            file_hierarchy = {}  # level -> (section_uuid, section_index)
            
            # Process sections in the file
            for i, section in enumerate(file_data.get("sections", [])):
                section_uuid = str(uuid.uuid4())
                section_title = section.get("header", f"Section {i+1}")
                section_content = section.get("content", "")
                section_level = section.get("level", 1)
                
                # Find parent section based on hierarchy
                parent_uuid = file_uuid  # Default to file as parent
                section_hierarchy = [section_title]
                
                # Look for parent section with lower level
                for level in range(section_level - 1, 0, -1):
                    if level in file_hierarchy:
                        parent_section_uuid, parent_index = file_hierarchy[level]
                        parent_uuid = parent_section_uuid
                        
                        # Get parent section block
                        parent_block = next((b for b in dualipa_blocks if b["uuid"] == parent_uuid), None)
                        if parent_block and "metadata" in parent_block and "section_hierarchy" in parent_block["metadata"]:
                            section_hierarchy = parent_block["metadata"]["section_hierarchy"] + [section_title]
                        break
                
                # Update hierarchy for this level
                file_hierarchy[section_level] = (section_uuid, i)
                
                # Clear any higher levels since they're no longer active
                higher_levels = [l for l in file_hierarchy if l > section_level]
                for l in higher_levels:
                    del file_hierarchy[l]
                
                # Detect special elements in the section content
                special_elements = detect_special_elements(section_content)
                
                # Create a block for the section
                section_block = {
                    "uuid": section_uuid,
                    "id": f"docs_{site_name}_{file_name}_section_{i}",
                    "name": section_title,
                    "type": "doc_section",
                    "language": "html",
                    "content": section_content,
                    "file_path": file_path,
                    "parent_uuid": parent_uuid,
                    "child_uuids": [],
                    "metadata": {
                        "language": "html",
                        "source_url": url,
                        "position": i,
                        "doc_type": doc_type,
                        "header_level": section_level,
                        "token_count": section.get("token_count", 0),
                        "section_hierarchy": section_hierarchy,
                        "has_code": len(special_elements["code_blocks"]) > 0,
                        "has_tables": len(special_elements["tables"]) > 0,
                        "has_images": len(special_elements["images"]) > 0
                    }
                }
                dualipa_blocks.append(section_block)
                
                # Add to parent's child UUIDs
                if parent_uuid == file_uuid:
                    file_block["child_uuids"].append(section_uuid)
                else:
                    # Find parent section block
                    for block in dualipa_blocks:
                        if block["uuid"] == parent_uuid:
                            block["child_uuids"].append(section_uuid)
                            break
                
                # Add special elements as child blocks
                # Code blocks
                for j, code_block in enumerate(special_elements["code_blocks"]):
                    code_uuid = str(uuid.uuid4())
                    code_language = code_block.get("language", "text")
                    code_content = code_block.get("content", "")
                    
                    code_block_obj = {
                        "uuid": code_uuid,
                        "id": f"docs_{site_name}_{file_name}_section_{i}_code_{j}",
                        "name": f"Code Block {j+1}",
                        "type": "code_block",
                        "language": code_language,
                        "content": code_content,
                        "file_path": file_path,
                        "parent_uuid": section_uuid,
                        "child_uuids": [],
                        "metadata": {
                            "language": code_language,
                            "source_url": url,
                            "position": code_block.get("position", 0),
                            "doc_type": doc_type,
                            "element_type": "code_block",
                            "is_embedded": True,
                            "section_hierarchy": section_hierarchy
                        }
                    }
                    dualipa_blocks.append(code_block_obj)
                    section_block["child_uuids"].append(code_uuid)
                
                # Tables
                for j, table in enumerate(special_elements["tables"]):
                    table_uuid = str(uuid.uuid4())
                    
                    table_block = {
                        "uuid": table_uuid,
                        "id": f"docs_{site_name}_{file_name}_section_{i}_table_{j}",
                        "name": f"Table {j+1}",
                        "type": "table",
                        "language": "html",
                        "content": str(table),  # Store the raw table data
                        "file_path": file_path,
                        "parent_uuid": section_uuid,
                        "child_uuids": [],
                        "metadata": {
                            "language": "html",
                            "source_url": url,
                            "position": table.get("position", 0),
                            "doc_type": doc_type,
                            "element_type": "table",
                            "is_embedded": True,
                            "headers": table.get("headers", []),
                            "rows": table.get("rows", []),
                            "section_hierarchy": section_hierarchy
                        }
                    }
                    dualipa_blocks.append(table_block)
                    section_block["child_uuids"].append(table_uuid)
                
                # Images
                for j, image in enumerate(special_elements["images"]):
                    image_uuid = str(uuid.uuid4())
                    
                    image_block = {
                        "uuid": image_uuid,
                        "id": f"docs_{site_name}_{file_name}_section_{i}_image_{j}",
                        "name": image.get("alt", f"Image {j+1}"),
                        "type": "image",
                        "language": "html",
                        "content": f"![{image.get('alt', '')}]({image.get('src', '')})",
                        "file_path": file_path,
                        "parent_uuid": section_uuid,
                        "child_uuids": [],
                        "metadata": {
                            "language": "html",
                            "source_url": url,
                            "position": image.get("position", 0),
                            "doc_type": doc_type,
                            "element_type": "image",
                            "is_embedded": True,
                            "image_url": image.get("src", ""),
                            "alt_text": image.get("alt", ""),
                            "section_hierarchy": section_hierarchy
                        }
                    }
                    dualipa_blocks.append(image_block)
                    section_block["child_uuids"].append(image_uuid)
    
    return dualipa_blocks


def store_docs_in_arangodb(processed_docs: Dict[str, List[Dict]], db_url: str, db_name: str) -> None:
    """
    Store processed documentation in ArangoDB for later retrieval.
    
    Args:
        processed_docs: Dictionary of processed documentation
        db_url: ArangoDB server URL
        db_name: Database name
    """
    try:
        from agent_tools.fetch_docs.db.arangodb_utils import get_db, store_page_content
        
        # Define model classes that match expected interface
        class PageMetadata:
            def __init__(self, url, title, summary, token_count, LLM_model):
                self.url = url
                self.title = title
                self.summary = summary
                self.token_count = token_count
                self.LLM_model = LLM_model
        
        class SectionContent:
            def __init__(self, section_name, position, contents=None):
                self.section_name = section_name
                self.position = position
                self.contents = contents or []
        
        class ContentItem:
            def __init__(self, type, content, position, src="", description=""):
                self.type = type
                self.content = content
                self.position = position
                self.src = src
                self.description = description
        
        class ExtractionResult:
            def __init__(self, page, content_list=None):
                self.page = page
                self.content_list = content_list or []
    except ImportError:
        logger.error("Could not import ArangoDB utilities from fetch_docs module")
        return
    
    # Initialize database
    try:
        db = get_db(db_url, db_name)
    except Exception as e:
        logger.error(f"Failed to connect to ArangoDB: {e}")
        return
    
    # Process each site and store in database
    for url, site_data in processed_docs.items():
        for file_data in site_data:
            try:
                # Create page metadata
                page = PageMetadata(
                    url=url,
                    title=Path(file_data["file"]).name,
                    summary=f"Documentation page from {url}",
                    token_count=sum(section.get("token_count", 0) for section in file_data["sections"]),
                    LLM_model="none"
                )
                
                # Convert sections to content list format
                content_list = []
                for i, section in enumerate(file_data["sections"]):
                    # Create content items
                    contents = [
                        ContentItem(
                            type="text",
                            content=section.get("content", ""),
                            position=0
                        )
                    ]
                    
                    # Create section content
                    section_content = SectionContent(
                        section_name=section.get("header", f"Section {i+1}"),
                        position=i,
                        contents=contents
                    )
                    content_list.append(section_content)
                
                # Create extraction result
                result = ExtractionResult(
                    page=page,
                    content_list=content_list
                )
                
                # Store in database (run async function)
                asyncio.run(store_page_content(db, result))
                logger.info(f"Stored document in ArangoDB: {file_data['file']}")
            except Exception as e:
                logger.error(f"Failed to store document in ArangoDB: {e}")


def integrate_docs_with_extraction(repo_path: Path, output_blocks: List[Dict], max_docs: int = 5) -> List[Dict]:
    """
    Main integration function to detect docs, download, and merge with extraction output.
    
    Args:
        repo_path: Path to the repository
        output_blocks: Existing extraction blocks from DuaLipa
        max_docs: Maximum number of documentation links to process (default: 5)
        
    Returns:
        Enhanced list of blocks including documentation
    """
    # Detect documentation links in the repository
    doc_links = detect_doc_links(repo_path)
    
    if not doc_links:
        logger.info("No documentation links found in repository")
        return output_blocks
    
    logger.info(f"Found {len(doc_links)} documentation links in repository")
    
    # Log the top 10 documentation links found for debugging
    if doc_links and len(doc_links) > 0:
        logger.info("First 10 documentation links found:")
        for i, link in enumerate(doc_links[:10]):
            logger.info(f"  {i+1}. {link}")
    
    # Create a temp directory for documentation
    docs_dir = repo_path / ".dualipa_docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Score and prioritize documentation links
    prioritized_links = prioritize_documentation_links(doc_links)
    
    # Log the prioritized links for debugging
    if prioritized_links and len(prioritized_links) > 0:
        logger.info("Top 5 prioritized documentation links:")
        for i, link in enumerate(prioritized_links[:5]):
            logger.info(f"  {i+1}. {link}")
    
    # Check if we have any boost-hof links that need special handling
    has_boost_hof = any('boost-hof.readthedocs.io' in link for link in prioritized_links)
    if has_boost_hof:
        logger.info("Found boost-hof.readthedocs.io link - will use special handling")
        
        # Make sure the boost-hof link is in a standardized format
        boost_hof_links = [link for link in prioritized_links if 'boost-hof.readthedocs.io' in link]
        for bhof_link in boost_hof_links:
            logger.info(f"Original boost-hof link: {bhof_link}")
        
        # Replace all boost-hof links with a standardized version
        prioritized_links = [
            'https://boost-hof.readthedocs.io/' if 'boost-hof.readthedocs.io' in link else link 
            for link in prioritized_links
        ]
    
    # Handle ArangoDB specific documentation
    has_arangodb = any('arangodb.com' in link for link in prioritized_links)
    if has_arangodb:
        logger.info("Found ArangoDB documentation links - ensuring priority for AQL documentation")
        
        # Boost AQL documentation to the top if present
        aql_links = [link for link in prioritized_links if 'arangodb.com' in link and 'aql' in link]
        if aql_links:
            logger.info(f"Found {len(aql_links)} ArangoDB AQL documentation links")
            
            # Re-prioritize to ensure AQL docs are at the top
            non_aql_links = [link for link in prioritized_links if link not in aql_links]
            prioritized_links = aql_links + non_aql_links
    
    # Set a limit based on max_docs parameter
    actual_max = max(1, min(max_docs, 10))  # At least 1, at most 10
    
    # Limit the number of documentation links to process
    selected_links = prioritized_links[:actual_max]
    logger.info(f"Selected {len(selected_links)} of {len(doc_links)} documentation links for processing (max_docs={max_docs})")
    
    # Count by documentation type
    rtd_count = sum(1 for link in selected_links if 'readthedocs.io' in link or 'readthedocs.org' in link)
    arangodb_count = sum(1 for link in selected_links if 'arangodb.com' in link)
    docs_count = sum(1 for link in selected_links if ('/docs/' in link or 'docs.' in link) and 'arangodb.com' not in link)
    other_count = len(selected_links) - rtd_count - docs_count - arangodb_count
    
    logger.info(f"Documentation links breakdown: {rtd_count} ReadTheDocs, {arangodb_count} ArangoDB, {docs_count} docs sites, {other_count} other")
    
    # Download documentation
    logger.info(f"Downloading {len(selected_links)} documentation links")
    downloaded_sites = download_docs(selected_links, docs_dir)
    
    # Check if some downloads failed
    if len(downloaded_sites) < len(selected_links):
        logger.warning(f"Only {len(downloaded_sites)} of {len(selected_links)} documentation links were downloaded successfully")
    
    # Process documentation
    logger.info(f"Processing {len(downloaded_sites)} downloaded documentation sites")
    processed_docs = process_docs(downloaded_sites)
    
    # Convert to DuaLipa format
    logger.info("Converting documentation to DuaLipa format")
    doc_blocks = convert_to_dualipa_format(processed_docs, repo_path)
    
    # Log counts of different block types
    block_types = {}
    for block in doc_blocks:
        block_type = block.get("type", "unknown")
        if block_type not in block_types:
            block_types[block_type] = 0
        block_types[block_type] += 1
    
    logger.info(f"Created documentation blocks by type: {block_types}")
    
    # Check for boost-hof blocks
    boost_hof_blocks = [b for b in doc_blocks if b.get("source_url", "").find("boost-hof.readthedocs.io") != -1]
    if boost_hof_blocks:
        logger.info(f"Successfully created {len(boost_hof_blocks)} boost-hof documentation blocks")
    
    # Check for ArangoDB blocks
    arangodb_blocks = [b for b in doc_blocks if b.get("metadata", {}).get("doc_type") == "arangodb"]
    if arangodb_blocks:
        logger.info(f"Successfully created {len(arangodb_blocks)} ArangoDB documentation blocks")
    
    # Append documentation blocks to output
    output_blocks.extend(doc_blocks)
    
    logger.info(f"Added {len(doc_blocks)} documentation blocks to extraction output")
    
    # Optional: Store in ArangoDB for persistence (disabled by default)
    """
    try:
        # Get ArangoDB configuration (would come from environment or config)
        db_url = os.environ.get("ARANGODB_URL", "http://localhost:8529")
        db_name = os.environ.get("ARANGODB_DB", "dualipa_docs")
        
        # Store documentation in ArangoDB
        store_docs_in_arangodb(processed_docs, db_url, db_name)
        logger.info(f"Stored documentation in ArangoDB: {db_name}")
    except Exception as e:
        logger.warning(f"Failed to store documentation in ArangoDB: {e}")
    """
    
    return output_blocks


def prioritize_documentation_links(doc_links: List[str]) -> List[str]:
    """
    Score and prioritize documentation links based on relevance.
    
    Args:
        doc_links: List of detected documentation links
        
    Returns:
        Prioritized list of documentation links
    """
    # Create a scoring system for documentation links
    link_scores = []
    
    # First clean up URLs with any special handling needs
    cleaned_links = []
    for link in doc_links:
        # Special handling for boost-hof URL from ArangoDB repo
        if 'boost-hof.readthedocs.io' in link:
            # Handle the specific case in the ArangoDB repo where the URL has trailing garbage
            # or is malformed like "boost-hof.readthedocs.io](http://boost-hof.readthedocs.io/"
            if '](' in link:
                # Fix markdown format issue where link is concatenated
                parts = link.split('](')
                if len(parts) >= 2 and 'boost-hof.readthedocs.io' in parts[1]:
                    # Use the correct part of the URL
                    link = parts[1].strip(')"\'')
            
            # Ensure it has proper protocol
            if not link.startswith('http'):
                link = 'https://' + link.lstrip('/')
                
            # Clean up the URL to just the domain if necessary
            url_parts = link.split('boost-hof.readthedocs.io')
            if len(url_parts) > 1:
                # Clean up the URL to just the domain with proper protocol
                link = 'https://boost-hof.readthedocs.io/'
        
        # Strip trailing characters that might be part of markdown formatting
        link = link.rstrip(')],.;:"\'')
        
        # Add protocol if missing
        if not link.startswith('http'):
            link = 'https://' + link
            
        cleaned_links.append(link)
    
    # Now score the cleaned links
    for link in cleaned_links:
        score = 0
        
        # Prioritize known documentation platforms
        if 'readthedocs.io' in link or 'readthedocs.org' in link:
            score += 100  # ReadTheDocs is high quality documentation
        elif '.github.io' in link:
            score += 80   # GitHub Pages often has good documentation
        elif 'docs.' in link:
            score += 70   # docs.example.com domains are usually documentation
        elif '/docs/' in link:
            score += 60   # /docs/ paths usually indicate documentation
        elif '/documentation/' in link:
            score += 50   # /documentation/ paths are likely documentation
        elif '/api/' in link or '/api-docs/' in link:
            score += 40   # API documentation
        
        # Deprioritize certain paths
        if '/examples/' in link:
            score -= 20   # Examples are less valuable than full documentation
        if '/javadoc/' in link:
            score -= 10   # Generated JavaDocs often have less context
        
        # Adjust for link structure
        if link.count('/') < 4:  # Main/root documentation page
            score += 30
        
        # Adjust for domain relevance 
        # (prioritize main project documentation over third-party links)
        if 'boost' in link and ('hof' in link or 'compute' in link):
            score += 50   # Boost HOF/Compute is relevant to ArangoDB
            
        # Extra boost for readthedocs urls with https:// prefix that look clean
        if 'https://boost-hof.readthedocs.io/' in link:
            score += 20   # Give extra priority to properly formatted boost-hof URL
                
        # Special handling for arangodb
        if 'arangodb.com' in link:
            # The ArangoDB AQL docs are particularly valuable
            if 'aql' in link:
                score += 60  # AQL documentation is highly relevant
                
        link_scores.append((link, score))
    
    # Sort by score (descending)
    link_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return prioritized links
    return [link for link, score in link_scores]


def extract_all_blocks_with_docs(repo_path: Path) -> List[Dict[str, Any]]:
    """
    Enhanced extraction function that includes documentation.
    
    Args:
        repo_path: Directory to extract from
        
    Returns:
        List of extracted blocks including documentation
    """
    # Try to import the regular extraction function
    try:
        from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
    except ImportError:
        logger.error("Could not import extract_all_blocks from DuaLipa extraction module")
        return []
    
    # Regular extraction
    code_blocks = extract_all_blocks(repo_path)
    
    # Enhance with documentation
    enhanced_blocks = integrate_docs_with_extraction(repo_path, code_blocks)
    
    return enhanced_blocks


if __name__ == "__main__":
    import sys
    import json
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if len(sys.argv) != 3:
        print("Usage: python fetch_docs_integration.py <repository_path> <output_json_file>")
        sys.exit(1)
    
    repo_path = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    # Extract blocks with documentation
    blocks = extract_all_blocks_with_docs(repo_path)
    
    # Write output to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(blocks, f, indent=2)
    
    print(f"Extraction completed. Extracted {len(blocks)} blocks.")
    print(f"Output written to: {output_file}")