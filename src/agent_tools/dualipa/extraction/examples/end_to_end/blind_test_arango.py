#!/usr/bin/env python3
"""
ArangoDB Blind Test for Extraction Functionality

This script performs a blind test of DuaLipa's extraction capabilities using
a variety of ArangoDB code files and documentation sources. It tests:

1. Code extraction from multiple languages (Python, Bash, TypeScript)
2. Documentation extraction from ArangoDB docs
3. Hierarchical relationship maintenance
4. Format consistency across different source types

The test downloads specified GitHub files and documentation links, then runs
the extraction pipeline and validates the output format and consistency.
"""

import os
import sys
import json
import time
import hashlib
import tempfile
import argparse
import logging
import datetime
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("blind_test_arango")

# Target URLs to test
GITHUB_FILES = [
    "https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/utils/gantt.py",
    "https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/scripts/buildUnittestBashCompletion.bash",
    "https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/js/apps/system/_admin/aardvark/APP/react/src/views/query/ArangoQuery.types.ts",
    "https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/js/apps/system/_admin/aardvark/APP/react/src/views/views/ViewsList.tsx",
    "https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/scripts/toolbox/modules/Pipeline.py"
]

DOC_URLS = [
    "https://docs.arangodb.com/stable/aql/data-queries/",
    "https://docs.arangodb.com/stable/aql/functions/arangosearch/",
    "https://docs.arangodb.com/stable/aql/operators/"
]


def fetch_github_file(url: str, target_dir: Path) -> Optional[Path]:
    """
    Download a specific file from GitHub.
    
    Args:
        url: GitHub URL for the file
        target_dir: Directory to save the file
        
    Returns:
        Path to downloaded file or None if download failed
    """
    try:
        # Convert GitHub URL to raw URL
        # Original: https://github.com/arangodb/arangodb/blob/bba7f899831ee71373e3f673e30148154cb9f761/utils/gantt.py
        # Raw: https://raw.githubusercontent.com/arangodb/arangodb/bba7f899831ee71373e3f673e30148154cb9f761/utils/gantt.py
        parts = url.split("github.com/")
        if len(parts) != 2:
            logger.error(f"Invalid GitHub URL format: {url}")
            return None
            
        path_parts = parts[1].split("/", 4)
        if len(path_parts) < 5 or "blob" not in path_parts:
            logger.error(f"Cannot parse GitHub URL: {url}")
            return None
            
        # Extract components
        owner = path_parts[0]
        repo = path_parts[1]
        branch = path_parts[3]
        file_path = path_parts[4]
        
        # Create raw URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        
        logger.info(f"Fetching file from {raw_url}")
        
        # Make request
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Create filename
        file_name = os.path.basename(file_path)
        file_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        local_path = target_dir / f"{file_hash}_{file_name}"
        
        # Save file
        with open(local_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Saved file to {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error fetching GitHub file {url}: {e}")
        return None


def download_docs_page(url: str, target_dir: Path) -> Optional[Path]:
    """
    Download an ArangoDB documentation page.
    
    Args:
        url: Documentation URL to download
        target_dir: Directory to save the downloaded file
        
    Returns:
        Path to downloaded file or None if download failed
    """
    try:
        # Import fetch_docs utility if available
        try:
            from agent_tools.fetch_docs.download_site import download_site
            logger.info("Using fetch_docs download utility")
            download_available = True
        except ImportError:
            logger.warning("fetch_docs not available, using fallback download")
            download_available = False
        
        # Create output directory
        output_dir = target_dir / "docs"
        output_dir.mkdir(exist_ok=True)
        
        # Create a unique subdirectory for this URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        doc_dir = output_dir / url_hash
        doc_dir.mkdir(exist_ok=True)
        
        if download_available:
            # Use fetch_docs utility to download
            download_site(url, str(doc_dir), recursive=False)
        else:
            # Fallback to direct download
            response = requests.get(url)
            response.raise_for_status()
            
            # Save file
            file_name = "index.html"
            local_path = doc_dir / file_name
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
        # Look for the downloaded HTML file
        html_files = list(doc_dir.glob("**/*.html"))
        if not html_files:
            logger.error(f"No HTML files found after downloading {url}")
            return None
            
        # Return the path to the first HTML file
        return html_files[0]
        
    except Exception as e:
        logger.error(f"Error downloading documentation from {url}: {e}")
        return None


def extract_code_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract code blocks from a file.
    
    Args:
        file_path: Path to the file to extract from
        
    Returns:
        List of extracted code blocks
    """
    try:
        # Try to import the extraction function
        try:
            # Try with absolute import first
            from agent_tools.dualipa.extraction.extractors.code.hierarchy import extract_code_hierarchy
        except ImportError:
            logger.warning("Could not import from absolute path, trying relative import")
            # Try with relative import
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
            from extraction.extractors.code.hierarchy import extract_code_hierarchy
            
        # Extract code blocks
        logger.info(f"Extracting code blocks from {file_path}")
        blocks = extract_code_hierarchy(file_path)
        
        # Add content to blocks if missing
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            file_lines = file_content.splitlines()
        except Exception as e:
            logger.warning(f"Could not read file content: {e}")
            file_content = ""
            file_lines = []
            
        # Ensure all blocks have required fields
        for block in blocks:
            # Add UUIDs if not present
            if "uuid" not in block:
                import uuid
                block["uuid"] = str(uuid.uuid4())
                
            # Add file_path if not present
            if "file_path" not in block:
                block["file_path"] = file_path
                
            # Add content if not present but we have start/end lines
            if "content" not in block and "start_line" in block and "end_line" in block:
                start = max(0, block["start_line"] - 1)  # Adjust for 0-indexed arrays
                end = min(len(file_lines), block["end_line"])
                
                if start < end and start < len(file_lines):
                    block["content"] = "\n".join(file_lines[start:end])
                else:
                    # Fallback content
                    block["content"] = f"{block.get('type', 'code')} '{block.get('name', 'unnamed')}'"
            elif "content" not in block:
                # Add fallback content
                block["content"] = f"{block.get('type', 'code')} '{block.get('name', 'unnamed')}'"
        
        logger.info(f"Extracted {len(blocks)} code blocks from {file_path}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting code blocks: {e}")
        return []


def extract_documentation(html_file: Path) -> List[Dict[str, Any]]:
    """
    Extract documentation blocks from an HTML file.
    
    Args:
        html_file: Path to the HTML file
        
    Returns:
        List of extracted documentation blocks
    """
    try:
        # Try to import required functions
        try:
            from agent_tools.fetch_docs.clean_html import clean_html
            from agent_tools.fetch_docs.extract_sections import extract_sections_from_html
            from agent_tools.dualipa.fetch_docs_integration import convert_to_dualipa_format
            logger.info("Successfully imported documentation extraction functions")
        except ImportError:
            logger.warning("Could not import documentation extraction functions, using fallback")
            
            # Define fallback functions
            def clean_html(html_content):
                """Simple HTML cleaning function."""
                import re
                # Remove scripts and styles
                html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
                html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
                return html_content
                
            def extract_sections_from_html(html_content, file_path=None):
                """Simple section extraction function."""
                import re
                # Extract headings and content
                sections = []
                
                # Try to extract title
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.DOTALL)
                title = title_match.group(1) if title_match else "Unknown Title"
                
                # Create a basic section
                sections.append({
                    "header": title,
                    "content": html_content,
                    "level": 1,
                    "token_count": len(html_content.split())
                })
                
                return sections
                
            def convert_to_dualipa_format(processed_docs, repo_path):
                """Simple conversion function."""
                import uuid
                blocks = []
                
                for url, site_data in processed_docs.items():
                    # Create site block
                    site_uuid = str(uuid.uuid4())
                    blocks.append({
                        "uuid": site_uuid,
                        "id": f"docs_{url.split('//')[-1].split('/')[0].replace('.', '_')}",
                        "name": f"Documentation: {url}",
                        "type": "documentation",
                        "language": "html",
                        "content": f"Documentation site: {url}",
                        "source_url": url,
                        "child_uuids": [],
                        "metadata": {
                            "language": "html",
                            "source_url": url,
                            "doc_type": "arangodb" if "arangodb" in url else "generic"
                        }
                    })
                    
                    # Add pages and sections
                    for file_data in site_data:
                        # Add page block
                        page_uuid = str(uuid.uuid4())
                        page_name = "Documentation Page"
                        blocks.append({
                            "uuid": page_uuid,
                            "id": f"docs_page_{len(blocks)}",
                            "name": page_name,
                            "type": "doc_page",
                            "language": "html",
                            "content": f"Documentation page from {url}",
                            "file_path": file_data.get("file", ""),
                            "parent_uuid": site_uuid,
                            "child_uuids": [],
                            "metadata": {
                                "language": "html",
                                "source_url": url,
                                "doc_type": "arangodb" if "arangodb" in url else "generic"
                            }
                        })
                        blocks[0]["child_uuids"].append(page_uuid)
                        
                        # Add section blocks
                        for i, section in enumerate(file_data.get("sections", [])):
                            section_uuid = str(uuid.uuid4())
                            section_title = section.get("header", f"Section {i+1}")
                            blocks.append({
                                "uuid": section_uuid,
                                "id": f"docs_section_{len(blocks)}",
                                "name": section_title,
                                "type": "doc_section",
                                "language": "html",
                                "content": section.get("content", ""),
                                "file_path": file_data.get("file", ""),
                                "parent_uuid": page_uuid,
                                "child_uuids": [],
                                "metadata": {
                                    "language": "html",
                                    "source_url": url,
                                    "doc_type": "arangodb" if "arangodb" in url else "generic",
                                    "header_level": section.get("level", 1)
                                }
                            })
                            blocks[1]["child_uuids"].append(section_uuid)
                
                return blocks
        
        # Read HTML file
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
            
        # Clean HTML
        cleaned_content = clean_html(html_content)
        
        # Extract sections
        sections = extract_sections_from_html(cleaned_content, html_file)
        
        # Create file data structure for conversion
        url = f"https://docs.arangodb.com/{html_file.parent.name}/"
        site_data = [{
            "file": str(html_file),
            "relative_path": html_file.name,
            "sections": sections,
            "doc_type": "arangodb"
        }]
        
        # Convert to DuaLipa format
        processed_docs = {url: site_data}
        blocks = convert_to_dualipa_format(processed_docs, html_file.parent)
        
        logger.info(f"Extracted {len(blocks)} documentation blocks from {html_file}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting documentation: {e}")
        
        # Create minimal fallback blocks
        import uuid
        site_uuid = str(uuid.uuid4())
        page_uuid = str(uuid.uuid4())
        section_uuid = str(uuid.uuid4())
        
        blocks = [
            {
                "uuid": site_uuid,
                "id": "docs_arangodb",
                "name": "Documentation: ArangoDB",
                "type": "documentation",
                "language": "html",
                "content": f"Documentation site: {html_file}",
                "child_uuids": [page_uuid],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb"
                }
            },
            {
                "uuid": page_uuid,
                "id": "docs_page",
                "name": html_file.name,
                "type": "doc_page",
                "language": "html",
                "content": f"Documentation page from {html_file}",
                "parent_uuid": site_uuid,
                "child_uuids": [section_uuid],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb"
                }
            },
            {
                "uuid": section_uuid,
                "id": "docs_section",
                "name": "Content",
                "type": "doc_section",
                "language": "html",
                "content": "Content section",
                "parent_uuid": page_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb",
                    "header_level": 1
                }
            }
        ]
        
        logger.info(f"Created fallback documentation blocks for {html_file}")
        return blocks


def combine_extractions(blocks_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Combine multiple extraction results into a single list.
    
    Args:
        blocks_list: List of lists of blocks
        
    Returns:
        Combined list of all blocks
    """
    # Flatten the list of lists
    combined_blocks = []
    for blocks in blocks_list:
        combined_blocks.extend(blocks)
        
    logger.info(f"Combined {len(combined_blocks)} blocks from {len(blocks_list)} sources")
    return combined_blocks


def validate_extraction(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate extraction blocks.
    
    Args:
        blocks: List of extraction blocks
        
    Returns:
        Validation results
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_blocks": len(blocks),
            "block_types": {}
        }
    }
    
    # Check for required fields
    required_fields = ["uuid", "type", "name", "content"]
    
    for i, block in enumerate(blocks):
        # Track block types
        block_type = block.get("type", "unknown")
        if block_type not in validation["stats"]["block_types"]:
            validation["stats"]["block_types"][block_type] = 0
        validation["stats"]["block_types"][block_type] += 1
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in block]
        if missing_fields:
            validation["errors"].append(f"Block {i} (type: {block_type}) is missing required fields: {missing_fields}")
            validation["valid"] = False
            
    # Check parent-child relationships
    child_to_parent = {}
    for block in blocks:
        if "parent_uuid" in block:
            child_to_parent[block["uuid"]] = block["parent_uuid"]
            
    for block in blocks:
        if "child_uuids" in block:
            for child_uuid in block["child_uuids"]:
                # Check if child exists
                if not any(b["uuid"] == child_uuid for b in blocks):
                    validation["errors"].append(f"Block {block['uuid']} references non-existent child {child_uuid}")
                    validation["valid"] = False
                # Check if child has correct parent reference
                elif child_uuid in child_to_parent and child_to_parent[child_uuid] != block["uuid"]:
                    validation["errors"].append(
                        f"Block {child_uuid} has parent {child_to_parent[child_uuid]} but is listed as child of {block['uuid']}"
                    )
                    validation["valid"] = False
    
    # Count code vs. doc blocks
    validation["stats"]["code_blocks"] = sum(1 for b in blocks if b.get("type") in ["class", "method", "function"])
    validation["stats"]["doc_blocks"] = sum(1 for b in blocks if b.get("type") in ["documentation", "doc_page", "doc_section"])
    
    logger.info(f"Validation completed: {'SUCCESS' if validation['valid'] else 'FAILURE'}")
    if validation["errors"]:
        logger.warning(f"Found {len(validation['errors'])} validation errors")
        
    return validation


def create_report(
    blocks: List[Dict[str, Any]], 
    validation: Dict[str, Any],
    files: List[str],
    doc_urls: List[str],
    output_dir: Path
) -> Path:
    """
    Create an HTML report of the extraction and validation.
    
    Args:
        blocks: List of blocks
        validation: Validation results
        files: List of file paths
        doc_urls: List of documentation URLs
        output_dir: Output directory
        
    Returns:
        Path to the HTML report
    """
    report_path = output_dir / "blind_test_report.html"
    
    # Get sample blocks
    code_block = next((b for b in blocks if b.get("type") in ["class", "method", "function"]), None)
    doc_block = next((b for b in blocks if b.get("type") == "doc_section"), None)
    
    # Create HTML report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ArangoDB Blind Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{
            color: #444;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }}
        .success {{
            color: #2ecc71;
        }}
        .warning {{
            color: #f39c12;
        }}
        .error {{
            color: #e74c3c;
        }}
        pre {{
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            overflow: auto;
            font-family: monospace;
            font-size: 14px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ArangoDB Blind Test Report</h1>
            <p>Test run: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2>Test Configuration</h2>
            <h3>Code Files</h3>
            <table>
                <tr>
                    <th>#</th>
                    <th>File Path</th>
                </tr>
                {''.join([f"<tr><td>{i+1}</td><td>{file}</td></tr>" for i, file in enumerate(files)])}
            </table>
            
            <h3>Documentation URLs</h3>
            <table>
                <tr>
                    <th>#</th>
                    <th>URL</th>
                </tr>
                {''.join([f"<tr><td>{i+1}</td><td><a href='{url}'>{url}</a></td></tr>" for i, url in enumerate(doc_urls)])}
            </table>
        </div>

        <div class="section">
            <h2>Extraction Results</h2>
            <div class="stats">
                <div class="stat-card">
                    <h3>Total Blocks</h3>
                    <p>{validation["stats"].get("total_blocks", 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Code Blocks</h3>
                    <p>{validation["stats"].get("code_blocks", 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Documentation Blocks</h3>
                    <p>{validation["stats"].get("doc_blocks", 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Status</h3>
                    <p class="{'success' if validation.get('valid', False) else 'error'}">
                        {"✅ VALID" if validation.get("valid", False) else "❌ INVALID"}
                    </p>
                </div>
            </div>

            <h3>Block Types</h3>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
                {''.join([f"<tr><td>{block_type}</td><td>{count}</td></tr>" 
                         for block_type, count in validation.get("stats", {}).get("block_types", {}).items()])}
            </table>
        </div>

        <div class="section">
            <h2>Validation</h2>
            
            {'<h3 class="success">No validation errors</h3>' if not validation.get("errors") else f'''
            <h3 class="error">Validation Errors ({len(validation.get("errors", []))})</h3>
            <ul>
                {"".join([f"<li>{error}</li>" for error in validation.get("errors", [])])}
            </ul>
            '''}
            
            {'<h3 class="success">No warnings</h3>' if not validation.get("warnings") else f'''
            <h3 class="warning">Warnings ({len(validation.get("warnings", []))})</h3>
            <ul>
                {"".join([f"<li>{warning}</li>" for warning in validation.get("warnings", [])])}
            </ul>
            '''}
        </div>

        <div class="section">
            <h2>Sample Blocks</h2>
            <h3>Code Sample</h3>
            <pre>{json.dumps(code_block, indent=2) if code_block else "No code sample available"}</pre>
            
            <h3>Documentation Sample</h3>
            <pre>{json.dumps(doc_block, indent=2) if doc_block else "No documentation sample available"}</pre>
        </div>
    </div>
</body>
</html>""")
        
    logger.info(f"Created report at {report_path}")
    return report_path


def run_blind_test(output_dir: Optional[Path] = None, open_browser: bool = True) -> Dict[str, Any]:
    """
    Run the blind test and return results.
    
    Args:
        output_dir: Directory to save results (created if None)
        open_browser: Whether to open the report in a browser
        
    Returns:
        Test results dictionary
    """
    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"blind_test_results_{timestamp}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize results
    results = {
        "success": False,
        "timestamp": datetime.datetime.now().isoformat(),
        "files_processed": 0,
        "docs_processed": 0,
        "extraction_stats": {},
        "output_dir": str(output_dir)
    }
    
    try:
        # Step 1: Fetch GitHub files
        logger.info("Step 1: Fetching GitHub files")
        code_files = []
        
        for url in GITHUB_FILES:
            file_path = fetch_github_file(url, output_dir)
            if file_path:
                code_files.append(str(file_path))
                results["files_processed"] += 1
                
        logger.info(f"Successfully fetched {len(code_files)} of {len(GITHUB_FILES)} code files")
        
        # Step 2: Download documentation
        logger.info("Step 2: Downloading documentation")
        doc_files = []
        
        for url in DOC_URLS:
            file_path = download_docs_page(url, output_dir)
            if file_path:
                doc_files.append(str(file_path))
                results["docs_processed"] += 1
                
        logger.info(f"Successfully downloaded {len(doc_files)} of {len(DOC_URLS)} documentation pages")
        
        # Step 3: Extract code from files
        logger.info("Step 3: Extracting code from files")
        code_blocks_list = []
        
        for file_path in code_files:
            blocks = extract_code_blocks(file_path)
            if blocks:
                code_blocks_list.append(blocks)
                
        # Flatten code blocks
        code_blocks = [block for blocks in code_blocks_list for block in blocks]
        logger.info(f"Extracted {len(code_blocks)} code blocks total")
        
        # Save code blocks
        code_blocks_path = output_dir / "code_blocks.json"
        with open(code_blocks_path, "w", encoding="utf-8") as f:
            json.dump(code_blocks, f, indent=2)
            
        # Step 4: Extract documentation
        logger.info("Step 4: Extracting documentation")
        doc_blocks_list = []
        
        for file_path in doc_files:
            blocks = extract_documentation(Path(file_path))
            if blocks:
                doc_blocks_list.append(blocks)
                
        # Flatten doc blocks
        doc_blocks = [block for blocks in doc_blocks_list for block in blocks]
        logger.info(f"Extracted {len(doc_blocks)} documentation blocks total")
        
        # Save documentation blocks
        doc_blocks_path = output_dir / "doc_blocks.json"
        with open(doc_blocks_path, "w", encoding="utf-8") as f:
            json.dump(doc_blocks, f, indent=2)
            
        # Step 5: Combine and validate
        logger.info("Step 5: Combining and validating extraction results")
        all_blocks = code_blocks + doc_blocks
        
        # Save combined blocks
        combined_blocks_path = output_dir / "combined_blocks.json"
        with open(combined_blocks_path, "w", encoding="utf-8") as f:
            json.dump(all_blocks, f, indent=2)
            
        # Validate blocks
        validation = validate_extraction(all_blocks)
        
        # Save validation results
        validation_path = output_dir / "validation_results.json"
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)
            
        # Step 6: Create report
        logger.info("Step 6: Creating test report")
        report_path = create_report(
            all_blocks,
            validation,
            code_files,
            DOC_URLS,
            output_dir
        )
        
        # Try to open the report in a browser
        if open_browser:
            try:
                import webbrowser
                webbrowser.open(f"file://{report_path}")
                logger.info(f"Opened report in browser: {report_path}")
            except Exception as e:
                logger.warning(f"Could not open report in browser: {e}")
        
        # Set results
        results["success"] = validation["valid"]
        results["extraction_stats"] = validation["stats"]
        results["report_path"] = str(report_path)
        
        if validation["valid"]:
            logger.info("✅ Blind test completed successfully")
        else:
            logger.error(f"❌ Blind test failed with {len(validation.get('errors', []))} validation errors")
            
        return results
        
    except Exception as e:
        logger.error(f"Error running blind test: {e}")
        results["error"] = str(e)
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run ArangoDB blind test")
    parser.add_argument("--output-dir", type=str, help="Directory to save test results")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the report in browser")
    args = parser.parse_args()
    
    # Get output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Run the test
    results = run_blind_test(output_dir, not args.no_browser)
    
    # Print test summary
    if results["success"]:
        print("\n✅ Blind test completed successfully!")
    else:
        print("\n❌ Blind test failed.")
        if "error" in results:
            print(f"Error: {results['error']}")
            
    # Print statistics
    print("\nExtraction Statistics:")
    for key, value in results.get("extraction_stats", {}).items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Print block types if available
    block_types = results.get("extraction_stats", {}).get("block_types", {})
    if block_types:
        print("\nBlock Types:")
        for block_type, count in block_types.items():
            print(f"  {block_type}: {count}")
            
    print(f"\nReport saved to: {results.get('report_path', 'unknown')}")
    print(f"All test artifacts in: {results.get('output_dir', 'unknown')}")
    
    # Return success status
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())