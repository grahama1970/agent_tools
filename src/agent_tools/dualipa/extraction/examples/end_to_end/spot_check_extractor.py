#!/usr/bin/env python3
"""
spot_check_extractor.py

A tool for spot-checking DuaLipa extraction on arbitrary files or URLs.
This allows quick verification of extraction results across different file types.

Usage:
    python spot_check_extractor.py --file /path/to/file.py
    python spot_check_extractor.py --url https://docs.arangodb.com/stable/aql/
    python spot_check_extractor.py --github arangodb/arangodb/blob/main/js/apps/system/_admin/aardvark/APP/react/src/views/databases/DatabasesContext.tsx

This will extract the content and print the resulting JSON blocks.
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import logging
import uuid
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import mimetypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spot_check_extractor")

# Try to import extraction modules, with graceful fallbacks
try:
    # Try to import from main package
    from agent_tools.dualipa.extraction.code_extraction import extract_code
    from agent_tools.dualipa.extraction.html_extraction import extract_html
    from agent_tools.dualipa.extraction.markdown_extraction import extract_markdown
    EXTRACTION_AVAILABLE = True
except ImportError:
    logger.warning("Could not import extraction modules directly, trying relative imports")
    try:
        # Try relative imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from extraction.code_extraction import extract_code
        from extraction.html_extraction import extract_html
        from extraction.markdown_extraction import extract_markdown
        EXTRACTION_AVAILABLE = True
    except ImportError:
        logger.warning("Extraction modules not available. Using mock implementations for demo.")
        EXTRACTION_AVAILABLE = False


def mock_extract_code(file_path: str, language: str = None) -> List[Dict[str, Any]]:
    """Mock implementation of code extraction for demonstration."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Create a mock extraction with basic structure
    code_uuid = str(uuid.uuid4())
    
    return [{
        "uuid": code_uuid,
        "type": "code_block",
        "name": f"Mock {language or 'code'} extraction",
        "content": content[:1000] + ("..." if len(content) > 1000 else ""),
        "language": language or "unknown",
        "metadata": {
            "file_path": file_path,
            "language": language or "unknown",
            "extraction_method": "mock",
            "mock_note": "This is a mock extraction for demonstration purposes only."
        }
    }]


def mock_extract_html(url: str, html_content: str) -> List[Dict[str, Any]]:
    """Mock implementation of HTML extraction for demonstration."""
    doc_uuid = str(uuid.uuid4())
    page_uuid = str(uuid.uuid4())
    section_uuid = str(uuid.uuid4())
    
    return [
        {
            "uuid": doc_uuid,
            "type": "documentation",
            "name": f"Documentation: {url}",
            "content": f"Documentation site: {url}",
            "language": "html",
            "source_url": url,
            "child_uuids": [page_uuid],
            "metadata": {
                "language": "html",
                "source_url": url,
                "doc_type": "html",
                "extraction_method": "mock"
            }
        },
        {
            "uuid": page_uuid,
            "type": "doc_page",
            "name": "Mock Page",
            "content": f"Documentation page from {url}",
            "language": "html",
            "parent_uuid": doc_uuid,
            "child_uuids": [section_uuid],
            "metadata": {
                "language": "html",
                "source_url": url,
                "doc_type": "html",
                "extraction_method": "mock"
            }
        },
        {
            "uuid": section_uuid,
            "type": "doc_section",
            "name": "Mock Section",
            "content": html_content[:1000] + ("..." if len(html_content) > 1000 else ""),
            "language": "html",
            "parent_uuid": page_uuid,
            "child_uuids": [],
            "metadata": {
                "language": "html",
                "source_url": url,
                "header_level": 1,
                "extraction_method": "mock"
            }
        }
    ]


def mock_extract_markdown(file_path: str) -> List[Dict[str, Any]]:
    """Mock implementation of markdown extraction for demonstration."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    doc_uuid = str(uuid.uuid4())
    section_uuid = str(uuid.uuid4())
    
    return [
        {
            "uuid": doc_uuid,
            "type": "documentation",
            "name": f"Documentation: {Path(file_path).name}",
            "content": f"Markdown document: {file_path}",
            "language": "markdown",
            "file_path": file_path,
            "child_uuids": [section_uuid],
            "metadata": {
                "language": "markdown",
                "file_path": file_path,
                "extraction_method": "mock"
            }
        },
        {
            "uuid": section_uuid,
            "type": "doc_section",
            "name": "Mock Markdown Section",
            "content": content[:1000] + ("..." if len(content) > 1000 else ""),
            "language": "markdown",
            "parent_uuid": doc_uuid,
            "child_uuids": [],
            "metadata": {
                "language": "markdown",
                "file_path": file_path,
                "header_level": 1,
                "extraction_method": "mock"
            }
        }
    ]


def detect_language_from_file(file_path: str) -> str:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected language string
    """
    ext = Path(file_path).suffix.lower()
    ext_to_language = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.jsx': 'javascript',
        '.html': 'html',
        '.htm': 'html',
        '.md': 'markdown',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.json': 'json',
        '.xml': 'xml',
        '.css': 'css',
        '.scss': 'scss',
        '.sql': 'sql',
    }
    return ext_to_language.get(ext, 'unknown')


def detect_content_type(file_path: str) -> str:
    """
    Detect the content type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Content type: "code", "html", or "markdown"
    """
    language = detect_language_from_file(file_path)
    
    if language in ['html']:
        return "html"
    elif language in ['markdown', 'md']:
        return "markdown"
    else:
        return "code"


def download_url(url: str) -> Tuple[str, str]:
    """
    Download content from a URL.
    
    Args:
        url: URL to download
        
    Returns:
        Tuple of (content, content_type)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Try to determine content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'text/html' in content_type:
            return response.text, 'html'
        elif 'text/markdown' in content_type or url.endswith('.md'):
            return response.text, 'markdown'
        else:
            # Default to code for other content types
            return response.text, 'code'
    
    except Exception as e:
        logger.error(f"Error downloading URL {url}: {e}")
        return "", "unknown"


def download_github_file(github_path: str) -> Tuple[str, str, str]:
    """
    Download a file from GitHub.
    
    Args:
        github_path: GitHub path in format owner/repo/blob/branch/path/to/file
        
    Returns:
        Tuple of (content, file_path, language)
    """
    try:
        # Convert GitHub URL to raw content URL
        parts = github_path.strip('/').split('/')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid GitHub path: {github_path}")
        
        owner = parts[0]
        repo = parts[1]
        
        # Check if 'blob' is in the path
        if 'blob' in parts:
            blob_index = parts.index('blob')
            branch = parts[blob_index + 1]
            file_path = '/'.join(parts[blob_index + 2:])
        else:
            # Assume default branch (main or master) and the rest is the file path
            branch = 'main'
            file_path = '/'.join(parts[3:])
        
        # Create raw URL
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        
        # Download the content
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        
        # Detect language from file extension
        language = detect_language_from_file(file_path)
        
        return response.text, file_path, language
    
    except Exception as e:
        logger.error(f"Error downloading GitHub file {github_path}: {e}")
        return "", "", "unknown"


def extract_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract content from a local file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of extraction blocks
    """
    try:
        content_type = detect_content_type(file_path)
        language = detect_language_from_file(file_path)
        
        if not EXTRACTION_AVAILABLE:
            # Use mock implementations
            if content_type == "html":
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    html_content = f.read()
                return mock_extract_html(f"file://{file_path}", html_content)
            elif content_type == "markdown":
                return mock_extract_markdown(file_path)
            else:
                return mock_extract_code(file_path, language)
        
        # Use real implementations
        if content_type == "html":
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Create temporary directory structure for HTML extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_html = Path(temp_dir) / "index.html"
                with open(temp_html, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Extract HTML
                return extract_html(str(temp_html), f"file://{file_path}")
                
        elif content_type == "markdown":
            return extract_markdown(file_path)
        else:
            return extract_code(file_path, language)
    
    except Exception as e:
        logger.error(f"Error extracting file {file_path}: {e}")
        return [{
            "uuid": str(uuid.uuid4()),
            "type": "error",
            "name": "Extraction Error",
            "content": f"Error extracting file {file_path}: {str(e)}",
            "language": "text",
            "metadata": {
                "file_path": file_path,
                "error": str(e)
            }
        }]


def extract_from_url(url: str) -> List[Dict[str, Any]]:
    """
    Extract content from a URL.
    
    Args:
        url: URL to extract
        
    Returns:
        List of extraction blocks
    """
    try:
        content, content_type = download_url(url)
        
        if not content:
            return [{
                "uuid": str(uuid.uuid4()),
                "type": "error",
                "name": "Download Error",
                "content": f"Could not download content from {url}",
                "metadata": {
                    "url": url
                }
            }]
        
        if not EXTRACTION_AVAILABLE:
            # Use mock implementations
            if content_type == "html":
                return mock_extract_html(url, content)
            elif content_type == "markdown":
                # Create a temporary file for markdown
                with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                try:
                    return mock_extract_markdown(temp_path)
                finally:
                    os.unlink(temp_path)
            else:
                # Create a temporary file for code
                with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                try:
                    return mock_extract_code(temp_path, 'unknown')
                finally:
                    os.unlink(temp_path)
        
        # Use real implementations
        if content_type == "html":
            # Create temporary directory structure for HTML extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_html = Path(temp_dir) / "index.html"
                with open(temp_html, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Extract HTML
                return extract_html(str(temp_html), url)
                
        elif content_type == "markdown":
            # Create a temporary file for markdown
            with tempfile.NamedTemporaryFile(suffix='.md', mode='w', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                return extract_markdown(temp_path)
            finally:
                os.unlink(temp_path)
        else:
            # Create a temporary file for code
            with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                return extract_code(temp_path, 'unknown')
            finally:
                os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error extracting URL {url}: {e}")
        return [{
            "uuid": str(uuid.uuid4()),
            "type": "error",
            "name": "Extraction Error",
            "content": f"Error extracting URL {url}: {str(e)}",
            "language": "text",
            "metadata": {
                "url": url,
                "error": str(e)
            }
        }]


def extract_from_github(github_path: str) -> List[Dict[str, Any]]:
    """
    Extract content from a GitHub file.
    
    Args:
        github_path: GitHub path in format owner/repo/blob/branch/path/to/file
        
    Returns:
        List of extraction blocks
    """
    try:
        content, file_path, language = download_github_file(github_path)
        
        if not content:
            return [{
                "uuid": str(uuid.uuid4()),
                "type": "error",
                "name": "Download Error",
                "content": f"Could not download content from GitHub: {github_path}",
                "metadata": {
                    "github_path": github_path
                }
            }]
        
        # Create a temporary file
        suffix = Path(file_path).suffix or '.txt'
        with tempfile.NamedTemporaryFile(suffix=suffix, mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            if not EXTRACTION_AVAILABLE:
                # Use mock implementations
                content_type = detect_content_type(file_path)
                
                if content_type == "html":
                    return mock_extract_html(f"https://github.com/{github_path}", content)
                elif content_type == "markdown":
                    return mock_extract_markdown(temp_path)
                else:
                    return mock_extract_code(temp_path, language)
            
            # Use real implementations
            content_type = detect_content_type(file_path)
            
            if content_type == "html":
                # Create temporary directory structure for HTML extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_html = Path(temp_dir) / "index.html"
                    with open(temp_html, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    # Extract HTML
                    return extract_html(str(temp_html), f"https://github.com/{github_path}")
                    
            elif content_type == "markdown":
                return extract_markdown(temp_path)
            else:
                return extract_code(temp_path, language)
        
        finally:
            os.unlink(temp_path)
    
    except Exception as e:
        logger.error(f"Error extracting GitHub file {github_path}: {e}")
        return [{
            "uuid": str(uuid.uuid4()),
            "type": "error",
            "name": "Extraction Error",
            "content": f"Error extracting GitHub file {github_path}: {str(e)}",
            "language": "text",
            "metadata": {
                "github_path": github_path,
                "error": str(e)
            }
        }]


def highlight_docstrings(blocks: List[Dict[str, Any]]) -> None:
    """Update blocks dictionary to highlight docstrings for visualization.
    
    Args:
        blocks: List of extraction blocks
    """
    for block in blocks:
        # Skip if no docstring field
        if "doc_string" not in block:
            block["doc_string"] = "No documentation provided"
            if "metadata" in block:
                block["metadata"]["has_docstring"] = False

def generate_html_report(source: str, blocks: List[Dict[str, Any]], output_dir: Optional[str] = None) -> str:
    """
    Generate an HTML report for the extracted blocks.
    
    Args:
        source: Source identifier (file path, URL, or GitHub path)
        blocks: List of extraction blocks
        output_dir: Optional directory to save the report
        
    Returns:
        Path to the HTML report
    """
    # Create HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Report: {source}</title>
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
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #b0c4de;
        }}
        .section {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }}
        .block {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border: 1px solid #ddd;
        }}
        .block-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .block-badge {{
            background-color: #e0e0e0;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow: auto;
            font-size: 0.9em;
        }}
        .metadata {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .metadata-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .block-relationships {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .tree-view {{
            margin-top: 20px;
        }}
        .tree-node {{
            margin-left: 20px;
        }}
        .tree-root {{
            margin-left: 0;
        }}
        .tree-label {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            margin: 2px 0;
            cursor: pointer;
        }}
        .tree-label:hover {{
            background-color: #e0e0e0;
        }}
        .count-badge {{
            background-color: #007bff;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-left: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .docstring-section {{
            margin: 10px 0 15px 0;
            padding: 10px;
            border-radius: 5px;
        }}
        .has-docstring {{
            background-color: #e6f5e6;
            border: 1px solid #c3e6c3;
        }}
        .no-docstring {{
            background-color: #f5f5f5;
            border: 1px solid #e0e0e0;
            color: #777;
        }}
        .docstring-header {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #444;
            border-bottom: 1px solid #ddd;
            padding-bottom: 3px;
        }}
        .docstring-content {{
            white-space: pre-wrap;
            font-family: inherit;
        }}
        .docstring-pre {{
            background-color: transparent;
            padding: 0;
            margin: 0;
            white-space: pre-wrap;
            font-family: inherit;
            overflow: visible;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Extraction Report</h1>
            <p>Source: <strong>{source}</strong></p>
            <p>Total blocks: <strong>{len(blocks)}</strong></p>
        </div>

        <div class="section">
            <h2>Block Type Summary</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
"""
    
    # Count block types
    block_types = {}
    for block in blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
    
    # Add block type summary
    for block_type, count in sorted(block_types.items()):
        html += f"""
                <tr>
                    <td>{block_type}</td>
                    <td>{count}</td>
                </tr>"""
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Block Hierarchy</h2>
            <div class="tree-view">
"""
    
    # Build a map of parent to children
    parent_to_children = {}
    uuid_to_block = {}
    
    for block in blocks:
        uuid = block.get("uuid")
        if uuid:
            uuid_to_block[uuid] = block
    
    for block in blocks:
        uuid = block.get("uuid")
        parent_uuid = block.get("parent_uuid")
        
        if parent_uuid:
            if parent_uuid not in parent_to_children:
                parent_to_children[parent_uuid] = []
            parent_to_children[parent_uuid].append(uuid)
    
    # Find root blocks (no parent or of type "documentation")
    root_blocks = []
    for block in blocks:
        if block.get("type") == "documentation" or "parent_uuid" not in block:
            root_blocks.append(block)
    
    # Function to recursively build tree HTML
    def build_tree_html(block, level=0):
        uuid = block.get("uuid")
        block_type = block.get("type", "unknown")
        name = block.get("name", "Unnamed Block")
        child_count = len(parent_to_children.get(uuid, []))
        
        tree_html = f"""
                <div class="tree-node {'tree-root' if level == 0 else ''}">
                    <div class="tree-label" onclick="document.getElementById('block-{uuid}').scrollIntoView({{behavior: 'smooth'}})">
                        {name} ({block_type}) {f'<span class="count-badge">{child_count}</span>' if child_count > 0 else ''}
                    </div>
"""
        
        # Add children
        for child_uuid in parent_to_children.get(uuid, []):
            child_block = uuid_to_block.get(child_uuid)
            if child_block:
                tree_html += build_tree_html(child_block, level + 1)
        
        tree_html += """
                </div>"""
        return tree_html
    
    # Add tree for each root block
    for root_block in root_blocks:
        html += build_tree_html(root_block)
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>Extraction Blocks</h2>
"""
    
    # Add each block
    for block in blocks:
        uuid = block.get("uuid", "no-uuid")
        block_type = block.get("type", "unknown")
        name = block.get("name", "Unnamed Block")
        content = block.get("content", "")
        language = block.get("language", "unknown")
        metadata = block.get("metadata", {})
        parent_uuid = block.get("parent_uuid", "")
        child_uuids = block.get("child_uuids", [])
        
        # Get docstring
        docstring = block.get("doc_string", "No documentation provided")
        has_docstring = block.get("metadata", {}).get("has_docstring", False)
        docstring_class = "has-docstring" if has_docstring else "no-docstring"
        
        # Format docstring section with better styling
        docstring_html = f"""
        <div class="docstring-section {docstring_class}">
            <div class="docstring-header">Documentation:</div>
            <div class="docstring-content"><pre class="docstring-pre">{docstring}</pre></div>
        </div>
        """
        
        # Format content based on type
        content_html = ""
        if block_type == "table" and isinstance(content, list):
            # Format as table
            content_html = "<table>"
            for i, row in enumerate(content):
                content_html += "<tr>"
                for cell in row:
                    tag = "th" if i == 0 else "td"
                    content_html += f"<{tag}>{cell}</{tag}>"
                content_html += "</tr>"
            content_html += "</table>"
        else:
            # Format as pre
            content_to_show = content
            # Truncate very long content
            if isinstance(content, str) and len(content) > 5000:
                content_to_show = content[:5000] + "... [truncated for display]"
            
            content_html = f"<pre>{content_to_show}</pre>"
        
        # Format metadata
        metadata_html = """
            <div class="metadata">
                <div class="metadata-title">Metadata:</div>
                <pre>{}</pre>
            </div>
        """.format(json.dumps(metadata, indent=2))
        
        # Format relationships
        relationships_html = """
            <div class="block-relationships">
                <div>Parent UUID: {}</div>
                <div>Child UUIDs: {}</div>
            </div>
        """.format(
            parent_uuid or "None", 
            ", ".join(child_uuids) if child_uuids else "None"
        )
        
        html += f"""
            <div id="block-{uuid}" class="block">
                <div class="block-header">
                    <h3>{name}</h3>
                    <span class="block-badge">{block_type}</span>
                </div>
                {docstring_html}
                {content_html}
                {metadata_html}
                {relationships_html}
            </div>
"""
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe filename
        safe_source = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in source)
        safe_source = safe_source.replace('/', '_').replace('\\', '_')
        report_path = os.path.join(output_dir, f"{safe_source}_extraction_report.html")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return report_path
    
    # Otherwise, return the HTML content
    return html


def create_json_report(blocks: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
    """Create a JSON report for the extracted blocks.
    
    Args:
        blocks: List of extraction blocks
        output_path: Optional file path to save the JSON
        
    Returns:
        Path to the JSON file or the JSON string
    """
    # Ensure all blocks have a docstring field
    highlight_docstrings(blocks)
    
    # Create JSON
    json_str = json.dumps(blocks, indent=2)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
    
    return json_str

def create_html_docstring_section(block: Dict[str, Any]) -> str:
    """
    Create a JSON report for the extracted blocks.
    
    Args:
        blocks: List of extraction blocks
        output_path: Optional file path to save the JSON
        
    Returns:
        Path to the JSON file or the JSON string
    """
    json_str = json.dumps(blocks, indent=2)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path
    
    return json_str


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Spot-check DuaLipa extraction on arbitrary files or URLs")
    
    # Source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", help="Local file path to extract")
    source_group.add_argument("--url", help="URL to extract")
    source_group.add_argument("--github", help="GitHub path in format owner/repo/blob/branch/path/to/file")
    
    # Output arguments
    parser.add_argument("--output-dir", help="Directory to save reports")
    parser.add_argument("--format", choices=["html", "json", "both"], default="both", 
                       help="Output format (html, json, or both)")
    
    args = parser.parse_args()
    
    # Determine source
    source = args.file or args.url or args.github
    print(f"📦 Extracting: {source}")
    
    # Extract content
    if args.file:
        blocks = extract_from_file(args.file)
        source_type = "file"
    elif args.url:
        blocks = extract_from_url(args.url)
        source_type = "url"
    elif args.github:
        blocks = extract_from_github(args.github)
        source_type = "github"
    
    print(f"✅ Extracted {len(blocks)} blocks")
    
    # Create output directory if requested
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate reports
    if args.format in ["html", "both"]:
        # Save HTML report
        if output_dir:
            safe_source = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in source)
            safe_source = safe_source.replace('/', '_').replace('\\', '_')
            html_path = os.path.join(output_dir, f"{safe_source}_extraction_report.html")
            html_report = generate_html_report(source, blocks, output_dir)
            print(f"📄 HTML report saved to: {html_path}")
            
            # Try to open in browser
            if os.name == 'posix':
                os.system(f"xdg-open {html_path}")
            elif os.name == 'nt':
                os.system(f"start {html_path}")
            elif os.name == 'darwin':
                os.system(f"open {html_path}")
        else:
            # Print HTML to stdout
            print(generate_html_report(source, blocks))
    
    if args.format in ["json", "both"]:
        # Save JSON report
        if output_dir:
            safe_source = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in source)
            safe_source = safe_source.replace('/', '_').replace('\\', '_')
            json_path = os.path.join(output_dir, f"{safe_source}_extraction_blocks.json")
            create_json_report(blocks, json_path)
            print(f"📄 JSON report saved to: {json_path}")
        else:
            # Print JSON to stdout
            print(create_json_report(blocks))
    
    # Summary statistics
    block_types = {}
    for block in blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
    
    print("\n📊 Extraction Summary:")
    print(f"Source: {source} ({source_type})")
    print(f"Total blocks: {len(blocks)}")
    
    print("\nBlock types:")
    for block_type, count in sorted(block_types.items()):
        print(f"  • {block_type}: {count}")


if __name__ == "__main__":
    main()