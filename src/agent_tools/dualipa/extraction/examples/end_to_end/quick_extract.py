#!/usr/bin/env python3
"""
quick_extract.py

A utility script for quickly extracting and validating code or markdown snippets.
This tool is designed for frictionless collaboration and verification of the 
extraction system.

Examples:
    # Extract from code snippet
    python quick_extract.py --code "def hello(): return 'world'" --language python
    
    # Extract from file
    python quick_extract.py --file path/to/file.py
    
    # Extract from URL with Playwright support
    python quick_extract.py --url https://example.com/docs --playwright
"""

import os
import sys
import json
import argparse
import uuid
import tempfile
from pathlib import Path

# Try to import the required modules
try:
    # First try importing from the DuaLipa package
    from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
    from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
    from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
    from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor import extract_markdown_blocks
    from agent_tools.fetch_docs.download_site import download_site
except ImportError:
    # If that fails, try importing from relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Try again with local imports
    from extraction_blocks import extract_all_blocks
    
    # Check which extractors are available
    try:
        from agent_tools.dualipa.extraction.extractors.code.python_extractor import extract_python_blocks
    except ImportError:
        # Fallback to a minimal implementation
        def extract_python_blocks(content, file_path=None):
            return [create_basic_block(content, "function", "python", file_path)]
    
    try:
        from agent_tools.dualipa.extraction.extractors.code.js_ts_extractor import extract_js_ts_blocks
    except ImportError:
        # Fallback to a minimal implementation
        def extract_js_ts_blocks(content, file_path=None, language="javascript"):
            return [create_basic_block(content, "function", language, file_path)]
    
    try:
        from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor import extract_markdown_blocks
    except ImportError:
        # Fallback to a minimal implementation
        def extract_markdown_blocks(content, file_path=None):
            return [create_basic_block(content, "section", "markdown", file_path)]
    
    try:
        from agent_tools.fetch_docs.download_site import download_site
    except ImportError:
        # Fallback to a minimal implementation
        def download_site(url, output_dir, recursive=False, use_playwright=False):
            print(f"Warning: download_site not available. Cannot fetch {url}")
            return False


def create_basic_block(content, block_type, language, file_path=None):
    """
    Create a basic extraction block with all required fields.
    
    Args:
        content: The content of the block
        block_type: The type of block (function, class, section, etc.)
        language: The language of the content
        file_path: Optional path to the source file
        
    Returns:
        A dictionary representing the extraction block
    """
    return {
        "uuid": str(uuid.uuid4()),
        "type": block_type,
        "name": f"{block_type.capitalize()} block",
        "content": content,
        "language": language,
        "file_path": file_path or "snippet.txt",
        "start_line": 1,
        "end_line": content.count("\n") + 1,
        "parent_uuid": None,
        "child_uuids": [],
        "metadata": {
            "language": language,
            "has_docstring": "\"\"\"" in content or "/**" in content or "/*" in content,
        }
    }


def extract_from_code(code, language):
    """
    Extract blocks from a code snippet.
    
    Args:
        code: The code snippet
        language: The programming language
        
    Returns:
        A list of extraction blocks
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=f".{language}", mode="w", delete=False) as temp:
        temp.write(code)
        temp_path = temp.name
    
    try:
        # Extract blocks based on language
        if language == "python":
            blocks = extract_python_blocks(code, temp_path)
        elif language in ["javascript", "js", "typescript", "ts"]:
            blocks = extract_js_ts_blocks(code, temp_path, language)
        else:
            # Generic fallback
            blocks = [create_basic_block(code, "function", language, temp_path)]
        
        return blocks
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def extract_from_markdown(markdown):
    """
    Extract blocks from a markdown snippet.
    
    Args:
        markdown: The markdown content
        
    Returns:
        A list of extraction blocks
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as temp:
        temp.write(markdown)
        temp_path = temp.name
    
    try:
        # Extract markdown blocks
        blocks = extract_markdown_blocks(markdown, temp_path)
        return blocks
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def extract_from_file(file_path):
    """
    Extract blocks from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        A list of extraction blocks
    """
    # Get the absolute path
    abs_path = os.path.abspath(file_path)
    
    # Determine file type
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Read the file
    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract based on file type
    if file_extension in [".py"]:
        return extract_from_code(content, "python")
    elif file_extension in [".js", ".jsx", ".ts", ".tsx"]:
        return extract_from_code(content, file_extension[1:])
    elif file_extension in [".md", ".markdown"]:
        return extract_from_markdown(content)
    else:
        # Generic fallback
        return [create_basic_block(content, "file", file_extension[1:], abs_path)]


def extract_from_url(url, output_dir, use_playwright=False):
    """
    Extract content from a URL.
    
    Args:
        url: The URL to extract from
        output_dir: Directory to save downloaded content
        use_playwright: Whether to use Playwright for JavaScript-rendered sites
        
    Returns:
        A list of extraction blocks
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the site
    success = download_site(url, output_dir, recursive=True, use_playwright=use_playwright)
    
    if not success:
        print(f"Error downloading site from {url}")
        return []
    
    # Run extraction on the downloaded site
    return extract_all_blocks(output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick extraction utility for code and markdown snippets")
    
    # Input source group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--code", help="Code snippet to extract")
    input_group.add_argument("--markdown", help="Markdown snippet to extract")
    input_group.add_argument("--file", help="File to extract from")
    input_group.add_argument("--url", help="URL to extract from")
    
    # Options
    parser.add_argument("--language", default="python", help="Programming language for --code (default: python)")
    parser.add_argument("--output", help="Output file for extraction results")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for URL extraction")
    
    args = parser.parse_args()
    
    # Perform extraction based on input type
    if args.code:
        blocks = extract_from_code(args.code, args.language)
    elif args.markdown:
        blocks = extract_from_markdown(args.markdown)
    elif args.file:
        blocks = extract_from_file(args.file)
    elif args.url:
        # Create a temporary directory for downloads
        output_dir = args.output.rsplit(".", 1)[0] if args.output else "url_extraction"
        os.makedirs(output_dir, exist_ok=True)
        
        blocks = extract_from_url(args.url, output_dir, args.playwright)
    
    # Format the output
    indent = 2 if args.pretty else None
    output = json.dumps(blocks, indent=indent)
    
    # Write output to file or stdout
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Extraction results saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()