"""
Markdown parser for DuaLipa.

Provides functionality to parse and extract content from markdown files,
using mistune as the primary parser backend with a robust regex fallback.

Official Documentation References:
- mistune: https://mistune.readthedocs.io/en/latest/
- loguru: https://loguru.readthedocs.io/en/stable/
"""

import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from loguru import logger

# Try importing mistune for markdown parsing
try:
    import mistune
    MISTUNE_AVAILABLE = True
    logger.info("mistune is available for markdown parsing")
except ImportError:
    MISTUNE_AVAILABLE = False
    logger.warning("mistune not available, will use regex-based fallback")

# Set MARKDOWN_IT_AVAILABLE to False for backward compatibility with tests
MARKDOWN_IT_AVAILABLE = False

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def extract_sections_from_markdown(content: str) -> List[Dict[str, Any]]:
    """
    Extract sections from markdown content using headers as delimiters.
    
    Args:
        content: The markdown content string
        
    Returns:
        List of dictionaries with keys: level, title, content
    """
    try:
        # Add a newline to ensure proper header detection
        if not content.endswith('\n'):
            content += '\n'
        
        # Pattern to match headers (# Header)
        header_pattern = r'^(#{1,6})\s+(.+?)\s*$'
        
        # Split content by headers
        lines = content.split('\n')
        sections = []
        current_level = 0
        current_title = "Overview"
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section
                if current_content:
                    sections.append({
                        "level": current_level,
                        "title": current_title,
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_level = len(header_match.group(1))  # Number of # characters
                current_title = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections.append({
                "level": current_level,
                "title": current_title,
                "content": '\n'.join(current_content).strip()
            })
        
        return sections
    except Exception as e:
        logger.error(f"Error extracting sections from markdown: {e}")
        raise


def extract_code_blocks(markdown_content: str) -> List[Dict[str, Any]]:
    """Extract code blocks from markdown content.
    
    Args:
        markdown_content: Raw markdown content
        
    Returns:
        List of dictionaries with keys: language, content
    """
    # Normalize line endings
    markdown_content = markdown_content.replace('\r\n', '\n')
    
    # Use mistune if available (preferred method)
    if MISTUNE_AVAILABLE:
        try:
            blocks = _extract_blocks_with_mistune(markdown_content)
            if blocks:
                logger.info(f"Successfully extracted {len(blocks)} code blocks using mistune")
                return blocks
        except Exception as e:
            logger.warning(f"Error using mistune: {e}")
    
    # Fall back to regex-based approach
    logger.info("Using regex fallback for code block extraction")
    return _extract_blocks_with_regex(markdown_content)


def _extract_blocks_with_mistune(content: str) -> List[Dict[str, Any]]:
    """Extract code blocks using mistune.
    
    This implementation uses a custom renderer to capture code blocks 
    during the markdown parsing process.
    
    Args:
        content: Markdown content
        
    Returns:
        List of dictionaries with language and content
    """
    blocks = []
    
    # Create a custom renderer to capture code blocks
    class CodeBlockRenderer(mistune.HTMLRenderer):
        def block_code(self, code, info=None):
            # Extract language from info string
            if info:
                language = info.strip().split(None, 1)[0]
            else:
                language = 'text'
            
            # Add the block to our collection
            blocks.append({
                "language": language,
                "content": code.strip()
            })
            
            # Return empty string as we don't need the HTML output
            return ""
    
    # Create markdown parser with our custom renderer
    markdown = mistune.create_markdown(renderer=CodeBlockRenderer())
    
    # Parse the markdown content
    markdown(content)
    
    # Log extraction details for debugging
    logger.debug(f"Extracted {len(blocks)} code blocks with mistune")
    for i, block in enumerate(blocks):
        logger.debug(f"Block {i+1}: language={block['language']}, content preview: {block['content'][:50]}...")
    
    return blocks


def _extract_blocks_with_regex(content: str) -> List[Dict[str, Any]]:
    """Extract code blocks using regular expressions.
    
    This implementation handles both fenced code blocks (```language)
    and indented code blocks (4 spaces or tab).
    
    Args:
        content: Markdown content
        
    Returns:
        List of dictionaries with language and content
    """
    blocks = []
    
    # Extract fenced code blocks: ```language\ncode\n```
    # This pattern is more robust and handles various markdown code block formats
    fenced_pattern = r'```([\w\-]*)\s*\n([\s\S]*?)\n\s*```'
    
    for match in re.finditer(fenced_pattern, content):
        language = match.group(1).strip() or 'text'
        code = match.group(2).strip()
        
        if code:  # Only add non-empty blocks
            blocks.append({
                "language": language,
                "content": code
            })
    
    # If no fenced blocks found, try to extract indented code blocks
    if not blocks:
        # Pattern for indented code blocks (4 spaces or 1 tab)
        indented_lines = []
        in_code_block = False
        
        for line in content.split('\n'):
            if line.startswith('    ') or line.startswith('\t'):
                # This is a code line (indented with 4 spaces or tab)
                if not in_code_block:
                    in_code_block = True
                indented_lines.append(line.removeprefix('    ').removeprefix('\t'))
            else:
                # This is not a code line
                if in_code_block and indented_lines:
                    # End of a code block - join the lines and add to blocks
                    code_content = '\n'.join(indented_lines).strip()
                    if code_content:
                        blocks.append({
                            "language": "text",  # Indented blocks don't specify language
                            "content": code_content
                        })
                    indented_lines = []
                    in_code_block = False
        
        # Don't forget to add the last block if we ended inside a code block
        if in_code_block and indented_lines:
            code_content = '\n'.join(indented_lines).strip()
            if code_content:
                blocks.append({
                    "language": "text",
                    "content": code_content
                })
    
    # Log extraction details for debugging
    logger.debug(f"Extracted {len(blocks)} code blocks using regex")
    for i, block in enumerate(blocks):
        logger.debug(f"Block {i+1}: language={block['language']}, content preview: {block['content'][:50]}...")
    
    return blocks


def get_markdown_files(repo_path: str, pattern: Optional[str] = None) -> List[str]:
    """Get all markdown files from a repository.
    
    Args:
        repo_path: Path to the repository
        pattern: Optional glob pattern to filter files (e.g., "**/docs/*.md")
        
    Returns:
        List of paths to markdown files
    """
    markdown_files = []
    
    try:
        repo_path = Path(repo_path)
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return markdown_files
        
        # Walk through the repository and find markdown files
        if pattern:
            # Use the provided pattern to find files
            logger.debug(f"Using pattern '{pattern}' to find markdown files")
            
            # If pattern already specifies extensions, use it directly
            if any(ext in pattern for ext in ['.md', '.markdown', '.mdown', '.mkd']):
                for file_path in repo_path.glob(pattern):
                    if file_path.is_file():
                        markdown_files.append(str(file_path))
            else:
                # Otherwise, still filter by markdown extensions
                for file_path in repo_path.glob(pattern):
                    if file_path.is_file() and file_path.suffix.lower() in ['.md', '.markdown', '.mdown', '.mkd']:
                        markdown_files.append(str(file_path))
        else:
            # Find all markdown files in the repository
            for file_path in repo_path.glob("**/*.*"):
                # Check if it's a markdown file
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.markdown', '.mdown', '.mkd']:
                    markdown_files.append(str(file_path))
        
        logger.info(f"Found {len(markdown_files)} markdown files in {repo_path}")
    except Exception as e:
        logger.error(f"Error finding markdown files: {e}")
    
    return markdown_files


def process_markdown_file(file_path: str) -> Dict[str, Any]:
    """Process a markdown file, extracting sections and code blocks.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with file information, sections, and code blocks
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        relative_path = os.path.basename(file_path)
        
        # Extract sections and code blocks
        sections = extract_sections_from_markdown(content)
        code_blocks = extract_code_blocks(content)
        
        return {
            "path": relative_path,
            "content": content,
            "sections": sections,
            "code_blocks": code_blocks
        }
    except Exception as e:
        logger.error(f"Error processing markdown file {file_path}: {e}")
        raise


def extract_code_blocks_from_documentation(repo_path: str) -> Dict[str, Any]:
    """Extract code blocks from documentation files in a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Dictionary with stats and extracted code blocks
    """
    markdown_files = get_markdown_files(repo_path)
    
    results = {
        "total_files": len(markdown_files),
        "processed_files": 0,
        "code_blocks": [],
        "languages": {},
        "errors": []
    }
    
    for file_path in markdown_files:
        try:
            file_result = process_markdown_file(file_path)
            results["processed_files"] += 1
            
            # Add code blocks to the results
            for block in file_result["code_blocks"]:
                block["file"] = file_path
                results["code_blocks"].append(block)
                
                # Count languages
                lang = block.get("language", "unknown")
                results["languages"][lang] = results["languages"].get(lang, 0) + 1
                
        except Exception as e:
            results["errors"].append({
                "file": file_path,
                "error": str(e)
            })
    
    return results


def markdown_to_html(markdown_content: str) -> str:
    """Convert markdown content to HTML.
    
    Args:
        markdown_content: Markdown content
        
    Returns:
        HTML content
    """
    if MISTUNE_AVAILABLE:
        try:
            # Create standard HTML renderer
            renderer = mistune.HTMLRenderer()
            markdown = mistune.create_markdown(renderer=renderer)
            return markdown(markdown_content)
        except Exception as e:
            logger.error(f"Error converting markdown to HTML: {e}")
    
    # If mistune not available or error occurred
    logger.warning("Could not convert markdown to HTML, returning raw content")
    return f"<pre>{markdown_content}</pre>"


def demo_markdown_parser() -> None:
    """Demonstrate the markdown parser functionality with examples."""
    try:
        logger.info("Markdown Parser Demo")
        logger.info("====================")
        
        # Example markdown with code blocks, headers, and formatting
        example_markdown = """# Markdown Parser Example
        
This is an example of markdown content with various elements.

## Code Blocks

Here's a Python code block:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And here's a JSON block:

```json
{
    "name": "DuaLipa",
    "version": "0.1.0",
    "description": "Dual Language Integration for Python AI"
}
```

## Lists and Formatting

- Item 1
- Item 2
  - Nested item
  - Another nested item
- Item 3

**Bold text** and *italic text* are supported.

## Links and References

Check out [Python](https://python.org) for more information.
"""
        
        # 1. Extract code blocks
        logger.info("\n1. Extracting code blocks:")
        code_blocks = extract_code_blocks(example_markdown)
        logger.info(f"  Found {len(code_blocks)} code blocks")
        
        for block in code_blocks:
            logger.info(f"  Block: Language: {block['language']}, Length: {len(block['content'])} chars")
            logger.info(f"  Example content:\n{block['content']}")
        
        # 2. Extract sections
        logger.info("\n2. Extracting sections:")
        sections = extract_sections_from_markdown(example_markdown)
        logger.info(f"  Found {len(sections)} sections")
        
        for section in sections:
            logger.info(f"  Section: '{section['title']}', Level: {section['level']}, Length: {len(section['content'])} chars")
        
        # 3. Convert to HTML
        logger.info("\n3. Converting to HTML:")
        html = markdown_to_html(example_markdown)
        logger.info(f"  Converted markdown to HTML (length: {len(html)} chars)")
        
        # Show a snippet of the HTML output
        html_preview = html[:150] + "..." if len(html) > 150 else html
        logger.info(f"  HTML preview:\n  {html_preview}")
        
        logger.info("\nMarkdown Parser Demo Completed")
        
    except Exception as e:
        logger.error(f"Error in markdown parser demo: {e}")


if __name__ == "__main__":
    # Run the demonstration when the module is executed directly
    demo_markdown_parser()
    
    # Example of processing a markdown file
    if len(sys.argv) > 1:
        try:
            input_file = sys.argv[1]
            logger.info(f"Processing markdown file: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Extract and display code blocks
            blocks = extract_code_blocks(markdown_content)
            logger.info(f"Found {len(blocks)} code blocks in the file")
            
            # Extract and display sections
            sections = extract_sections_from_markdown(markdown_content)
            logger.info(f"Found {len(sections)} sections in the file")
            
            # Optional: Save output to JSON
            output_file = Path(input_file).with_suffix('.json')
            result = {
                "code_blocks": blocks,
                "sections": sections
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Results saved to: {output_file}")
            
            # Convert to HTML if requested
            if "--html" in sys.argv:
                html_file = Path(input_file).with_suffix('.html')
                html_content = markdown_to_html(markdown_content)
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"HTML output saved to: {html_file}")
                
        except Exception as e:
            logger.error(f"Error processing file: {e}") 