"""
Markdown parser for DuaLipa.

Provides functionality to parse and extract content from markdown files,
using either markdown-it-py or mistune as the parser backend.

Official Documentation References:
- markdown-it-py: https://markdown-it-py.readthedocs.io/en/latest/
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

# Try importing markdown parsers - we support multiple options
try:
    import markdown_it
    MARKDOWN_IT_AVAILABLE = True
    logger.info("markdown-it-py is available for markdown parsing")
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    logger.warning("markdown-it-py not available, will try alternative parsers")

try:
    import mistune
    MISTUNE_AVAILABLE = True
    logger.info("mistune is available for markdown parsing")
except ImportError:
    MISTUNE_AVAILABLE = False
    logger.warning("mistune not available")

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Check if any parser is available
if not (MARKDOWN_IT_AVAILABLE or MISTUNE_AVAILABLE):
    logger.error("No markdown parser available. Install markdown-it-py or mistune.")


def extract_sections_from_markdown(content: str) -> Dict[str, str]:
    """
    Extract sections from markdown content using headers as delimiters.
    
    Args:
        content: The markdown content string
        
    Returns:
        Dictionary mapping section titles to their content
    """
    try:
        # Add a newline to ensure proper header detection
        if not content.endswith('\n'):
            content += '\n'
        
        # Pattern to match headers (# Header)
        header_pattern = r'^(#{1,6})\s+(.+?)\s*$'
        
        # Split content by headers
        lines = content.split('\n')
        sections = {}
        current_section = "Overview"
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                level = len(header_match.group(1))  # Number of # characters
                title = header_match.group(2).strip()
                current_section = title
                current_content = []
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    except Exception as e:
        logger.error(f"Error extracting sections from markdown: {e}")
        return {"Error": str(e)}


def extract_code_blocks(markdown_content: str) -> List[Dict[str, str]]:
    """Extract code blocks from markdown content.
    
    Args:
        markdown_content: Raw markdown content
        
    Returns:
        List of code blocks with language and code content
    """
    # Pattern for code blocks: ```language\ncode\n```
    pattern = r'```(\w*)\n([\s\S]*?)\n```'
    matches = re.finditer(pattern, markdown_content)
    
    code_blocks = []
    for match in matches:
        language = match.group(1) or "text"
        code = match.group(2)
        code_blocks.append({
            "language": language,
            "content": code
        })
    
    return code_blocks


def get_markdown_files(repo_path: str) -> List[str]:
    """Get all markdown files from a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of paths to markdown files
    """
    markdown_files = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.lower().endswith(('.md', '.markdown')):
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)
    
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


def demo_markdown_parser() -> None:
    """Demonstrate the markdown parser functionality with examples.
    
    This function shows how to use the main components of the markdown parser:
    1. Extracting code blocks from markdown
    2. Parsing and extracting sections from markdown
    3. Converting markdown to HTML
    
    Returns:
        None - prints results to the console
    """
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
        
        for i, block in enumerate(code_blocks):
            language = block.get("language", "unknown")
            content = block.get("content", "")
            logger.info(f"  Block {i+1}: Language: {language}, Length: {len(content)} chars")
            if i == 0:  # Show the first block as example
                logger.info(f"  Example content:\n  {content.strip()}")
        
        # 2. Extract sections
        logger.info("\n2. Extracting sections:")
        sections = extract_sections_from_markdown(example_markdown)
        logger.info(f"  Found {len(sections)} sections")
        
        for title, content in sections.items():
            logger.info(f"  Section: '{title}', Length: {len(content)} chars")
        
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
                "sections": {k: v for k, v in sections.items()}
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