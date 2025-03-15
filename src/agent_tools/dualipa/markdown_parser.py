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


def extract_code_blocks(markdown_content: str) -> Dict[str, str]:
    """Extract code blocks from markdown content.
    
    Args:
        markdown_content: Raw markdown content
        
    Returns:
        Dictionary of code blocks with language as key and code content as value
    """
    code_blocks = {}
    
    # Pattern for backtick code blocks: ```language\ncode\n```
    backtick_pattern = r'```(\w*)\n([\s\S]*?)\n```'
    backtick_matches = re.finditer(backtick_pattern, markdown_content)
    
    for match in backtick_matches:
        language = match.group(1) or "text"
        code = match.group(2)
        # Use a unique key if language already exists
        key = language
        counter = 1
        while key in code_blocks:
            key = f"{language}_{counter}"
            counter += 1
        code_blocks[key] = code
    
    # Pattern for indented code blocks - 4+ spaces at beginning of lines
    # First, split content into lines and find indented blocks
    lines = markdown_content.split('\n')
    in_indented_block = False
    current_block = []
    
    for i, line in enumerate(lines):
        # Check if line starts with 4+ spaces or a tab
        if re.match(r'^( {4,}|\t)', line):
            if not in_indented_block:
                in_indented_block = True
                current_block = []
            # Remove the first 4 spaces (or tab) from the line
            dedented_line = re.sub(r'^( {4}|\t)', '', line, 1)
            current_block.append(dedented_line)
        else:
            # If we were in a block and now we're not, save the block
            if in_indented_block and current_block:
                # Try to detect language from first line comment or keep as "indented"
                language = "indented"
                if current_block and current_block[0].strip().startswith('#'):
                    language = "python"  # Assume Python for # comments
                elif current_block and current_block[0].strip().startswith('//'):
                    language = "javascript"  # Assume JS for // comments
                
                # Use a unique key
                key = language
                counter = 1
                while key in code_blocks:
                    key = f"{language}_{counter}"
                    counter += 1
                
                # Join the block lines and add to code blocks
                code_blocks[key] = '\n'.join(current_block)
                in_indented_block = False
                current_block = []
    
    # Don't forget the last block if file ends with an indented block
    if in_indented_block and current_block:
        language = "indented"
        if current_block and current_block[0].strip().startswith('#'):
            language = "python"
        elif current_block and current_block[0].strip().startswith('//'):
            language = "javascript"
        
        key = language
        counter = 1
        while key in code_blocks:
            key = f"{language}_{counter}"
            counter += 1
        
        code_blocks[key] = '\n'.join(current_block)
    
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

    # Python code
    def hello_world():
        print("Hello, World!")
        return True

And here's a JSON block:

    {
        "name": "DuaLipa",
        "version": "0.1.0",
        "description": "Dual Language Integration for Python AI"
    }

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
        
        for language, content in code_blocks.items():
            logger.info(f"  Block: Language: {language}, Length: {len(content)} chars")
            logger.info(f"  Example content:\n{content.strip()}")
        
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