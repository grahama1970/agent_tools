"""
Markdown extraction module for DuaLipa.

This module handles extraction of content blocks from markdown files,
including sections, code blocks, and metadata.

Key Features:
1. Section extraction
2. Code block extraction
3. Metadata parsing
4. Block validation

Dependencies:
- markdown-it-py: For markdown parsing
- loguru: For logging

Related Files:
- code_extractor.py: Used for code block extraction
- stats_utils.py: Used for statistics tracking
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    logger.warning("markdown-it-py not available, markdown extraction will be limited")

from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.code.code_extractor import extract_code_blocks

def extract_markdown_blocks(file_path: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract content blocks from a markdown file.
    
    Args:
        file_path: Path to markdown file
        output_dir: Output directory for extracted blocks
        
    Returns:
        List of extracted blocks
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse markdown
        if MARKDOWN_IT_AVAILABLE:
            blocks = _extract_with_markdown_it(content, file_path)
        else:
            blocks = _extract_with_regex(content, file_path)
            
        # Extract code blocks
        code_blocks = []
        for block in blocks:
            if block["type"] == "code":
                language = block["metadata"].get("language")
                if language and language != "unknown":
                    temp_file = output_dir / f"temp.{language}"
                    with open(temp_file, 'w') as f:
                        f.write(block["content"])
                    extracted = extract_code_blocks(str(temp_file), output_dir)
                    code_blocks.extend(extracted)
                    temp_file.unlink()
                    
        # Update stats
        blocks.extend(code_blocks)
        update_stats(stats, blocks, "markdown")
        
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting markdown blocks: {e}")
        return []

def _extract_with_markdown_it(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract blocks using markdown-it."""
    try:
        # Initialize parser
        md = MarkdownIt()
        tokens = md.parse(content)
        
        # Track blocks
        blocks = []
        current_section = None
        
        for token in tokens:
            if token.type == "heading_open":
                # Start new section
                if current_section:
                    blocks.append(current_section)
                current_section = {
                    "type": "section",
                    "name": "",
                    "content": "",
                    "line_start": token.map[0] + 1 if token.map else 0,
                    "line_end": 0,
                    "metadata": {
                        "file": file_path,
                        "level": int(token.tag[1])
                    }
                }
                
            elif token.type == "heading_close" and current_section:
                current_section["line_end"] = token.map[1] + 1 if token.map else 0
                
            elif token.type == "inline" and current_section:
                current_section["name"] = token.content
                current_section["content"] += token.content + "\n"
                
            elif token.type == "fence":
                # Code block
                blocks.append({
                    "type": "code",
                    "name": f"code_block_{len(blocks)}",
                    "content": token.content,
                    "line_start": token.map[0] + 1 if token.map else 0,
                    "line_end": token.map[1] + 1 if token.map else 0,
                    "metadata": {
                        "file": file_path,
                        "language": token.info or "unknown"
                    }
                })
                
            elif token.type == "paragraph_open":
                # Regular content
                if current_section:
                    current_section["content"] += "\n"
                    
            elif token.type == "inline":
                if current_section:
                    current_section["content"] += token.content + "\n"
                    
        # Add final section
        if current_section:
            blocks.append(current_section)
            
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting with markdown-it: {e}")
        return []

def _extract_with_regex(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract blocks using regex patterns."""
    try:
        # Track blocks
        blocks = []
        
        # Extract sections
        section_pattern = r'^(#{1,6})\s+(.+)$'
        current_section = None
        
        for i, line in enumerate(content.split('\n')):
            section_match = re.match(section_pattern, line)
            
            if section_match:
                # End previous section
                if current_section:
                    current_section["line_end"] = i
                    blocks.append(current_section)
                    
                # Start new section
                level = len(section_match.group(1))
                title = section_match.group(2).strip()
                current_section = {
                    "type": "section",
                    "name": title,
                    "content": line + "\n",
                    "line_start": i + 1,
                    "line_end": 0,
                    "metadata": {
                        "file": file_path,
                        "level": level
                    }
                }
                
            elif current_section:
                current_section["content"] += line + "\n"
                
        # Add final section
        if current_section:
            current_section["line_end"] = len(content.split('\n'))
            blocks.append(current_section)
            
        # Extract code blocks
        code_pattern = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            language = match.group(1) or "unknown"
            code = match.group(2)
            start_line = content.count('\n', 0, match.start()) + 1
            end_line = content.count('\n', 0, match.end()) + 1
            
            blocks.append({
                "type": "code",
                "name": f"code_block_{len(blocks)}",
                "content": code,
                "line_start": start_line,
                "line_end": end_line,
                "metadata": {
                    "file": file_path,
                    "language": language
                }
            })
            
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting with regex: {e}")
        return []

def usage_example() -> None:
    """Example usage of markdown extraction."""
    # Example markdown file
    markdown_content = '''
    # Example Document
    
    This is an example markdown document.
    
    ## Code Section
    
    Here's a Python code block:
    
    ```python
    def example_function(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    ```
    
    And a TypeScript block:
    
    ```typescript
    interface Person {
        name: string;
        age: number;
    }
    
    class Example {
        constructor(private data: Person) {}
        
        greet(): string {
            return `Hello, ${this.data.name}!`;
        }
    }
    ```
    '''
    
    # Save to temp file
    with open('example.md', 'w') as f:
        f.write(markdown_content)
        
    # Extract blocks
    blocks = extract_markdown_blocks('example.md', Path('output'))
    
    print("Extracted Blocks:")
    for block in blocks:
        print(f"\nType: {block['type']}")
        print(f"Name: {block['name']}")
        print(f"Lines: {block['line_start']}-{block['line_end']}")
        print("Metadata:", block['metadata'])
        print("Content:")
        print(block['content'])
        
    # Cleanup
    import os
    os.remove('example.md') 