"""
Markdown-it parser module for DuaLipa.

This is a compatibility module that re-exports functions from the new
markdown_parser module in the extraction package.

Dependencies:
- markdown-it-py
- mdformat
- agent_tools.dualipa.extraction.extractors.markdown.markdown_parser
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

try:
    # Import from the new location
    from agent_tools.dualipa.extraction.extractors.markdown.markdown_parser import (
        parse_markdown, parse_markdown_file, extract_sections_from_markdown,
        extract_code_blocks_from_markdown, extract_tables_from_markdown,
        extract_images_from_markdown, MarkdownParseMode
    )
    
    # Re-export all functions
    __all__ = [
        'parse_markdown', 
        'parse_markdown_file',
        'extract_sections_from_markdown',
        'extract_code_blocks_from_markdown',
        'extract_tables_from_markdown',
        'extract_images_from_markdown',
        'MarkdownParseMode'
    ]
    
except ImportError:
    # Fallback implementations if the new module is not available
    from enum import Enum
    import re
    
    class MarkdownParseMode(Enum):
        """Markdown parsing modes."""
        SECTIONS = "sections"
        CODE_BLOCKS = "code_blocks"
        TABLES = "tables"
        IMAGES = "images"
        ALL = "all"
    
    def parse_markdown(content: str, mode: MarkdownParseMode = MarkdownParseMode.ALL) -> Dict[str, Any]:
        """
        Parse markdown content.
        
        Args:
            content: Markdown content
            mode: Parsing mode
            
        Returns:
            Dictionary with parsing results
        """
        return {
            "sections": extract_sections_from_markdown(content) if mode in [MarkdownParseMode.SECTIONS, MarkdownParseMode.ALL] else [],
            "code_blocks": extract_code_blocks_from_markdown(content) if mode in [MarkdownParseMode.CODE_BLOCKS, MarkdownParseMode.ALL] else [],
            "tables": extract_tables_from_markdown(content) if mode in [MarkdownParseMode.TABLES, MarkdownParseMode.ALL] else [],
            "images": extract_images_from_markdown(content) if mode in [MarkdownParseMode.IMAGES, MarkdownParseMode.ALL] else []
        }
    
    def parse_markdown_file(file_path: Union[str, Path], mode: MarkdownParseMode = MarkdownParseMode.ALL) -> Dict[str, Any]:
        """
        Parse markdown file.
        
        Args:
            file_path: Path to markdown file
            mode: Parsing mode
            
        Returns:
            Dictionary with parsing results
        """
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        result = parse_markdown(content, mode)
        result["file_path"] = str(file_path)
        return result
    
    def extract_sections_from_markdown(content: str) -> List[Dict[str, Any]]:
        """
        Extract sections from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of sections
        """
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            # Check for headings
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": title,
                    "level": level,
                    "content": line,
                    "line_start": i + 1,
                    "line_end": i + 1
                }
            elif current_section:
                # Add to current section
                current_section["content"] += '\n' + line
                current_section["line_end"] = i + 1
        
        # Add last section
        if current_section:
            sections.append(current_section)
            
        return sections
    
    def extract_code_blocks_from_markdown(content: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of code blocks
        """
        code_blocks = []
        pattern = r'```([a-zA-Z0-9]*)\n(.*?)```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1).strip() or "text"
            code = match.group(2)
            
            code_blocks.append({
                "language": language,
                "content": code,
                "line_start": content[:match.start()].count('\n') + 1,
                "line_end": content[:match.end()].count('\n') + 1
            })
            
        return code_blocks
    
    def extract_tables_from_markdown(content: str) -> List[Dict[str, Any]]:
        """
        Extract tables from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of tables
        """
        tables = []
        # Find tables by looking for lines with multiple | characters
        lines = content.split('\n')
        table_start = None
        
        for i, line in enumerate(lines):
            if '|' in line and '-|-' in line.replace(' ', ''):
                # This is a table divider line, table started on the previous line
                if table_start is None and i > 0:
                    table_start = i - 1
                continue
                
            if table_start is not None and ('|' not in line or line.strip() == ''):
                # Table ended
                table_content = '\n'.join(lines[table_start:i])
                
                # Parse table content
                table_rows = []
                for row in table_content.split('\n'):
                    if row.strip() and '|' in row:
                        cells = [cell.strip() for cell in row.split('|')]
                        # Remove empty cells at the beginning and end (caused by leading/trailing |)
                        if cells[0] == '':
                            cells = cells[1:]
                        if cells[-1] == '':
                            cells = cells[:-1]
                        table_rows.append(cells)
                
                if len(table_rows) >= 2:  # Need at least header and separator
                    tables.append({
                        "content": table_content,
                        "rows": table_rows,
                        "header": table_rows[0] if table_rows else [],
                        "line_start": table_start + 1,
                        "line_end": i
                    })
                    
                table_start = None
                
        # Check if table continues to end of content
        if table_start is not None:
            table_content = '\n'.join(lines[table_start:])
            
            # Parse table content
            table_rows = []
            for row in table_content.split('\n'):
                if row.strip() and '|' in row:
                    cells = [cell.strip() for cell in row.split('|')]
                    # Remove empty cells at the beginning and end
                    if cells[0] == '':
                        cells = cells[1:]
                    if cells[-1] == '':
                        cells = cells[:-1]
                    table_rows.append(cells)
            
            if len(table_rows) >= 2:  # Need at least header and separator
                tables.append({
                    "content": table_content,
                    "rows": table_rows,
                    "header": table_rows[0] if table_rows else [],
                    "line_start": table_start + 1,
                    "line_end": len(lines)
                })
                
        return tables
    
    def extract_images_from_markdown(content: str) -> List[Dict[str, Any]]:
        """
        Extract images from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of images
        """
        images = []
        # Match both ![alt](url) and ![alt][ref] with [ref]: url
        pattern = r'!\[(.*?)\]\((.*?)\)|!\[(.*?)\]\[(.*?)\]'
        
        for match in re.finditer(pattern, content):
            if match.group(1) is not None:
                # Direct image
                alt_text = match.group(1)
                url = match.group(2)
            else:
                # Reference image
                alt_text = match.group(3)
                ref = match.group(4)
                # Try to find reference definition
                ref_pattern = fr'\[{re.escape(ref)}\]:\s*(.*?)(?:\s+["\'](.*?)["\'])?\s*$'
                ref_match = re.search(ref_pattern, content, re.MULTILINE)
                url = ref_match.group(1) if ref_match else f"[Reference: {ref}]"
            
            images.append({
                "alt_text": alt_text,
                "url": url,
                "line": content[:match.start()].count('\n') + 1
            })
            
        return images