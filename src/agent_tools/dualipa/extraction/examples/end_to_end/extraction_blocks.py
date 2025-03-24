#!/usr/bin/env python3
"""
Code Block Extraction Module for DuaLipa.

This module provides functions for finding source files and extracting code blocks
from those files. It handles different languages and creates structured block
representations that can be further processed for hierarchy analysis and QA generation.

Key Functions:
- find_source_files: Find all source files with specified extensions
- extract_all_blocks: Extract code blocks from all files in a source directory
- extract_markdown_sections: Extract sections from markdown files based on headings

Dependencies:
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- uuid: For unique ID generation (https://docs.python.org/3/library/uuid.html)
- tempfile: For temporary directory creation (https://docs.python.org/3/library/tempfile.html)
- re: For regular expressions (https://docs.python.org/3/library/re.html)

Examples:
    >>> source_dir = Path('./test_repos/python-sample')
    >>> source_files = find_source_files(source_dir)
    >>> blocks = extract_all_blocks(source_dir)
    >>> print(f"Extracted {len(blocks)} blocks")
    Extracted 12 blocks
"""

import os
import uuid
import tempfile
import re  # Make sure re is properly imported and available
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import extraction modules
try:
    from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Make sure you run this script from the project root or add the project to your PYTHONPATH")

# Setup logging
logger = logging.getLogger("extraction.blocks")


def extract_markdown_sections(content: str, file_path: str, parent_uuid: str) -> List[Dict[str, Any]]:
    """Extract sections from markdown files based on headings.
    
    This function parses markdown content and creates blocks for each section
    based on heading levels (# for h1, ## for h2, etc.). It maintains proper
    hierarchical relationships between sections. It also extracts and orders
    special elements like code blocks, tables, and images within sections.
    
    Args:
        content: The content of the markdown file
        file_path: Path to the source file
        parent_uuid: UUID of the parent file block
        
    Returns:
        List of section blocks with hierarchical relationships
        
    Example:
        >>> content = "# Title\\n\\nText\\n\\n## Section\\n\\nMore text"
        >>> sections = extract_markdown_sections(content, "file.md", "parent-uuid")
        >>> len(sections)
        2
    """
    if not content.strip():
        return []
    
    # Regex to match markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    # Find all headings
    headings = [(match.group(1), match.group(2), match.start()) 
                for match in heading_pattern.finditer(content)]
    
    if not headings:
        # If no headings found, create a single section for the entire content
        section_uuid = str(uuid.uuid4())
        section_block = {
            "uuid": section_uuid,
            "id": f"{Path(file_path).stem}_section",
            "name": "Content",
            "type": "section",
            "language": "markdown",
            "content": content,
            "file_path": file_path,
            "parent_uuid": parent_uuid,
            "child_uuids": [],
            "metadata": {
                "language": "markdown",
                "source_file": file_path,
                "level": 0,
                "has_code": "```" in content,
                "has_tables": "|" in content,
                "has_images": "![" in content,
                "position": 0,
                "section_hierarchy": ["Content"]
            }
        }
        
        # Extract ordered elements (code blocks, tables, images) from the content
        elements = extract_ordered_elements_from_markdown(content, section_uuid, file_path)
        if elements:
            section_block["child_uuids"].extend([e["uuid"] for e in elements])
            return [section_block] + elements
        
        return [section_block]
    
    # Sort headings by position in the document
    headings.sort(key=lambda h: h[2])
    
    # Create sections with proper content slices
    sections = []
    special_elements = []  # All extracted elements (code, tables, images)
    
    # Map of level -> (position, uuid) for each active section at that level
    # This allows us to find the parent section for a given heading level
    level_map = {}
    
    for i, (hashes, title, start_pos) in enumerate(headings):
        level = len(hashes)  # Heading level (1-6)
        
        # Get content to the next heading or end of file
        end_pos = headings[i+1][2] if i < len(headings) - 1 else len(content)
        
        # Extract section content excluding the heading itself
        header_end = content.find('\n', start_pos)
        if header_end == -1:
            header_end = len(content)
        
        # Extract just the content after the heading
        section_content_start = header_end + 1
        section_content = content[section_content_start:end_pos].strip()
        
        # For accurate position tracking, get the full section including the heading
        full_section_content = content[start_pos:end_pos].strip()
        
        # Create section block
        section_uuid = str(uuid.uuid4())
        
        # Determine parent UUID based on heading level
        parent_section_uuid = parent_uuid  # Default to file as parent
        section_hierarchy = [title]
        
        # Find the parent section by looking for the closest section with a lower level
        for parent_level in range(level - 1, 0, -1):
            if parent_level in level_map:
                parent_pos, parent_uuid = level_map[parent_level]
                if parent_pos < start_pos:  # Only use if this section appears after parent
                    parent_section_uuid = parent_uuid
                    
                    # Get parent's hierarchy and append this section's title
                    for section in sections:
                        if section["uuid"] == parent_uuid:
                            if "metadata" in section and "section_hierarchy" in section["metadata"]:
                                section_hierarchy = section["metadata"]["section_hierarchy"] + [title]
                            break
                    break
        
        # Update level map with this section
        level_map[level] = (start_pos, section_uuid)
        
        # Clear any higher levels since they're no longer active
        for l in list(level_map.keys()):
            if l > level:
                del level_map[l]
        
        # Create section block with hierarchy information
        section_block = {
            "uuid": section_uuid,
            "id": f"{Path(file_path).stem}_{title.lower().replace(' ', '_')}",
            "name": title,
            "type": "section",
            "language": "markdown",
            "content": full_section_content,  # Include heading in content
            "pure_content": section_content,  # Content without the heading
            "file_path": file_path,
            "parent_uuid": parent_section_uuid,
            "child_uuids": [],
            "metadata": {
                "language": "markdown",
                "source_file": file_path,
                "level": level,
                "has_code": "```" in section_content,
                "has_tables": "|" in section_content,
                "has_images": "![" in section_content,
                "position": start_pos,  # Store position for ordering
                "section_hierarchy": section_hierarchy,
                "section_start": start_pos,
                "section_end": end_pos
            }
        }
        
        # Extract and order elements within this section's content
        elements = extract_ordered_elements_from_markdown(section_content, section_uuid, file_path)
        if elements:
            # Update element positions to be relative to the file, not just the section
            for element in elements:
                if "metadata" in element and "position" in element["metadata"]:
                    # Adjust position to account for the section heading
                    element["metadata"]["position"] += section_content_start
                
                # Add section hierarchy to each element
                if "metadata" in element:
                    element["metadata"]["section_hierarchy"] = section_hierarchy
                
            section_block["child_uuids"].extend([e["uuid"] for e in elements])
            special_elements.extend(elements)
        
        # Add parent-child relationships for sections
        if parent_section_uuid != parent_uuid:  # If parent is another section, not the file
            for section in sections:
                if section["uuid"] == parent_section_uuid:
                    if section_uuid not in section["child_uuids"]:
                        section["child_uuids"].append(section_uuid)
                    break
        
        sections.append(section_block)
    
    # Combine sections and special elements
    all_blocks = sections + special_elements
    
    # Sort all blocks by their position in the document
    all_blocks.sort(key=lambda b: b.get("metadata", {}).get("position", float('inf')))
    
    return all_blocks


def extract_ordered_elements_from_markdown(content: str, parent_uuid: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract ordered elements (text blocks, code blocks, tables, images) from markdown content.
    
    This function finds all elements in markdown content and creates
    separate blocks for each, maintaining their original order.
    
    Args:
        content: The content of the markdown section
        parent_uuid: UUID of the parent section
        file_path: Path to the source file
        
    Returns:
        List of element blocks in their original order
    """
    if not content.strip():
        return []
        
    # First, find all special elements with their positions
    elements = []
    
    # Find all code blocks with positions
    code_matches = list(re.finditer(r'```(\w*)\n(.*?)\n```', content, re.DOTALL))
    for i, match in enumerate(code_matches):
        elements.append({
            "type": "code_block",
            "position": match.start(),
            "end_position": match.end(),
            "match": match,
            "index": i
        })
    
    # Find all tables with positions
    # This regex matches markdown tables with a header row, separator row, and at least one data row
    table_matches = list(re.finditer(r'(\|[^\n]+\|\n\|[\s\-:]+\|(?:\n\|[^\n]+\|)*)', content, re.MULTILINE))
    for i, match in enumerate(table_matches):
        elements.append({
            "type": "table",
            "position": match.start(),
            "end_position": match.end(),
            "match": match,
            "index": i
        })
    
    # Find all images with positions - both standard markdown and HTML wrapped images
    # Standard markdown image syntax: ![alt](url)
    image_matches = list(re.finditer(r'!\[(.*?)\]\((.*?)\)', content))
    for i, match in enumerate(image_matches):
        elements.append({
            "type": "image",
            "position": match.start(),
            "end_position": match.end(),
            "match": match,
            "index": i
        })
    
    # Find HTML-wrapped images with positions - like those in <p align="center"> tags
    html_image_matches = list(re.finditer(r'<(?:p|div)[^>]*>\s*<img\s+src=[\'\"](.*?)[\'\"]+\s+alt=[\'\"](.*?)[\'\"]+[^>]*>\s*</(?:p|div)>', content, re.DOTALL))
    for i, match in enumerate(html_image_matches):
        elements.append({
            "type": "image",
            "position": match.start(),
            "end_position": match.end(),
            "match": match,
            "index": i + len(image_matches),  # Continue indexing from where markdown images left off
            "is_html": True
        })
    
    # Sort all elements by position
    elements.sort(key=lambda e: e["position"])
    
    # Now, add text blocks between special elements
    all_blocks = []
    current_position = 0
    
    for element in elements:
        # If there's text before this element, create a text block
        if element["position"] > current_position:
            text_content = content[current_position:element["position"]].strip()
            if text_content:
                text_uuid = str(uuid.uuid4())
                text_block = {
                    "uuid": text_uuid,
                    "id": f"{Path(file_path).stem}_text_{len(all_blocks)}",
                    "name": f"Text Block {len(all_blocks) + 1}",
                    "type": "text",
                    "language": "markdown",
                    "content": text_content,
                    "file_path": file_path,
                    "parent_uuid": parent_uuid,
                    "child_uuids": [],
                    "metadata": {
                        "language": "markdown",
                        "source_file": file_path,
                        "is_embedded": True,
                        "position": current_position,
                        "element_type": "text"
                    }
                }
                all_blocks.append(text_block)
        
        # Create the appropriate block for this element
        if element["type"] == "code_block":
            match = element["match"]
            language = match.group(1) or "text"
            code = match.group(2)
            
            if code.strip():
                block_uuid = str(uuid.uuid4())
                all_blocks.append({
                    "uuid": block_uuid,
                    "id": f"{Path(file_path).stem}_code_{element['index']}",
                    "name": f"Code Block {element['index'] + 1}",
                    "type": "code_block",
                    "language": language,
                    "content": code,
                    "file_path": file_path,
                    "parent_uuid": parent_uuid,
                    "child_uuids": [],
                    "metadata": {
                        "language": language,
                        "source_file": file_path,
                        "is_embedded": True,
                        "position": element["position"],
                        "element_type": "code_block"
                    }
                })
            
        elif element["type"] == "table":
            match = element["match"]
            table_content = match.group(1)
            
            # Extract table caption/title from preceding line if available
            table_name = f"Table {element['index'] + 1}"
            preceding_text = content[max(0, element["position"]-200):element["position"]].strip()
            if preceding_text:
                lines = preceding_text.split('\n')
                if lines and not lines[-1].startswith('#') and len(lines[-1]) < 200:
                    table_name = lines[-1].strip()
            
            # Parse table rows to get column count
            rows = table_content.strip().split('\n')
            col_count = len([col for col in rows[0].split('|') if col.strip()]) if rows else 0
            row_count = len(rows) - 1  # Subtract the separator row
            
            block_uuid = str(uuid.uuid4())
            all_blocks.append({
                "uuid": block_uuid,
                "id": f"{Path(file_path).stem}_table_{element['index']}",
                "name": table_name,
                "type": "table",
                "language": "markdown",
                "content": table_content,
                "file_path": file_path,
                "parent_uuid": parent_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "markdown",
                    "source_file": file_path,
                    "is_embedded": True,
                    "position": element["position"],
                    "element_type": "table",
                    "row_count": row_count,
                    "col_count": col_count
                }
            })
            
        elif element["type"] == "image":
            match = element["match"]
            is_html = element.get("is_html", False)
            
            if is_html:
                # Extract from HTML image tag
                image_url = match.group(1)
                alt_text = match.group(2)
            else:
                # Extract from markdown image syntax
                alt_text = match.group(1)
                image_url = match.group(2)
            
            # Determine image name/caption
            image_name = alt_text if alt_text else f"Image {element['index'] + 1}"
            
            block_uuid = str(uuid.uuid4())
            all_blocks.append({
                "uuid": block_uuid,
                "id": f"{Path(file_path).stem}_image_{element['index']}",
                "name": image_name,
                "type": "image",
                "language": "markdown",
                "content": f"![{alt_text}]({image_url})",
                "file_path": file_path,
                "parent_uuid": parent_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "markdown",
                    "source_file": file_path,
                    "is_embedded": True,
                    "position": element["position"],
                    "element_type": "image",
                    "alt_text": alt_text,
                    "image_url": image_url,
                    "is_html": is_html
                }
            })
        
        # Update current position to after this element
        current_position = element["end_position"]
    
    # Add any remaining text after the last element
    if current_position < len(content):
        text_content = content[current_position:].strip()
        if text_content:
            text_uuid = str(uuid.uuid4())
            text_block = {
                "uuid": text_uuid,
                "id": f"{Path(file_path).stem}_text_{len(all_blocks)}",
                "name": f"Text Block {len(all_blocks) + 1}",
                "type": "text",
                "language": "markdown",
                "content": text_content,
                "file_path": file_path,
                "parent_uuid": parent_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "markdown",
                    "source_file": file_path,
                    "is_embedded": True,
                    "position": current_position,
                    "element_type": "text"
                }
            }
            all_blocks.append(text_block)
    
    return all_blocks


def find_source_files(source_dir: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """Find all source files with given extensions.
    
    Args:
        source_dir: Directory to search for files
        extensions: List of file extensions to include, defaults to common programming languages
        
    Returns:
        List of file paths matching the specified extensions
        
    Example:
        >>> source_dir = Path('./test_repos/python-sample')
        >>> py_files = find_source_files(source_dir, ['.py'])
        >>> len(py_files)
        2
    """
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go", ".rb", ".md"]
    
    files = []
    for ext in extensions:
        files.extend(source_dir.glob(f"**/*{ext}"))
    
    return files


def extract_all_blocks(source_dir: Path) -> List[Dict[str, Any]]:
    """Extract code blocks from all files in the source directory.
    
    This function finds all source files in the directory, detects their language,
    and extracts code blocks (classes, functions, etc.) from each file.
    
    Args:
        source_dir: Directory containing source files to process
        
    Returns:
        List of extracted code blocks with metadata
        
    Example:
        >>> source_dir = Path('./test_repos/python-sample')
        >>> blocks = extract_all_blocks(source_dir)
        >>> block_types = {block['type'] for block in blocks}
        >>> sorted(list(block_types))
        ['class', 'file', 'function']
    """
    logger.info(f"Extracting code blocks from {source_dir}")
    
    # Import regex here to avoid variable reference errors
    import re
    
    # Find all source files
    source_files = find_source_files(source_dir)
    logger.info(f"Found {len(source_files)} source files")
    
    # Create output directory for extracted blocks
    output_dir = Path(tempfile.mkdtemp(prefix="dualipa_extraction_"))
    
    # Extract blocks from each file
    all_blocks = []
    overall_stats = init_stats()
    overall_stats["total_files"] = len(source_files)
    
    for file_path in source_files:
        try:
            # Skip files larger than 1MB to avoid memory issues
            if file_path.stat().st_size > 1_000_000:
                logger.warning(f"Skipping large file: {file_path}")
                continue
            
            # Use a simpler approach for end-to-end example
            # This directly creates blocks without going through the full extraction pipeline
            file_stats = init_stats()
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Detect language
            language = detect_language(str(file_path))
            if language == "unknown":
                # For markdown files, we need special handling
                if file_path.suffix.lower() == ".md":
                    language = "markdown"
                else:
                    logger.warning(f"Unknown language for file: {file_path}")
                    continue
                
            # Create a simple block for the file itself
            file_block = {
                "uuid": str(uuid.uuid4()),
                "id": Path(file_path).stem,
                "name": Path(file_path).name,
                "type": "file",
                "language": language,
                "content": content,
                "file_path": str(file_path),
                "child_uuids": [],  # Initialize child_uuids for every file
                "metadata": {
                    "language": language,
                    "source_file": str(file_path)
                }
            }
            all_blocks.append(file_block)
            
            # Update stats with language
            file_stats["languages"][language] = file_stats["languages"].get(language, 0) + 1
            
            # Markdown file extraction
            if language == "markdown" or file_path.suffix.lower() == ".md":
                # Extract sections based on headers
                section_blocks = extract_markdown_sections(content, str(file_path), file_block["uuid"])
                if section_blocks:
                    all_blocks.extend(section_blocks)
                    
                    # Update file stats
                    for block in section_blocks:
                        block_type = block.get("type")
                        file_stats["block_types"][block_type] = file_stats["block_types"].get(block_type, 0) + 1
                    
                    # Update file block's child UUIDs for top-level sections
                    for section in section_blocks:
                        # Only add top-level sections as direct children of the file
                        if section.get("parent_uuid") == file_block["uuid"]:
                            file_block["child_uuids"].append(section["uuid"])
            
            # Simple function detection for Python
            elif language == "python":
                import re
                # Find function definitions
                for match in re.finditer(r'def\s+(\w+)\s*\(', content):
                    func_name = match.group(1)
                    start_pos = match.start()
                    
                    # Extract function content (simplified)
                    func_content = content[start_pos:content.find("\n\n", start_pos)]
                    if not func_content:
                        func_content = content[start_pos:content.find("\ndef", start_pos + 4)]
                        
                    if func_content:
                        # Create block for the function
                        func_uuid = str(uuid.uuid4())
                        func_block = {
                            "uuid": func_uuid,
                            "id": f"{Path(file_path).stem}_{func_name}",
                            "name": func_name,
                            "type": "function",
                            "language": language,
                            "content": func_content,
                            "file_path": str(file_path),
                            "parent_uuid": file_block["uuid"],
                            "child_uuids": [],
                            "metadata": {
                                "language": language,
                                "source_file": str(file_path)
                            }
                        }
                        all_blocks.append(func_block)
                        
                        # Add function to file's child UUIDs
                        file_block["child_uuids"].append(func_uuid)
                        
                        # Update file stats
                        file_stats["block_types"]["function"] = file_stats["block_types"].get("function", 0) + 1
                
                # Find class definitions
                for match in re.finditer(r'class\s+(\w+)\s*(?:\(|:)', content):
                    class_name = match.group(1)
                    start_pos = match.start()
                    
                    # Extract class content (simplified)
                    class_content = content[start_pos:content.find("\n\n", start_pos)]
                    if not class_content:
                        class_content = content[start_pos:content.find("\nclass", start_pos + 6)]
                        if not class_content:
                            class_content = content[start_pos:]
                        
                    if class_content:
                        # Create block for the class
                        class_uuid = str(uuid.uuid4())
                        class_block = {
                            "uuid": class_uuid,
                            "id": f"{Path(file_path).stem}_{class_name}",
                            "name": class_name,
                            "type": "class",
                            "language": language,
                            "content": class_content,
                            "file_path": str(file_path),
                            "parent_uuid": file_block["uuid"],
                            "child_uuids": [],
                            "metadata": {
                                "language": language,
                                "source_file": str(file_path)
                            }
                        }
                        all_blocks.append(class_block)
                        
                        # Add class to file's child UUIDs
                        file_block["child_uuids"].append(class_uuid)
                        
                        # Update file stats
                        file_stats["block_types"]["class"] = file_stats["block_types"].get("class", 0) + 1
                        
                        # Find class methods 
                        method_pattern = re.compile(r'def\s+(\w+)\s*\(self', re.MULTILINE)
                        for method_match in method_pattern.finditer(class_content):
                            method_name = method_match.group(1)
                            method_start = method_match.start()
                            
                            # Create a method block
                            method_uuid = str(uuid.uuid4())
                            method_block = {
                                "uuid": method_uuid,
                                "id": f"{Path(file_path).stem}_{class_name}_{method_name}",
                                "name": method_name,
                                "type": "method",
                                "language": language,
                                "content": class_content[method_start:],
                                "file_path": str(file_path),
                                "parent_uuid": class_uuid,
                                "child_uuids": [],
                                "metadata": {
                                    "language": language,
                                    "source_file": str(file_path),
                                    "class_name": class_name
                                }
                            }
                            all_blocks.append(method_block)
                            
                            # Add method to class's child UUIDs
                            class_block["child_uuids"].append(method_uuid)
                            
                            # Update file stats
                            file_stats["block_types"]["method"] = file_stats["block_types"].get("method", 0) + 1
            
            # Add JavaScript/TypeScript support
            elif language in ["javascript", "typescript"]:
                # Find functions and classes using regex (enhanced)
                # Function patterns: matches various function declarations
                js_func_patterns = [
                    # Regular function declarations
                    r'function\s+(\w+)\s*\(',
                    # Variable assignments with functions
                    r'(?:var|let|const)\s+(\w+)\s*=\s*function',
                    # Arrow functions
                    r'(?:var|let|const)\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
                    # Exports
                    r'exports\.(\w+)\s*=',
                    # Prototype methods
                    r'(?:\w+)\.prototype\.(\w+)\s*=',
                ]
                
                # Find all function matches
                func_matches = []
                for pattern in js_func_patterns:
                    compiled = re.compile(pattern, re.MULTILINE)
                    for match in compiled.finditer(content):
                        func_name = match.group(1)
                        if func_name:
                            func_matches.append((func_name, match.start()))
                
                # Sort by position
                func_matches.sort(key=lambda x: x[1])
                
                # Process each function
                for i, (func_name, start_pos) in enumerate(func_matches):
                    # Find next function start position for content extraction
                    next_pos = len(content)
                    if i < len(func_matches) - 1:
                        next_pos = func_matches[i+1][1]
                    
                    # Extract function content
                    func_content = content[start_pos:next_pos].strip()
                    
                    # The function content has already been extracted above
                    # func_content already contains the content between this function and the next
                    
                    # Create block for the function
                    func_uuid = str(uuid.uuid4())
                    func_block = {
                        "uuid": func_uuid,
                        "id": f"{Path(file_path).stem}_{func_name}",
                        "name": func_name,
                        "type": "function",
                        "language": language,
                        "content": func_content,
                        "file_path": str(file_path),
                        "parent_uuid": file_block["uuid"],
                        "child_uuids": [],
                        "metadata": {
                            "language": language,
                            "source_file": str(file_path)
                        }
                    }
                    all_blocks.append(func_block)
                    
                    # Add function to file's child UUIDs
                    file_block["child_uuids"].append(func_uuid)
                    
                    # Update file stats
                    file_stats["block_types"]["function"] = file_stats["block_types"].get("function", 0) + 1
                
                # Find classes
                class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', re.MULTILINE)
                for match in class_pattern.finditer(content):
                    class_name = match.group(1)
                    parent_class = match.group(2)
                    start_pos = match.start()
                    
                    # Find the end of the class
                    # Look for next class or end of content
                    next_class = content.find("class", start_pos + 5)
                    if next_class == -1:
                        next_class = len(content)
                    
                    class_content = content[start_pos:next_class].strip()
                    
                    # Create block for the class
                    class_uuid = str(uuid.uuid4())
                    class_block = {
                        "uuid": class_uuid,
                        "id": f"{Path(file_path).stem}_{class_name}",
                        "name": class_name,
                        "type": "class",
                        "language": language,
                        "content": class_content,
                        "file_path": str(file_path),
                        "parent_uuid": file_block["uuid"],
                        "child_uuids": [],
                        "metadata": {
                            "language": language,
                            "source_file": str(file_path)
                        }
                    }
                    
                    # Add inheritance information if available
                    if parent_class:
                        class_block["metadata"]["inheritance"] = [parent_class]
                    
                    all_blocks.append(class_block)
                    
                    # Add class to file's child UUIDs
                    file_block["child_uuids"].append(class_uuid)
                    
                    # Update file stats
                    file_stats["block_types"]["class"] = file_stats["block_types"].get("class", 0) + 1
                    
                    # Find methods
                    method_pattern = re.compile(r'(?:(\w+)\s*\([^)]*\)\s*{|\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))', re.MULTILINE)
                    for method_match in method_pattern.finditer(class_content):
                        method_name = method_match.group(1) or method_match.group(2)
                        if not method_name or method_name == "constructor":
                            continue
                            
                        method_start = method_match.start()
                        
                        # Create a method block
                        method_uuid = str(uuid.uuid4())
                        method_block = {
                            "uuid": method_uuid,
                            "id": f"{Path(file_path).stem}_{class_name}_{method_name}",
                            "name": method_name,
                            "type": "method",
                            "language": language,
                            "content": class_content[method_start:],
                            "file_path": str(file_path),
                            "parent_uuid": class_uuid,
                            "child_uuids": [],
                            "metadata": {
                                "language": language,
                                "source_file": str(file_path),
                                "class_name": class_name
                            }
                        }
                        all_blocks.append(method_block)
                        
                        # Add method to class's child UUIDs
                        class_block["child_uuids"].append(method_uuid)
                        
                        # Update file stats
                        file_stats["block_types"]["method"] = file_stats["block_types"].get("method", 0) + 1
            
            # Update file stats
            file_stats["block_types"]["file"] = 1
            file_stats["total_files"] += 1
            update_stats(overall_stats, [file_block], language)
            
            logger.debug(f"Extracted blocks from {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting blocks from {file_path}: {e}")
            overall_stats["errors"].append(str(e))
    
    logger.info(f"Extraction complete: {len(all_blocks)} blocks from {overall_stats['total_files']} files")
    
    # Try to enhance with documentation from fetch_docs
    try:
        from agent_tools.dualipa.fetch_docs_integration import integrate_docs_with_extraction
        logger.info("Enhancing extraction with documentation from fetch_docs")
        all_blocks = integrate_docs_with_extraction(source_dir, all_blocks)
        logger.info(f"Enhanced extraction complete: {len(all_blocks)} blocks total (including documentation)")
    except ImportError:
        logger.info("fetch_docs_integration module not found, skipping documentation enhancement")
    except Exception as e:
        logger.error(f"Error enhancing extraction with documentation: {e}")
    
    return all_blocks