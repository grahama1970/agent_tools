"""
Markdown parser implementation using markdown-it-py.

This module provides functionality to extract code blocks and hierarchical sections
from markdown content using the markdown-it-py library.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import OrderedDict

try:
    from markdown_it import MarkdownIt
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False
    print("markdown-it-py not available. Install with: pip install markdown-it-py")

# Import spacy_utils for token counting
try:
    from ..utils.spacy_utils import count_tokens
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spacy_utils not available. Token counting will be disabled.")
    
    # Fallback token counter if spacy is not available
    def count_tokens(text: str) -> int:
        """Simple fallback token counter that splits on whitespace."""
        return len(text.split())


def token_to_dict(token: Any) -> Dict[str, Any]:
    """
    Convert a markdown-it token to a dictionary for debugging and inspection.
    
    Args:
        token: A markdown-it token
        
    Returns:
        Dictionary representation of the token
    """
    result = OrderedDict()
    
    # Add basic token properties
    result["type"] = token.type
    result["tag"] = token.tag if hasattr(token, "tag") else ""
    result["content"] = token.content if hasattr(token, "content") else ""
    
    # Add attributes if present
    if hasattr(token, "attrs") and token.attrs:
        result["attrs"] = OrderedDict(token.attrs)
    
    # Add other properties that might be useful
    if hasattr(token, "level"):
        result["level"] = token.level
    if hasattr(token, "map") and token.map:
        result["map"] = token.map
    
    # Process nested tokens/children
    if hasattr(token, "children") and token.children:
        result["children"] = [token_to_dict(child) for child in token.children]
    
    return result


def markdown_to_hierarchical_json(
    markdown_text: str, 
    file_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convert markdown text to a hierarchical JSON structure.
    
    Args:
        markdown_text: Markdown content as string
        file_path: Optional path to the source file for metadata
        
    Returns:
        Dictionary with hierarchical representation including sections, content blocks,
        and metadata
    """
    if not MARKDOWN_IT_AVAILABLE:
        raise ImportError("markdown-it-py is required but not installed")
    
    # Initialize the parser
    # Using commonmark preset with explicit table support
    md = MarkdownIt("commonmark").enable("table")
    
    # Parse the markdown into tokens
    tokens = md.parse(markdown_text)
    
    # Build section hierarchy first
    hierarchy = build_section_hierarchy(tokens, markdown_text)
    
    # Extract content blocks
    content_blocks = extract_content_blocks(tokens, 0, markdown_text=markdown_text)
    
    # Extract standalone code blocks
    code_blocks = extract_code_blocks(tokens, markdown_text)
    
    # Calculate total token count from sections for consistency
    total_token_count = 0
    for section in hierarchy.values():
        total_token_count += section["metadata"]["total_token_count_with_subsections"]
    
    # File metadata
    file_info = {
        "path": str(file_path) if file_path else None,
        "filename": Path(file_path).name if file_path else None,
        "extension": Path(file_path).suffix if file_path else None,
        "directory": str(Path(file_path).parent) if file_path else None,
        "line_count": markdown_text.count('\n') + 1,
        "token_count": total_token_count
    }
    
    # Create the final result
    result = {
        "document": {
            "file_info": file_info,
            "content_blocks": content_blocks,
            "hierarchy": hierarchy
        },
        "code_blocks": code_blocks
    }
    
    return result


def extract_code_blocks(tokens: List[Any], markdown_text: str) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown tokens.
    
    Args:
        tokens: List of markdown-it tokens
        markdown_text: Original markdown text for line number mapping
        
    Returns:
        List of code blocks with language and position information
    """
    code_blocks = []
    lines = markdown_text.split('\n')
    
    for i, token in enumerate(tokens):
        if token.type == 'fence':  # Code block with language info
            # Estimate line numbers
            line_offset = markdown_text[:token.map[0]].count('\n') if hasattr(token, 'map') and token.map else 0
            start_line = line_offset + 1  # 1-indexed
            end_line = line_offset + token.content.count('\n') + 2  # +2 for the fence lines
            
            # Count tokens in the code block
            token_count = count_tokens(token.content)
            
            code_blocks.append({
                'type': 'code_block',
                'language': token.info.strip(),
                'content': token.content,
                'start_line': start_line,
                'end_line': end_line,
                'token_count': token_count,
                'metadata': {
                    'token_count': token_count
                }
            })
    
    return code_blocks


def process_image(token: Any) -> Dict[str, Any]:
    """
    Process an image token into a standardized block format.
    
    Args:
        token: Image token from markdown-it
        
    Returns:
        Dictionary representation of the image
    """
    # Extract attributes safely handling different attribute structures in markdown-it
    src = ""
    alt = ""
    title = ""
    
    if hasattr(token, "attrs") and token.attrs:
        # Handle both dictionary-style and tuple-list-style attributes
        if isinstance(token.attrs, dict):
            src = token.attrs.get("src", "")
            alt = token.attrs.get("alt", "")
            title = token.attrs.get("title", "")
        else:
            # Assume it's a list of tuples or similar
            for attr_name, attr_value in token.attrs:
                if attr_name == "src":
                    src = attr_value
                elif attr_name == "alt":
                    alt = attr_value
                elif attr_name == "title":
                    title = attr_value
    
    # Get content as alt text if it exists and alt is empty
    if not alt and hasattr(token, "content") and token.content:
        alt = token.content
    
    # Count tokens in alt text
    token_count = count_tokens(alt)
    
    return {
        "type": "image",
        "src": src,
        "alt": alt,
        "title": title,
        "token_count": token_count,
        "metadata": {
            "token_count": token_count,
            "src": src
        }
    }


def extract_content_blocks(tokens: List[Any], start_idx: int, end_idx: Optional[int] = None, markdown_text: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract content blocks from tokens between start_idx and end_idx.
    
    Args:
        tokens: List of markdown-it tokens
        start_idx: Start index in the tokens list
        end_idx: End index in the tokens list (exclusive)
        markdown_text: Original markdown text for token counting
        
    Returns:
        List of content blocks with type, content, and position information
    """
    content_blocks = []
    i = start_idx
    
    if end_idx is None:
        end_idx = len(tokens)
    
    while i < end_idx:
        token = tokens[i]
        
        # Skip heading tokens as they're handled separately
        if token.type in ["heading_open", "heading_close"]:
            i += 1
            continue
        
        # Handle paragraph blocks
        if token.type == "paragraph_open":
            content = ""
            images = []
            i += 1
            while i < end_idx and tokens[i].type != "paragraph_close":
                if tokens[i].type == "inline":
                    content = tokens[i].content
                    
                    # Process any images within the inline content
                    if hasattr(tokens[i], "children"):
                        for child in tokens[i].children:
                            if child.type == "image":
                                images.append(process_image(child))
                i += 1
            
            # Count tokens in the paragraph
            token_count = count_tokens(content)
            
            # Add paragraph content block
            block = {
                "type": "paragraph", 
                "content": content,
                "token_count": token_count,
                "metadata": {
                    "token_count": token_count
                }
            }
            content_blocks.append(block)
            
            # Add image blocks that were inside the paragraph
            content_blocks.extend(images)
            
            i += 1  # Skip paragraph_close
        
        # Handle code blocks
        elif token.type == "fence":
            token_count = count_tokens(token.content)
            
            block = {
                "type": "code",
                "language": token.info if token.info else "",
                "content": token.content,
                "token_count": token_count,
                "metadata": {
                    "token_count": token_count,
                    "language": token.info if token.info else ""
                }
            }
            content_blocks.append(block)
            i += 1
        
        # Handle standalone images (usually inside paragraphs/inline)
        elif token.type == "image":
            content_blocks.append(process_image(token))
            i += 1
        
        # Handle tables - improved extraction logic
        elif token.type == "table_open":
            table_content = ""
            table_block = {
                "type": "table", 
                "header": [], 
                "rows": [], 
                "alignment": []
            }
            i += 1
            
            # Process table header
            if i < end_idx and tokens[i].type == "thead_open":
                i += 1
                if i < end_idx and tokens[i].type == "tr_open":
                    i += 1
                    while i < end_idx and tokens[i].type != "tr_close":
                        if tokens[i].type == "th_open":
                            # Extract alignment information
                            if hasattr(tokens[i], "attrs") and tokens[i].attrs:
                                style = next((v for k, v in tokens[i].attrs if k == "style"), "")
                                if "text-align:right" in style:
                                    table_block["alignment"].append("right")
                                elif "text-align:center" in style:
                                    table_block["alignment"].append("center")
                                elif "text-align:left" in style:
                                    table_block["alignment"].append("left")
                                else:
                                    table_block["alignment"].append("")
                            else:
                                table_block["alignment"].append("")
                            
                            i += 1
                            if i < end_idx and tokens[i].type == "inline":
                                header_text = tokens[i].content
                                table_block["header"].append(header_text)
                                table_content += header_text + " "
                            i += 1  # Skip th_close
                        else:
                            i += 1
                    i += 1  # Skip tr_close
                i += 1  # Skip thead_close
            
            # Process table rows
            while i < end_idx and tokens[i].type != "table_close":
                if tokens[i].type == "tbody_open":
                    i += 1
                    while i < end_idx and tokens[i].type != "tbody_close":
                        if tokens[i].type == "tr_open":
                            row = []
                            i += 1
                            while i < end_idx and tokens[i].type != "tr_close":
                                if tokens[i].type == "td_open":
                                    i += 1
                                    if i < end_idx and tokens[i].type == "inline":
                                        cell_text = tokens[i].content
                                        row.append(cell_text)
                                        table_content += cell_text + " "
                                    i += 1  # Skip td_close
                                else:
                                    i += 1
                            table_block["rows"].append(row)
                            i += 1  # Skip tr_close
                        else:
                            i += 1
                    i += 1  # Skip tbody_close
                else:
                    i += 1
            
            # Count tokens in the table content
            token_count = count_tokens(table_content)
            
            # Add token count to the table block
            table_block["token_count"] = token_count
            table_block["metadata"] = {
                "token_count": token_count,
                "row_count": len(table_block["rows"]),
                "column_count": len(table_block["header"])
            }
            
            content_blocks.append(table_block)
            i += 1  # Skip table_close
        
        # Handle lists
        elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
            list_type = "unordered" if token.type == "bullet_list_open" else "ordered"
            list_block = {"type": "list", "list_type": list_type, "items": []}
            list_content = ""
            i += 1
            
            while i < end_idx and tokens[i].type != "bullet_list_close" and tokens[i].type != "ordered_list_close":
                if tokens[i].type == "list_item_open":
                    i += 1
                    item_content = ""
                    while i < end_idx and tokens[i].type != "list_item_close":
                        if tokens[i].type == "inline":
                            item_content = tokens[i].content
                            list_content += item_content + " "
                        i += 1
                    list_block["items"].append(item_content)
                    i += 1  # Skip list_item_close
                else:
                    i += 1
            
            # Count tokens in the list content
            token_count = count_tokens(list_content)
            
            # Add token count to the list block
            list_block["token_count"] = token_count
            list_block["metadata"] = {
                "token_count": token_count,
                "item_count": len(list_block["items"]),
                "list_type": list_type
            }
            
            content_blocks.append(list_block)
            i += 1  # Skip list_close
        
        # Skip other tokens
        else:
            i += 1
    
    return content_blocks


def build_section_hierarchy(tokens: List[Any], markdown_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a hierarchical structure based on heading levels.
    
    Args:
        tokens: List of markdown-it tokens
        markdown_text: Original markdown text for token counting
        
    Returns:
        Dictionary representing the document's section hierarchy with content
    """
    hierarchy = OrderedDict()
    current_path = []
    section_map = {}  # Maps heading level and position to section object
    
    # First pass: identify all headings and create the structure
    for idx, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])  # Extract level from h1, h2, etc.
            
            # Get the heading text from the next token
            heading_text = ""
            if idx + 1 < len(tokens) and tokens[idx + 1].type == "inline":
                heading_text = tokens[idx + 1].content
            else:
                heading_text = f"Section {level}"
            
            # Create new section with token count field
            new_section = OrderedDict([
                ("title", heading_text),
                ("level", level),
                ("content", []),
                ("subsections", OrderedDict()),
                ("token_count", 0),  # Will be updated in second pass
                ("metadata", {
                    "heading_level": level,
                    "token_count": 0,  # Will be updated in second pass
                    "content_block_count": 0,  # Will be updated in second pass
                    "total_token_count_with_subsections": 0  # Will be updated in third pass
                })
            ])
            
            # Update the current path based on level
            while len(current_path) >= level:
                current_path.pop()
            current_path.append(heading_text)
            
            # Store the section in our map
            section_map[(level, idx)] = new_section
            
            # Add to hierarchy
            if level == 1:
                hierarchy[heading_text] = new_section
            else:
                # Find parent section
                parent_level = level - 1
                parent_idx = idx
                while parent_idx >= 0:
                    if (parent_level, parent_idx) in section_map:
                        parent = section_map[(parent_level, parent_idx)]
                        parent["subsections"][heading_text] = new_section
                        break
                    parent_idx -= 1
                if parent_idx < 0:
                    # Fallback if parent not found
                    hierarchy[heading_text] = new_section
    
    # Handle case where no headings were found
    if not hierarchy:
        # Create a default section
        default_section = OrderedDict([
            ("title", "Document"),
            ("level", 1),
            ("content", extract_content_blocks(tokens, 0, len(tokens), markdown_text)),
            ("subsections", OrderedDict()),
            ("token_count", 0),  # Will be updated below
            ("metadata", {
                "heading_level": 1,
                "token_count": 0,  # Will be updated below 
                "content_block_count": 0,  # Will be updated below
                "total_token_count_with_subsections": 0  # Will be updated below
            })
        ])
        
        # Calculate token count for the default section
        section_token_count = 0
        for block in default_section["content"]:
            section_token_count += block.get("token_count", 0)
        
        default_section["token_count"] = section_token_count
        default_section["metadata"]["token_count"] = section_token_count
        default_section["metadata"]["content_block_count"] = len(default_section["content"])
        default_section["metadata"]["total_token_count_with_subsections"] = section_token_count
        
        hierarchy["Document"] = default_section
        return hierarchy
    
    # Second pass: add content to each section and calculate token counts
    section_boundaries = []
    for idx, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            section_boundaries.append((level, idx))
    
    # Sort boundaries by position
    section_boundaries.sort(key=lambda x: x[1])
    
    # Add content to each section and calculate token counts
    for i, (level, start_idx) in enumerate(section_boundaries):
        # Find the end of this section (next heading or end of document)
        end_idx = len(tokens)
        if i + 1 < len(section_boundaries):
            end_idx = section_boundaries[i + 1][1]
        
        # Get content blocks for this section
        content_blocks = extract_content_blocks(tokens, start_idx + 2, end_idx, markdown_text)  # +2 to skip heading_open and inline
        
        # Calculate total token count for this section
        section_token_count = 0
        for block in content_blocks:
            section_token_count += block.get("token_count", 0)
        
        # Get heading token count (from the heading_text)
        heading_text = ""
        if start_idx + 1 < len(tokens) and tokens[start_idx + 1].type == "inline":
            heading_text = tokens[start_idx + 1].content
        heading_token_count = count_tokens(heading_text)
        
        # Update the section token count and metadata
        section = section_map[(level, start_idx)]
        section["content"] = content_blocks
        section["token_count"] = section_token_count + heading_token_count
        section["metadata"]["token_count"] = section_token_count + heading_token_count
        section["metadata"]["content_block_count"] = len(content_blocks)
        section["metadata"]["heading_token_count"] = heading_token_count
    
    # Third pass: calculate total token counts for each section including subsections
    def calculate_total_tokens(section):
        # Start with tokens from this section's content
        own_tokens = section["token_count"]
        total_tokens = own_tokens
        
        # Add tokens from all subsections
        for title, subsection in section.get("subsections", {}).items():
            subsection_tokens = calculate_total_tokens(subsection)
            total_tokens += subsection_tokens
        
        # Update the metadata to include total tokens (including subsections)
        section["metadata"]["total_token_count_with_subsections"] = total_tokens
        return total_tokens
    
    # Calculate total tokens for each top-level section
    for title, section in hierarchy.items():
        calculate_total_tokens(section)
    
    return hierarchy


def flatten_hierarchy(hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert hierarchical structure to flat list of blocks for compatibility with code_extractor.
    
    Args:
        hierarchy: Hierarchical document structure
        
    Returns:
        Flattened list of content blocks with section information
    """
    flat_blocks = []
    
    def process_section(section, parent_path=None):
        if parent_path is None:
            parent_path = []
        
        # Current section path
        current_path = parent_path + [section["title"]]
        section_path = " > ".join(current_path)
        
        # Add section blocks with section path
        for block in section["content"]:
            block_copy = block.copy()
            block_copy["section"] = section_path
            block_copy["section_level"] = section["level"]
            block_copy["metadata"] = block_copy.get("metadata", {}).copy()
            block_copy["metadata"]["section_path"] = section_path
            block_copy["metadata"]["section_level"] = section["level"]
            flat_blocks.append(block_copy)
        
        # Process subsections
        for sub_title, subsection in section.get("subsections", {}).items():
            process_section(subsection, current_path)
    
    # Process each top-level section
    for title, section in hierarchy.items():
        process_section(section)
    
    return flat_blocks


def process_markdown_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Process a markdown file and extract its structure with metadata.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with extracted sections, code blocks, and metadata
    """
    # Ensure it's a Path object
    path = Path(file_path)
    
    # Check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")
    
    # Check if it's a markdown file
    if not path.suffix.lower() in ('.md', '.markdown', '.mdown'):
        raise ValueError(f"File does not appear to be a markdown file: {path}")
    
    # Read the file
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Error reading markdown file {path}: {e}")
    
    # Process the content
    return markdown_to_hierarchical_json(content, file_path=path)


def get_markdown_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Find all markdown files in a directory.
    
    Args:
        directory: Directory to search in
        recursive: Whether to search in subdirectories
        
    Returns:
        List of paths to markdown files
    """
    path = Path(directory)
    
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Not a valid directory: {path}")
    
    if recursive:
        return list(path.glob('**/*.md')) + list(path.glob('**/*.markdown')) + list(path.glob('**/*.mdown'))
    else:
        return list(path.glob('*.md')) + list(path.glob('*.markdown')) + list(path.glob('*.mdown'))


def process_repository_markdown(
    repo_path: Union[str, Path], 
    output_file: Optional[Union[str, Path]] = None,
    max_files: int = 100
) -> Dict[str, Any]:
    """
    Process markdown files from a repository and extract their structure.
    
    Args:
        repo_path: Path to the repository
        output_file: Optional path to save the JSON output
        max_files: Maximum number of files to process
        
    Returns:
        Dictionary with extracted information from all markdown files
    """
    if not MARKDOWN_IT_AVAILABLE:
        raise ImportError("markdown-it-py is required for processing repositories")
    
    # Get markdown files
    try:
        md_files = get_markdown_files(repo_path)
    except Exception as e:
        raise ValueError(f"Error finding markdown files: {e}")
    
    # Limit the number of files
    md_files = md_files[:max_files]
    
    # Process each file
    results = []
    total_token_count = 0
    
    for file_path in md_files:
        try:
            file_result = process_markdown_file(file_path)
            # Add relative path to make it easier to reference
            file_result['relative_path'] = str(file_path.relative_to(repo_path))
            # Track total token count
            total_token_count += file_result.get('document', {}).get('file_info', {}).get('token_count', 0)
            results.append(file_result)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create the final result
    repository_result = {
        'repository': str(repo_path),
        'file_count': len(results),
        'total_token_count': total_token_count,
        'files': results
    }
    
    # Save to file if output_file is provided
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(repository_result, f, indent=2)
        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
    
    return repository_result


def get_flattened_markdown_content(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Process a markdown file and return a flattened representation for code_extractor compatibility.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with flattened content blocks and code blocks
    """
    # Process the file to get hierarchical structure
    result = process_markdown_file(file_path)
    
    # Flatten the hierarchy
    flat_blocks = flatten_hierarchy(result["document"]["hierarchy"])
    
    # Update the result with flattened blocks
    result["document"]["flat_blocks"] = flat_blocks
    
    return result


# Simple usage example that serves as a smoke test
if __name__ == "__main__":
    # Sample markdown document
    sample_markdown = """# Sample Document

This is a paragraph in the main section.

## Subsection

This is a paragraph in the subsection.

```python
def hello_world():
    print("Hello, World!")
```

### Nested Subsection

Another paragraph in a nested section.

![Sample Image](https://example.com/image.png "Image Title")

## Another Subsection

- List item 1
- List item 2

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
| Cell 3   | Cell 4   |
"""
    
    if MARKDOWN_IT_AVAILABLE:
        # Parse the markdown string
        print("Parsing markdown with markdown-it-py...\n")
        
        try:
            # Test with string content - simple version first
            result = markdown_to_hierarchical_json(sample_markdown)
            
            # Print the JSON output
            print("\nJSON Output:")
            print(json.dumps(result, indent=2))
            
            # Show flattened output
            flat_blocks = flatten_hierarchy(result["document"]["hierarchy"])
            print("\nFlattened Blocks:")
            print(json.dumps(flat_blocks, indent=2))
            
            # Test with file if provided as command line argument
            if len(sys.argv) > 1:
                arg = sys.argv[1]
                
                # Check if it's a file or directory
                path = Path(arg)
                if path.is_file():
                    print(f"\nProcessing markdown file: {path}")
                    try:
                        file_result = process_markdown_file(path)
                        print(f"File processed successfully:")
                        print(f"- Sections: {sum(1 for _ in file_result['document']['hierarchy'].values())}")
                        print(f"- Code blocks: {len(file_result['code_blocks'])}")
                        print(f"- Token count: {file_result['document']['file_info']['token_count']}")
                        
                        # Count tables
                        table_count = 0
                        image_count = 0
                        total_section_tokens = 0
                        for section in file_result['document']['hierarchy'].values():
                            total_section_tokens += section.get('token_count', 0)
                            for block in section.get('content', []):
                                if block.get('type') == 'table':
                                    table_count += 1
                                elif block.get('type') == 'image':
                                    image_count += 1

                        print(f"- Tables: {table_count}")
                        print(f"- Images: {image_count}")
                        print(f"- Total section tokens: {total_section_tokens}")
                        
                        # Show flattened output
                        flat_result = get_flattened_markdown_content(path)
                        print(f"- Flattened blocks: {len(flat_result['document']['flat_blocks'])}")
                        
                    except Exception as e:
                        print(f"Error processing file: {e}")
                
                elif path.is_dir():
                    print(f"\nProcessing repository: {path}")
                    try:
                        repo_result = process_repository_markdown(path, max_files=5)
                        print(f"Repository processed successfully:")
                        print(f"- Files: {repo_result['file_count']}")
                        print(f"- Total token count: {repo_result['total_token_count']}")
                        
                        # Show summary
                        print("\nRepository Summary:")
                        for i, file_data in enumerate(repo_result['files']):
                            print(f"File {i+1}: {file_data['relative_path']}")
                            print(f"  - Sections: {sum(1 for _ in file_data['document']['hierarchy'].values())}")
                            print(f"  - Code blocks: {len(file_data['code_blocks'])}")
                            print(f"  - Token count: {file_data['document']['file_info']['token_count']}")
                        
                    except Exception as e:
                        print(f"Error processing repository: {e}")
                
                else:
                    print(f"Path not found: {path}")
            
            print("\nBasic markdown-it-py functionality working!")
            
        except Exception as e:
            print(f"Error processing markdown: {e}")
    else:
        print("markdown-it-py is not installed. Please install it to use this module.") 