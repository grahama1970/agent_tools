import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from markdown_it import MarkdownIt
from collections import OrderedDict
import textwrap

def slugify(title: str) -> str:
    """
    Convert a title to a slug for use in filenames and URLs.
    
    Args:
        title: String to slugify
        
    Returns:
        Slugified string
    """
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

def detect_special_content_from_tokens(tokens, start_idx, end_idx):
    """
    Detect special content types from markdown-it tokens.
    
    Args:
        tokens: List of markdown-it tokens
        start_idx: Starting index in tokens list
        end_idx: Ending index in tokens list
        
    Returns:
        Dictionary of special content flags
    """
    special_content = {
        'has_table': False,
        'has_image': False,
        'has_code_block': False,
        'has_admonition': False,
        'has_bold': False,
        'has_italic': False,
        'has_links': False,
        'has_code_span': False,
        'has_blockquote': False
    }
    
    # Examine tokens to detect special content
    for i in range(start_idx, end_idx):
        token = tokens[i]
        
        # Table detection
        if token.type == 'table_open':
            special_content['has_table'] = True
        
        # Image detection
        elif token.type == 'image':
            special_content['has_image'] = True
        
        # Code block detection
        elif token.type in ('fence', 'code_block') and hasattr(token, 'content'):
            if token.type == 'fence':
                info = token.info if hasattr(token, 'info') else ''
                special_content['has_code_block'] = True
            else:
                # Fix the f-string backslash error
                special_content['has_code_block'] = True
        
        # Blockquote detection
        elif token.type == 'blockquote_open':
            special_content['has_blockquote'] = True
            
            # Check for admonition patterns within blockquotes
            admonition_idx = i + 1
            while admonition_idx < end_idx and tokens[admonition_idx].type != 'blockquote_close':
                if tokens[admonition_idx].type == 'inline' and re.search(
                    r'^\*\*(Note|Warning|Tip|Important|Caution|Danger|Info)\*\*', 
                    tokens[admonition_idx].content, 
                    re.IGNORECASE
                ):
                    special_content['has_admonition'] = True
                    break
                admonition_idx += 1
        
        # Inline elements detection
        elif token.type == 'inline':
            content = token.content
            
            # Bold detection
            if '**' in content or '__' in content:
                special_content['has_bold'] = True
            
            # Italic detection
            if re.search(r'\*[^*].*?\*', content) or re.search(r'_[^_].*?_', content):
                special_content['has_italic'] = True
            
            # Link detection
            if '[' in content and '](' in content:
                special_content['has_links'] = True
            
            # Code span detection
            if '`' in content:
                special_content['has_code_span'] = True
        
        # Direct token type checks for inline formatting
        elif token.type == 'strong_open':
            special_content['has_bold'] = True
        elif token.type == 'em_open':
            special_content['has_italic'] = True
        elif token.type == 'link_open':
            special_content['has_links'] = True
        elif token.type == 'code_inline':
            special_content['has_code_span'] = True
    
    return special_content

def extract_content_between_tokens(tokens, start_idx, end_idx, original_markdown):
    """
    Extract the original markdown content between two token indices.
    
    Args:
        tokens: List of markdown-it tokens
        start_idx: Starting index in tokens list
        end_idx: Ending index in tokens list
        original_markdown: Original markdown text
        
    Returns:
        Original markdown content between the tokens
    """
    if start_idx >= len(tokens) or end_idx > len(tokens) or start_idx >= end_idx:
        return ""
    
    # Get the map positions from tokens
    if hasattr(tokens[start_idx], 'map') and tokens[start_idx].map:
        start_line = tokens[start_idx].map[0]
        
        # Find the end line from the last token that has a map
        end_line = None
        for i in range(end_idx - 1, start_idx - 1, -1):
            if i < len(tokens) and hasattr(tokens[i], 'map') and tokens[i].map:
                end_line = tokens[i].map[1]
                break
        
        if end_line is not None:
            # Extract the content from the original markdown
            lines = original_markdown.split('\n')
            return '\n'.join(lines[start_line:end_line])
    
    # If we can't extract by line numbers, try to reconstruct content from tokens
    content = []
    for i in range(start_idx, end_idx):
        if tokens[i].type == 'inline':
            content.append(tokens[i].content)
        elif tokens[i].type in ('fence', 'code_block') and hasattr(tokens[i], 'content'):
            if tokens[i].type == 'fence':
                info = tokens[i].info if hasattr(tokens[i], 'info') else ''
                content.append("```" + info)
                content.append(tokens[i].content)
                content.append("```")
            else:
                # Fix the f-string backslash error
                indented_content = tokens[i].content.replace('\n', '\n    ')
                content.append(f"    {indented_content}")
    
    return '\n'.join(content)

def extract_hierarchical_sections(markdown_or_path: str) -> List[Dict[str, Any]]:
    """
    Extract hierarchical sections from markdown content or a file path.
    
    Args:
        markdown_or_path: Either markdown content as a string, or a path to a markdown file
        
    Returns:
        List of section objects with title, content, depth, and path information
    """
    # Check if this is a file path
    if os.path.exists(markdown_or_path) and (
        markdown_or_path.lower().endswith('.md') or 
        markdown_or_path.lower().endswith('.markdown') or 
        markdown_or_path.lower().endswith('.mdown')
    ):
        # It's a file path, read the content
        try:
            with open(markdown_or_path, 'r', encoding='utf-8', errors='replace') as f:
                markdown = f.read()
            
            # Get total number of lines for validation
            with open(markdown_or_path, 'r', encoding='utf-8', errors='replace') as f:
                total_line_count = len(f.readlines())
        except Exception as e:
            print(f"Error reading markdown file {markdown_or_path}: {e}")
            return []
    else:
        # Assume it's markdown content
        markdown = markdown_or_path
        total_line_count = len(markdown.split('\n'))
    
    # Normalize line endings and dedent
    markdown = markdown.replace('\r\n', '\n').replace('\r', '\n')
    markdown = textwrap.dedent(markdown)
    
    # Parse with markdown-it
    md = MarkdownIt("commonmark").enable(["table"])
    tokens = md.parse(markdown)
    
    # Find all heading tokens and their positions
    headings = []
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            level = int(token.tag[1])  # Extract level from h1, h2, etc.
            
            # Get the heading text from the next token
            if i + 1 < len(tokens) and tokens[i + 1].type == 'inline':
                title = tokens[i + 1].content
                
                # Get line numbers if available
                start_line = token.map[0] if hasattr(token, 'map') and token.map else 0
                
                headings.append({
                    'title': title,
                    'level': level,
                    'token_idx': i,
                    'start_line': start_line
                })
    
    # If no headings found, return empty list
    if not headings:
        return []
    
    # Split markdown into lines for easier processing
    markdown_lines = markdown.split('\n')
    
    # Determine section content boundaries
    for i, heading in enumerate(headings):
        if i < len(headings) - 1:
            heading['end_line'] = headings[i + 1]['start_line']
        else:
            heading['end_line'] = total_line_count
    
    # Extract content for each section directly from markdown lines
    sections = []
    for heading in headings:
        # Extract content directly from markdown lines
        section_lines = markdown_lines[heading['start_line']:heading['end_line']]
        content = '\n'.join(section_lines)
        
        # Analyze content for special elements
        content_text = content
        content_lines = content.split('\n')
        
        # Detect special content types with enhanced patterns
        # Table detection - improved to better align with markdown-it's implementation
        has_table = bool(re.search(r'\|.*\|[\r\n]+\|[-:\s]*\|', content_text, re.MULTILINE))
        
        # Image detection - standard markdown image syntax
        has_image = bool(re.search(r'!\[.*?\]\(.*?\)', content_text))
        
        # Code block detection - improved to properly identify start and end markers
        has_code_block = bool(re.search(r'```', content_text) or 
                             re.search(r'~~~', content_text) or
                             any(line.strip().startswith('    ') for line in content_lines))
        
        # Admonition detection - enhanced to detect modern formats
        has_admonition = bool(
            re.search(r'^>\s+\*\*(Note|Warning|Tip|Important|Caution|Danger|Info)\*\*', content_text, re.MULTILINE | re.IGNORECASE)
        )
        
        # Link detection
        has_links = bool(re.search(r'\[.*?\]\(.*?\)', content_text))
        
        # Formatting detection
        has_bold = bool(re.search(r'\*\*.*?\*\*', content_text) or re.search(r'__.*?__', content_text))
        has_italic = bool(re.search(r'\*[^*].*?\*', content_text) or re.search(r'_[^_].*?_', content_text))
        
        # Blockquote detection
        has_blockquote = bool(re.search(r'^>\s', content_text, re.MULTILINE))
        
        # Code span detection
        has_code_span = bool(re.search(r'`[^`]*`', content_text))
        
        special_content = {
            'has_table': has_table,
            'has_image': has_image,
            'has_code_block': has_code_block,
            'has_admonition': has_admonition,
            'has_bold': has_bold,
            'has_italic': has_italic,
            'has_links': has_links,
            'has_code_span': has_code_span,
            'has_blockquote': has_blockquote
        }
        
        # Check if section is empty (other than its header)
        is_empty = len(content_lines) <= 1 or all(not line.strip() for line in content_lines[1:])
        
        # Create section object
        section = {
            'title': heading['title'],
            'level': heading['level'],
            'depth': heading['level'],  # For backward compatibility
            'start_line': heading['start_line'] + 1,  # +1 to make it 1-indexed
            'end_line': heading['end_line'],
            'content': content,
            'is_empty': is_empty,
            'special_content': special_content,
            'subsections': []
        }
        
        sections.append(section)
    
    # Build section paths
    for i, section in enumerate(sections):
        path = []
        current_level = section['level']
        
        # Go backward to find parents
        for j in range(i - 1, -1, -1):
            if sections[j]['level'] < current_level:
                path.insert(0, sections[j]['title'])
                current_level = sections[j]['level']
        
        section['path'] = path
        
        # Generate file paths
        title_slug = slugify(section['title'])
        file_name = f"{title_slug}.md"
        file_paths = [file_name]
        
        if section['path']:
            parent_path = '/'.join([slugify(p) for p in section['path']])
            file_paths.append(f"{parent_path}/{file_name}")
        
        section['file_paths'] = file_paths
    
    # Build the hierarchical structure
    section_by_id = {i: section for i, section in enumerate(sections)}
    top_level_sections = []
    
    # Create a map to track parent-child relationships
    children_by_parent = {i: [] for i in range(len(sections))}
    
    # Identify parent-child relationships
    for i, section in enumerate(sections):
        if not section['path']:  # Top-level section
            top_level_sections.append(section)
        else:
            # Find the parent section
            for j in range(i - 1, -1, -1):
                if (sections[j]['level'] < section['level'] and 
                    sections[j]['title'] == section['path'][-1]):
                    children_by_parent[j].append(i)
                    break
    
    # Build the hierarchy
    for parent_idx, child_indices in children_by_parent.items():
        parent = section_by_id[parent_idx]
        for child_idx in child_indices:
            child = section_by_id[child_idx]
            parent['subsections'].append(child)
    
    # If we have no top-level sections but have sections, use the lowest level as top
    if not top_level_sections and sections:
        min_level = min(section['level'] for section in sections)
        for section in sections:
            if section['level'] == min_level and section not in top_level_sections:
                top_level_sections.append(section)
    
    return top_level_sections

def extract_content_blocks(markdown_text):
    """
    Extract content blocks from markdown text using markdown-it-py.
    
    Args:
        markdown_text: Markdown text to parse
        
    Returns:
        List of content blocks with type and content information
    """
    # Parse markdown using markdown-it-py with table plugin enabled
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(markdown_text)
    
    # Extract content blocks
    content_blocks = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        # Handle paragraph blocks
        if token.type == "paragraph_open":
            block = {"type": "paragraph", "content": ""}
            i += 1
            while i < len(tokens) and tokens[i].type != "paragraph_close":
                if tokens[i].type == "inline":
                    block["content"] = tokens[i].content
                i += 1
            content_blocks.append(block)
        
        # Handle code blocks
        elif token.type == "fence":
            block = {
                "type": "code",
                "language": token.info if token.info else "",
                "content": token.content
            }
            content_blocks.append(block)
        
        # Handle images (usually inside paragraphs/inline)
        elif token.type == "image":
            block = {
                "type": "image",
                "src": token.attrs.get("src", "") if hasattr(token, "attrs") and token.attrs else "",
                "alt": token.attrs.get("alt", "") if hasattr(token, "attrs") and token.attrs else "",
                "title": token.attrs.get("title", "") if hasattr(token, "attrs") and token.attrs else ""
            }
            content_blocks.append(block)
        
        # Handle tables
        elif token.type == "table_open":
            table_block = {"type": "table", "header": [], "rows": [], "alignment": []}
            i += 1
            
            # Process table header
            if i < len(tokens) and tokens[i].type == "thead_open":
                i += 1
                if i < len(tokens) and tokens[i].type == "tr_open":
                    i += 1
                    while i < len(tokens) and tokens[i].type != "tr_close":
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
                            if i < len(tokens) and tokens[i].type == "inline":
                                table_block["header"].append(tokens[i].content)
                            i += 1  # Skip th_close
                        else:
                            i += 1
                i += 1  # Skip tr_close
            i += 1  # Skip thead_close
            
            # Process table rows
            while i < len(tokens) and tokens[i].type != "table_close":
                if tokens[i].type == "tbody_open":
                    i += 1
                    while i < len(tokens) and tokens[i].type != "tbody_close":
                        if tokens[i].type == "tr_open":
                            row = []
                            i += 1
                            while i < len(tokens) and tokens[i].type != "tr_close":
                                if tokens[i].type == "td_open":
                                    i += 1
                                    if i < len(tokens) and tokens[i].type == "inline":
                                        cell_content = tokens[i].content
                                        row.append(cell_content)
                                    i += 1  # Skip td_close
                                else:
                                    i += 1
                            table_block["rows"].append(row)
                        i += 1
                i += 1
            
            content_blocks.append(table_block)
        
        # Handle lists
        elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
            list_type = "unordered" if token.type == "bullet_list_open" else "ordered"
            list_block = {"type": "list", "list_type": list_type, "items": []}
            i += 1
            
            while i < len(tokens) and tokens[i].type != "bullet_list_close" and tokens[i].type != "ordered_list_close":
                if tokens[i].type == "list_item_open":
                    i += 1
                    item_content = ""
                    while i < len(tokens) and tokens[i].type != "list_item_close":
                        if tokens[i].type == "inline":
                            item_content = tokens[i].content
                        i += 1
                    list_block["items"].append(item_content)
                i += 1
            
            content_blocks.append(list_block)
        
        # Handle blockquotes (including admonitions)
        elif token.type == "blockquote_open":
            block = {"type": "blockquote", "content": ""}
            blockquote_content = []
            i += 1
            
            # Collect all content within the blockquote
            start_idx = i
            while i < len(tokens) and tokens[i].type != "blockquote_close":
                if tokens[i].type == "paragraph_open":
                    i += 1
                    while i < len(tokens) and tokens[i].type != "paragraph_close":
                        if tokens[i].type == "inline":
                            blockquote_content.append(tokens[i].content)
                        i += 1
                i += 1
            
            block["content"] = "\n".join(blockquote_content)
            
            # Check if this is an admonition
            if block["content"] and re.search(r'^\*\*(Note|Warning|Tip|Important|Caution|Danger|Info)\*\*', block["content"], re.IGNORECASE):
                block["type"] = "admonition"
                match = re.search(r'^\*\*(Note|Warning|Tip|Important|Caution|Danger|Info)\*\*', block["content"], re.IGNORECASE)
                block["admonition_type"] = match.group(1).lower()
                block["content"] = block["content"][match.end():].strip()
            
            content_blocks.append(block)
        
        i += 1
    
    return content_blocks


def markdown_to_json(markdown_text):
    """Convert markdown text to hierarchical JSON"""
    # Extract hierarchical sections
    sections = extract_hierarchical_sections(markdown_text)
    
    # Process each section to extract content blocks
    for section in sections:
        section['content_blocks'] = extract_content_blocks(section['content'])
        # Process subsections recursively
        process_subsections_content_blocks(section)
    
    # Create final result
    result = OrderedDict([
        ("document", OrderedDict([
            ("hierarchy", sections)
        ]))
    ])
    
    return result

def process_subsections_content_blocks(section):
    """Process content blocks for all subsections recursively"""
    for subsection in section.get('subsections', []):
        subsection['content_blocks'] = extract_content_blocks(subsection['content'])
        # Process nested subsections
        process_subsections_content_blocks(subsection)

# For JavaScript parser initialization
def initialize_js_parser():
    """Initialize the JavaScript parser with tree-sitter."""
    try:
        from tree_sitter import Language, Parser
        
        # Create parser
        parser = Parser()
        
        # Updated API: Use the language property instead of set_language method
        JS_LANGUAGE_PATH = os.path.join(os.path.dirname(__file__), 'tree-sitter-javascript.so')
        
        # Make sure the language file exists
        if not os.path.exists(JS_LANGUAGE_PATH):
            raise FileNotFoundError(f"JavaScript language file not found at {JS_LANGUAGE_PATH}")
        
        # Load the JavaScript language
        JAVASCRIPT = Language(JS_LANGUAGE_PATH, 'javascript')
        
        # Set the language (new API)
        parser.language = JAVASCRIPT
        
        return parser
    except Exception as e:
        print(f"Error initializing JavaScript parser: {e}")
        return None
        
# Example usage
if __name__ == "__main__":
    # Example markdown document with nested sections and various content types
    markdown_text = """
# Server Operations

Channels for controlling and monitoring the WireGuard server:

| Channel | Type | Description | Parameters | Returns |
| --- | --- | --- | --- | --- |
| SERVER_START | Request-Response | Starts the WireGuard server | None | boolean (success) |
| SERVER_STOP | Request-Response | Stops the WireGuard server | None | boolean (success) |
| SERVER_STATUS | Request-Response | Gets the current server status | None | ServerStatus object |
| SERVER_STATUS_CHANGED | Event | Notifies of server status changes | N/A | ServerStatus object |

## Configuration Operations

Channels for managing WireGuard configurations:

| Channel | Type | Description | Parameters | Returns |
| --- | --- | --- | --- | --- |
| CONFIG_GET_SERVER | Request-Response | Gets the current server configuration | None | WireGuardConfig object |
| CONFIG_UPDATE_SERVER | Request-Response | Updates the server configuration | WireGuardConfig object | boolean (success) |
"""
    
    # Convert to JSON
    result = markdown_to_json(markdown_text)
    
    import json
    # Print formatted JSON
    print(json.dumps(result, indent=2))
    
    # Optionally save to file
    with open("markdown_structure.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\nJSON output saved to 'markdown_structure.json'")
