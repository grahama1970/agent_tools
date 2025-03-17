from markdown_it import MarkdownIt
import json
from collections import OrderedDict
from pathlib import Path

def token_to_ordered_dict(token):
    """Convert a markdown-it token to an OrderedDict for JSON serialization"""
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
        result["children"] = [token_to_ordered_dict(child) for child in token.children]
    
    return result

def extract_content_blocks(tokens, start_idx, end_idx=None):
    """Extract content blocks from tokens between start_idx and end_idx"""
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
            block = {"type": "paragraph", "content": ""}
            i += 1
            while i < end_idx and tokens[i].type != "paragraph_close":
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
                "src": token.attrs.get("src", [""])[0] if hasattr(token, "attrs") and token.attrs else "",
                "alt": token.attrs.get("alt", [""])[0] if hasattr(token, "attrs") and token.attrs else "",
                "title": token.attrs.get("title", [""])[0] if hasattr(token, "attrs") and token.attrs else ""
            }
            content_blocks.append(block)
        
        # Handle tables - improved extraction logic
        elif token.type == "table_open":
            table_block = {"type": "table", "header": [], "rows": [], "alignment": []}
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
                                table_block["header"].append(tokens[i].content)
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
                                        row.append(tokens[i].content)
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
            
            while i < end_idx and tokens[i].type != "bullet_list_close" and tokens[i].type != "ordered_list_close":
                if tokens[i].type == "list_item_open":
                    i += 1
                    item_content = ""
                    while i < end_idx and tokens[i].type != "list_item_close":
                        if tokens[i].type == "inline":
                            item_content = tokens[i].content
                        i += 1
                    list_block["items"].append(item_content)
                i += 1
            
            content_blocks.append(list_block)
        
        i += 1
    
    return content_blocks

def build_section_hierarchy(tokens):
    """Build a hierarchical structure based on heading levels"""
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
            
            # Create new section
            new_section = OrderedDict([
                ("title", heading_text),
                ("level", level),
                ("content", []),
                ("subsections", OrderedDict())
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
    
    # Second pass: add content to each section
    section_boundaries = []
    for idx, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            section_boundaries.append((level, idx))
    
    # Sort boundaries by position
    section_boundaries.sort(key=lambda x: x[1])
    
    # Add content to each section
    for i, (level, start_idx) in enumerate(section_boundaries):
        # Find the end of this section (next heading or end of document)
        end_idx = len(tokens)
        if i + 1 < len(section_boundaries):
            end_idx = section_boundaries[i + 1][1]
        
        # Get content blocks for this section
        content_blocks = extract_content_blocks(tokens, start_idx + 2, end_idx)  # +2 to skip heading_open and inline
        
        # Add to the appropriate section
        section_map[(level, start_idx)]["content"] = content_blocks
    
    return hierarchy

def markdown_to_json(markdown_text):
    """Convert markdown text to hierarchical JSON"""
    # Parse markdown using markdown-it-py with table plugin enabled
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(markdown_text)
    
    # Build section hierarchy
    hierarchy = build_section_hierarchy(tokens)
    
    # Create final result
    result = OrderedDict([
        ("document", OrderedDict([
            ("hierarchy", hierarchy)
        ]))
    ])
    
    return result

def count_tables_in_hierarchy(hierarchy):
    """Count tables in a section hierarchy"""
    table_count = 0
    
    # Check this section's content for tables
    for section in hierarchy.values():
        for block in section.get("content", []):
            if block.get("type") == "table":
                table_count += 1
        
        # Recursively check subsections
        if "subsections" in section and section["subsections"]:
            table_count += count_tables_in_hierarchy(section["subsections"])
    
    return table_count

def print_tables_in_hierarchy(hierarchy, indent=0):
    """Print information about tables in a section hierarchy"""
    for section_title, section in hierarchy.items():
        # Check this section's content for tables
        for block in section.get("content", []):
            if block.get("type") == "table":
                print(f"{'  ' * indent}Section '{section_title}' contains a table:")
                print(f"{'  ' * (indent+1)}Headers: {block.get('header', [])}")
                print(f"{'  ' * (indent+1)}Rows: {len(block.get('rows', []))}")
                if block.get("rows"):
                    print(f"{'  ' * (indent+1)}Sample row: {block.get('rows', [])[0]}")
                print()
        
        # Recursively check subsections
        if "subsections" in section and section["subsections"]:
            print_tables_in_hierarchy(section["subsections"], indent+1)

# Test with the ipc-interface.md file
if __name__ == "__main__":
    # Path to the ipc-interface.md file
    file_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/cpp-sample/docs/api/ipc-interface.md")
    
    # Check if the file exists
    if not file_path.exists():
        print(f"File not found: {file_path}")
        exit(1)
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert to JSON
    result = markdown_to_json(content)
    
    # Print information about tables
    print(f"Processing file: {file_path}")
    print(f"Number of sections: {len(result['document']['hierarchy'])}")
    
    # Count tables
    table_count = count_tables_in_hierarchy(result["document"]["hierarchy"])
    print(f"Total tables found: {table_count}")
    
    # Print details about tables
    print("\nTable details:")
    print_tables_in_hierarchy(result["document"]["hierarchy"])
    
    # Save the JSON to a file
    output_file = Path("ipc_interface_markdown_structure.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nJSON output saved to: {output_file}") 