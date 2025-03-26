#!/usr/bin/env python3
"""
QA Formatter Module for DuaLipa Extraction.

This module transforms the extracted blocks into a format suitable for QA generation.
It organizes sections hierarchically, groups elements by type, and creates relationships
between blocks to maintain proper content context.

Key Functions:
- create_qa_compatible_blocks: Convert extracted blocks to QA-compatible format
- create_qa_compatible_output: Create final output with metadata
- generate_section_relationships: Establish relationships between sections
- validate_and_fix_parent_references: Ensure valid parent-child relationships

Dependencies:
- uuid: For UUID generation (https://docs.python.org/3/library/uuid.html)
- re: For regular expression handling (https://docs.python.org/3/library/re.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)
"""

import uuid
import re
import logging
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Setup logging
logger = logging.getLogger("extraction.qa_formatter")


def create_qa_compatible_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert extracted blocks to QA-compatible format.
    
    This function takes the raw extracted blocks and transforms them into a format
    suitable for QA generation by organizing sections hierarchically and grouping
    elements by type within their parent sections.
    
    Args:
        blocks: List of extracted blocks
        
    Returns:
        List of QA-compatible blocks
    """
    # First, separate blocks by type
    file_blocks = [b for b in blocks if b.get("type") == "file"]
    section_blocks = [b for b in blocks if b.get("type") == "section"]
    code_blocks = [b for b in blocks if b.get("type") == "code_block"]
    table_blocks = [b for b in blocks if b.get("type") == "table"]
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    
    # Establish relationships between sections based on hierarchy
    generate_section_relationships(section_blocks)
    
    # Fix any invalid parent references
    validate_and_fix_parent_references(blocks)
    
    # Build a map from UUID to block for easy lookup
    block_map = {block["uuid"]: block for block in blocks}
    
    # Create QA-compatible blocks
    qa_blocks = []
    
    # First, let's create a function to ensure all required fields are present
    def enhance_block_for_qa(block):
        enhanced = block.copy()
        
        # Add required fields for QA module
        if "uuid" not in enhanced:
            enhanced["uuid"] = str(uuid.uuid4())
        if "extraction_focus" not in enhanced:
            if enhanced.get("type") in ["section", "text"]:
                enhanced["extraction_focus"] = ["documentation"]
            else:
                enhanced["extraction_focus"] = ["code"]
        if "summary_instructions" not in enhanced:
            enhanced["summary_instructions"] = "Extract key points from content"
        if "breadcrumb" not in enhanced:
            # Create breadcrumb based on file path and name
            file_path = enhanced.get("file_path", "")
            name = enhanced.get("name", "")
            if file_path:
                file_name = file_path.split("/")[-1] if "/" in file_path else file_path
                enhanced["breadcrumb"] = [file_name]
                if name:
                    enhanced["breadcrumb"].append(name)
            else:
                enhanced["breadcrumb"] = [name or "Unnamed Block"]
        if "parent_uuid" not in enhanced:
            enhanced["parent_uuid"] = None
        if "child_uuids" not in enhanced:
            enhanced["child_uuids"] = []
        
        return enhanced
    
    # Process all blocks - we want to include everything for our QA module
    # First, add all function, class, and method blocks with required fields
    code_element_blocks = [enhance_block_for_qa(b) for b in blocks if b.get("type") in ["function", "class", "method"]]
    qa_blocks.extend(code_element_blocks)
    
    # Process each file separately
    for file_block in file_blocks:
        # Include the file block itself with required fields
        qa_blocks.append(enhance_block_for_qa(file_block))
        
        file_path = file_block.get("file_path", "unknown")
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        
        # Process markdown files with special formatting for DeepSeek format
        if file_name.lower().endswith(".md") and "deepseek.md" in file_path:
            # Special handling for deepseek.md
            md_blocks = convert_to_deepseek_format(file_block, block_map)
            # Add required fields to each md block
            enhanced_md_blocks = [enhance_block_for_qa(b) for b in md_blocks]
            qa_blocks.extend(enhanced_md_blocks)
        else:
            # Get top-level sections for this file
            file_sections = [b for b in section_blocks if b.get("parent_uuid") == file_block["uuid"]]
            
            # Process each section and its sub-sections
            for section in file_sections:
                process_section_and_children(section, block_map, qa_blocks)
                
    # Add other blocks that weren't included above (code_blocks, tables, images, texts)
    other_blocks = [enhance_block_for_qa(b) for b in blocks if b.get("type") in ["code_block", "table", "image", "text"]]
    qa_blocks.extend(other_blocks)
    
    return qa_blocks


def convert_to_deepseek_format(file_block: Dict[str, Any], block_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert blocks to the special DeepSeek format based on the example JSON.
    
    Args:
        file_block: The file block for deepseek.md
        block_map: Map from UUID to block for lookup
        
    Returns:
        List of blocks in DeepSeek format
    """
    # Get all sections related to this file
    section_blocks = [b for b in block_map.values() if b.get("type") == "section" and is_descendant_of(b, file_block["uuid"], block_map)]
    
    # Build a mapping of section UUID to its children by type
    section_children = defaultdict(lambda: defaultdict(list))
    
    # Find all elements and associate them with their parent section
    for block_uuid, block in block_map.items():
        if block.get("type") in ["code_block", "table", "image", "text"]:
            parent_uuid = block.get("parent_uuid")
            if parent_uuid and parent_uuid in [s["uuid"] for s in section_blocks]:
                section_children[parent_uuid][block.get("type")].append(block)
    
    # Create hierarchical section mapping
    section_hierarchy = {}
    for section in section_blocks:
        build_section_hierarchy(section, block_map, section_hierarchy)
    
    # Create DeepSeek-format blocks
    deepseek_blocks = []
    
    # Process each section
    for section in section_blocks:
        section_uuid = section.get("uuid")
        section_title = section.get("name", "Untitled Section")
        section_content = section.get("pure_content", "") or section.get("content", "")
        
        # Only use the text blocks as main content if they exist
        text_blocks = section_children[section_uuid].get("text", [])
        if text_blocks:
            # Combine text blocks into one content, sorted by position
            sorted_texts = sorted(text_blocks, key=lambda b: b.get("metadata", {}).get("position", float('inf')))
            section_content = "\n\n".join([b.get("content", "") for b in sorted_texts])
        
        # Get hierarchy path
        hierarchy_path = section.get("metadata", {}).get("section_hierarchy", [section_title])
        
        # Create DeepSeek-format block
        deepseek_block = {
            "uuid": section_uuid,
            "section_hierarchy_depth": hierarchy_path,
            "title": section_title,
            "content": section_content,
            "images": [],
            "tests": [],
            "tables": [],
            "code": []
        }
        
        # Add images - including both standard markdown and HTML-wrapped images
        for image_block in section_children[section_uuid].get("image", []):
            metadata = image_block.get("metadata", {})
            image_url = metadata.get("image_url", "")
            alt_text = metadata.get("alt_text", "")
            # Include the image if we have a valid URL
            if image_url:
                deepseek_block["images"].append({
                    "uuid": image_block.get("uuid", str(uuid.uuid4())),
                    "src": image_url,
                    "alt": alt_text
                })
        
        # Add tables
        for table_block in section_children[section_uuid].get("table", []):
            table_content = table_block.get("content", "")
            if table_content:
                # Parse table content to extract headers and rows
                table_rows = table_content.strip().split('\n')
                if len(table_rows) >= 2:  # Need at least header and separator rows
                    headers = [h.strip() for h in table_rows[0].split('|') if h.strip()]
                    # Skip separator row and get data rows
                    data_rows = []
                    for row in table_rows[2:]:  # Skip header and separator
                        if row and '|' in row:
                            data_rows.append([cell.strip() for cell in row.split('|') if cell.strip()])
                    
                    deepseek_block["tables"].append({
                        "uuid": table_block.get("uuid", str(uuid.uuid4())),
                        "content": {
                            "headers": headers,
                            "rows": data_rows
                        }
                    })
        
        # Add code blocks
        for code_block in section_children[section_uuid].get("code_block", []):
            code_content = code_block.get("content", "")
            code_language = code_block.get("language", "text")
            if code_content:
                deepseek_block["code"].append({
                    "uuid": code_block.get("uuid", str(uuid.uuid4())),
                    "language": code_language,
                    "content": code_content
                })
        
        deepseek_blocks.append(deepseek_block)
    
    return deepseek_blocks


def is_descendant_of(block: Dict[str, Any], ancestor_uuid: str, block_map: Dict[str, Dict[str, Any]]) -> bool:
    """Check if a block is a descendant of the given ancestor."""
    current_uuid = block.get("parent_uuid")
    while current_uuid:
        if current_uuid == ancestor_uuid:
            return True
        current_block = block_map.get(current_uuid)
        if not current_block:
            break
        current_uuid = current_block.get("parent_uuid")
    return False


def build_section_hierarchy(section: Dict[str, Any], block_map: Dict[str, Dict[str, Any]], hierarchy_dict: Dict[str, List[str]]) -> List[str]:
    """Build the hierarchy path for a section."""
    section_uuid = section.get("uuid")
    section_title = section.get("name", "Untitled Section")
    
    # If already computed, return it
    if section_uuid in hierarchy_dict:
        return hierarchy_dict[section_uuid]
    
    # Get parent UUID
    parent_uuid = section.get("parent_uuid")
    
    # If parent is a file or no parent, this is a top-level section
    parent_block = block_map.get(parent_uuid)
    if not parent_block or parent_block.get("type") == "file":
        hierarchy_dict[section_uuid] = [section_title]
        return [section_title]
    
    # Get parent's hierarchy and add this section
    parent_hierarchy = build_section_hierarchy(parent_block, block_map, hierarchy_dict)
    hierarchy_dict[section_uuid] = parent_hierarchy + [section_title]
    return hierarchy_dict[section_uuid]


def process_section_and_children(section: Dict[str, Any], block_map: Dict[str, Dict[str, Any]], qa_blocks: List[Dict[str, Any]]):
    """Process a section and all its child sections recursively."""
    # Create initial QA block for this section
    qa_block = {
        "uuid": section["uuid"],
        "id": section.get("id", f"section_{len(qa_blocks)}"),
        "type": "section",
        "name": section.get("name", "Untitled Section"),
        "content": section.get("content", ""),
        "metadata": section.get("metadata", {}),
        "elements": {
            "text": [],
            "code_blocks": [],
            "tables": [],
            "images": []
        }
    }
    
    # Add hierarchy information if available
    if "metadata" in section and "section_hierarchy" in section["metadata"]:
        qa_block["hierarchy"] = section["metadata"]["section_hierarchy"]
        qa_block["breadcrumb"] = section["metadata"]["section_hierarchy"]
    elif "breadcrumb" not in qa_block:
        qa_block["breadcrumb"] = [qa_block.get("name", "Untitled Section")]
    
    # Add required fields for QA module
    qa_block["extraction_focus"] = ["documentation"]
    qa_block["summary_instructions"] = "Extract key points from section content"
    if "parent_uuid" not in qa_block:
        qa_block["parent_uuid"] = section.get("parent_uuid")
    if "child_uuids" not in qa_block:
        qa_block["child_uuids"] = section.get("child_uuids", [])
    
    # Find all child elements of this section
    for block_uuid, block in block_map.items():
        if block.get("parent_uuid") == section["uuid"]:
            if block.get("type") == "text":
                qa_block["elements"]["text"].append({
                    "uuid": block["uuid"],
                    "content": block.get("content", ""),
                    "position": block.get("metadata", {}).get("position", 0)
                })
            elif block.get("type") == "code_block":
                qa_block["elements"]["code_blocks"].append({
                    "uuid": block["uuid"],
                    "language": block.get("language", "text"),
                    "content": block.get("content", ""),
                    "position": block.get("metadata", {}).get("position", 0)
                })
            elif block.get("type") == "table":
                qa_block["elements"]["tables"].append({
                    "uuid": block["uuid"],
                    "content": block.get("content", ""),
                    "position": block.get("metadata", {}).get("position", 0)
                })
            elif block.get("type") == "image":
                qa_block["elements"]["images"].append({
                    "uuid": block["uuid"],
                    "alt_text": block.get("metadata", {}).get("alt_text", ""),
                    "image_url": block.get("metadata", {}).get("image_url", ""),
                    "position": block.get("metadata", {}).get("position", 0)
                })
    
    # Sort elements by position
    for element_type in qa_block["elements"]:
        qa_block["elements"][element_type].sort(key=lambda e: e.get("position", 0))
    
    # Add the section block to results
    qa_blocks.append(qa_block)
    
    # Find and process child sections
    child_sections = [b for b in block_map.values() 
                     if b.get("type") == "section" and b.get("parent_uuid") == section["uuid"]]
    
    for child_section in child_sections:
        process_section_and_children(child_section, block_map, qa_blocks)


def generate_section_relationships(sections: List[Dict[str, Any]]) -> None:
    """
    Establish relationships between sections based on hierarchy.
    
    This function analyzes the sections' hierarchy levels and establishes
    parent-child relationships between them based on their nesting level.
    
    Args:
        sections: List of section blocks
    """
    # Sort sections by position to ensure we process them in document order
    sections.sort(key=lambda s: s.get("metadata", {}).get("position", float('inf')))
    
    # Create a stack of active sections by level
    active_sections = {}  # level -> section_uuid
    
    for section in sections:
        level = section.get("metadata", {}).get("level", 1)
        
        # Find the parent section (closest lower level)
        parent_found = False
        for parent_level in range(level - 1, 0, -1):
            if parent_level in active_sections:
                # Found a potential parent
                parent_uuid = active_sections[parent_level]
                # Only update if not already set or currently set to file
                if "parent_uuid" not in section or not section.get("parent_uuid"):
                    section["parent_uuid"] = parent_uuid
                parent_found = True
                break
        
        # Update active sections for this level
        active_sections[level] = section["uuid"]
        
        # Clear any higher levels since they're no longer active
        # (this section becomes the new parent for subsequent sections at higher levels)
        for l in list(active_sections.keys()):
            if l > level:
                del active_sections[l]


def validate_and_fix_parent_references(blocks: List[Dict[str, Any]]) -> None:
    """
    Ensure all parent-child relationships are valid.
    
    This function checks that each block's parent UUID refers to a valid
    block and fixes any issues by finding alternative parents or making
    the block a top-level element.
    
    Args:
        blocks: List of blocks to validate
    """
    # Create a set of all UUIDs for quick lookup
    all_uuids = {block["uuid"] for block in blocks}
    
    # Create a map from UUID to block for easier updates
    uuid_to_block = {block["uuid"]: block for block in blocks}
    
    # Track invalid and fixed references
    invalid_refs = 0
    fixed_refs = 0
    
    # Check each block's parent reference
    for block in blocks:
        if "parent_uuid" in block and block["parent_uuid"]:
            parent_uuid = block["parent_uuid"]
            
            if parent_uuid not in all_uuids:
                invalid_refs += 1
                logger.warning(f"Block {block.get('name')} references non-existent parent UUID: {parent_uuid}")
                
                # Try to find another parent based on section hierarchy
                fixed = False
                
                if "metadata" in block and "section_hierarchy" in block["metadata"]:
                    hierarchy = block["metadata"]["section_hierarchy"]
                    if len(hierarchy) > 1:
                        # Try to find a section with matching hierarchy
                        parent_hierarchy = hierarchy[:-1]
                        for potential_parent in blocks:
                            if (potential_parent.get("type") == "section" and 
                                    "metadata" in potential_parent and 
                                    "section_hierarchy" in potential_parent["metadata"] and
                                    potential_parent["metadata"]["section_hierarchy"] == parent_hierarchy):
                                block["parent_uuid"] = potential_parent["uuid"]
                                
                                # Add this block to the parent's children if needed
                                if "child_uuids" not in potential_parent:
                                    potential_parent["child_uuids"] = []
                                if block["uuid"] not in potential_parent["child_uuids"]:
                                    potential_parent["child_uuids"].append(block["uuid"])
                                
                                fixed = True
                                fixed_refs += 1
                                break
                
                # If still not fixed, try to find file block
                if not fixed:
                    file_path = block.get("file_path", "")
                    if file_path:
                        for potential_parent in blocks:
                            if (potential_parent.get("type") == "file" and 
                                    potential_parent.get("file_path") == file_path):
                                block["parent_uuid"] = potential_parent["uuid"]
                                
                                # Add this block to the parent's children if needed
                                if "child_uuids" not in potential_parent:
                                    potential_parent["child_uuids"] = []
                                if block["uuid"] not in potential_parent["child_uuids"]:
                                    potential_parent["child_uuids"].append(block["uuid"])
                                
                                fixed = True
                                fixed_refs += 1
                                break
    
    # Now check child references
    invalid_child_refs = 0
    fixed_child_refs = 0
    
    for block in blocks:
        if "child_uuids" in block:
            valid_children = []
            for child_uuid in block["child_uuids"]:
                if child_uuid in all_uuids:
                    valid_children.append(child_uuid)
                    
                    # Make sure the child's parent_uuid is set correctly
                    child_block = uuid_to_block.get(child_uuid)
                    if child_block and child_block.get("parent_uuid") != block["uuid"]:
                        child_block["parent_uuid"] = block["uuid"]
                        fixed_child_refs += 1
                else:
                    invalid_child_refs += 1
            
            # Update with only valid children
            block["child_uuids"] = valid_children
    
    logger.info(f"Fixed {fixed_refs} invalid parent references and {fixed_child_refs} invalid child references")


def create_qa_compatible_output(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create the final QA-compatible output with metadata.
    
    Args:
        blocks: List of QA-compatible blocks
        
    Returns:
        Dictionary with blocks and metadata
    """
    # Build statistics
    block_types = {}
    languages = {}
    file_count = 0
    
    for block in blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
        
        if block_type == "file":
            file_count += 1
        
        language = block.get("language", "unknown")
        languages[language] = languages.get(language, 0) + 1
    
    # Generate metadata
    metadata = {
        "statistics": {
            "total_blocks": len(blocks),
            "total_files": file_count,
            "block_types": block_types,
            "languages": languages
        }
    }
    
    # Special handling for deepseek.md
    deepseek_blocks = [b for b in blocks if isinstance(b, dict) and 
                      b.get("section_hierarchy_depth") is not None]
    
    if deepseek_blocks:
        # Return just the deepseek blocks in their special format
        return deepseek_blocks
    
    # Add required fields to each block for QA module compatibility
    enhanced_blocks = []
    for block in blocks:
        # Create a copy to avoid modifying the original
        enhanced_block = block.copy()
        
        # Ensure all blocks have required fields
        if "uuid" not in enhanced_block:
            enhanced_block["uuid"] = str(uuid.uuid4())
        if "extraction_focus" not in enhanced_block:
            enhanced_block["extraction_focus"] = ["documentation" if enhanced_block.get("type") in ["section", "text"] else "code"]
        if "summary_instructions" not in enhanced_block:
            enhanced_block["summary_instructions"] = "Extract key points from content"
        if "breadcrumb" not in enhanced_block:
            enhanced_block["breadcrumb"] = [enhanced_block.get("name", "Unnamed Block")]
        if "parent_uuid" not in enhanced_block:
            enhanced_block["parent_uuid"] = None
        if "child_uuids" not in enhanced_block:
            enhanced_block["child_uuids"] = []
            
        # Ensure language field is present (this is required by QA module)
        if "language" not in enhanced_block:
            # Try to determine language from metadata or file extension
            if "metadata" in enhanced_block and "language" in enhanced_block["metadata"]:
                enhanced_block["language"] = enhanced_block["metadata"]["language"]
            elif enhanced_block.get("type") in ["section", "text"]:
                enhanced_block["language"] = "markdown"
            elif enhanced_block.get("type") == "file":
                file_path = enhanced_block.get("file_path", "")
                if file_path.endswith(".md"):
                    enhanced_block["language"] = "markdown"
                elif file_path.endswith((".py", ".pyx", ".pyw")):
                    enhanced_block["language"] = "python"
                elif file_path.endswith((".js", ".jsx")):
                    enhanced_block["language"] = "javascript"
                elif file_path.endswith((".ts", ".tsx")):
                    enhanced_block["language"] = "typescript"
                else:
                    enhanced_block["language"] = "text"
            else:
                enhanced_block["language"] = "text"
            
        enhanced_blocks.append(enhanced_block)
    
    # Build section relationships
    section_relationships = {
        "parent_child": {},
        "imports": {},
        "inheritance": {}
    }
    
    # Create relationship structures
    for block in enhanced_blocks:
        block_uuid = block.get("uuid")
        parent_uuid = block.get("parent_uuid")
        child_uuids = block.get("child_uuids", [])
        
        # Add to parent-child relationships
        if block_uuid:
            section_relationships["parent_child"][block_uuid] = {
                "parent": parent_uuid,
                "children": child_uuids
            }
    
    # Add model information to metadata
    metadata["model_used"] = "dualipa-extraction"
    metadata["timestamp"] = str(datetime.now().isoformat())
    metadata["version"] = "1.0"
    
    # Return standard format with all blocks as sections and metadata
    return {
        "sections": enhanced_blocks,
        "section_relationships": section_relationships,
        "extraction_metadata": metadata
    }