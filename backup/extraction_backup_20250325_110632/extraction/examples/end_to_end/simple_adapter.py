#!/usr/bin/env python3
"""
Simple Format Adapter for Documentation Extraction Validation.

This module provides a simpler approach to adapt extraction outputs
to match expected validation format.
"""

import logging
from typing import Dict, Any, List
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_adapter")


def create_simple_validation_format(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a simple validation format directly from extracted blocks.
    
    Instead of adapting existing blocks, this creates a new validation-ready 
    structure based on extracted content.
    
    Args:
        blocks: Original extraction blocks
        
    Returns:
        Validation-compatible blocks
    """
    # Check if already in correct format
    doc_types = set(block.get("type", "") for block in blocks)
    if "documentation" in doc_types and "doc_page" in doc_types and "doc_section" in doc_types:
        return blocks
    
    # Group blocks by file_path
    blocks_by_file = {}
    for block in blocks:
        file_path = block.get("file_path", "unknown")
        if file_path not in blocks_by_file:
            blocks_by_file[file_path] = []
        blocks_by_file[file_path].append(block)
    
    # Create new validation-format blocks
    validation_blocks = []
    
    # For each file, create a documentation > doc_page > doc_section hierarchy
    for file_path, file_blocks in blocks_by_file.items():
        # Find "file" type blocks as documentation roots
        file_block = next((b for b in file_blocks if b.get("type") == "file"), None)
        
        # If no file block found, create a synthetic one
        if not file_block:
            # Use the first block as reference
            if file_blocks:
                ref_block = file_blocks[0]
                file_name = ref_block.get("file_path", "unknown").split("/")[-1]
                doc_uuid = f"doc-{len(validation_blocks)}"
                file_block = {
                    "uuid": doc_uuid,
                    "id": f"doc_{file_name}",
                    "name": f"Documentation: {file_name}",
                    "type": "documentation",
                    "language": ref_block.get("language", "text"),
                    "content": f"Documentation source: {file_path}",
                    "file_path": file_path,
                    "source_url": f"file://{file_path}",
                    "child_uuids": [],
                    "metadata": {
                        "language": ref_block.get("language", "text"),
                        "source_url": f"file://{file_path}",
                        "doc_type": "generic"
                    }
                }
        
        # If file block but not documentation type, convert it
        elif file_block.get("type") != "documentation":
            doc_uuid = file_block.get("uuid", f"doc-{len(validation_blocks)}")
            file_block = {
                "uuid": doc_uuid,
                "id": file_block.get("id", f"doc_{file_path.split('/')[-1]}"),
                "name": f"Documentation: {file_block.get('name', file_path.split('/')[-1])}",
                "type": "documentation",
                "language": file_block.get("language", "text"),
                "content": file_block.get("content", f"Documentation source: {file_path}"),
                "file_path": file_path,
                "source_url": file_block.get("source_url", f"file://{file_path}"),
                "child_uuids": file_block.get("child_uuids", []).copy(),
                "metadata": {
                    "language": file_block.get("language", "text"),
                    "source_url": file_block.get("source_url", f"file://{file_path}"),
                    "doc_type": "generic"
                }
            }
            
            # If language is markdown, set doc_type to markdown
            if "markdown" in file_block["language"].lower():
                file_block["metadata"]["doc_type"] = "markdown"
        
        # Add documentation block to validation blocks
        validation_blocks.append(file_block)
        
        # Create doc_page block
        page_uuid = f"{file_block['uuid']}-page"
        page_block = {
            "uuid": page_uuid,
            "id": f"{file_block['id']}_page",
            "name": f"{file_block['name']} Page",
            "type": "doc_page",
            "language": file_block["language"],
            "content": f"Documentation page from {file_path}",
            "file_path": file_path,
            "parent_uuid": file_block["uuid"],
            "child_uuids": [],
            "metadata": {
                "language": file_block["language"],
                "doc_type": file_block["metadata"]["doc_type"],
                "relative_path": file_path
            }
        }
        
        # Add doc_page to validation blocks
        validation_blocks.append(page_block)
        
        # Update documentation block's child_uuids to include doc_page
        file_block["child_uuids"] = [page_uuid]
        
        # Find section blocks and convert to doc_section
        section_blocks = [b for b in file_blocks if b.get("type") == "section"]
        for section in section_blocks:
            section_uuid = section.get("uuid", f"section-{len(validation_blocks)}")
            doc_section = {
                "uuid": section_uuid,
                "id": section.get("id", f"section_{section_uuid}"),
                "name": section.get("name", "Unnamed Section"),
                "type": "doc_section",
                "language": section.get("language", file_block["language"]),
                "content": section.get("content", ""),
                "file_path": file_path,
                "parent_uuid": page_uuid,
                "child_uuids": section.get("child_uuids", []).copy(),
                "metadata": {
                    "language": section.get("language", file_block["language"]),
                    "level": section.get("metadata", {}).get("level", 1),
                    "section_hierarchy": section.get("metadata", {}).get("section_hierarchy", [section.get("name", "Unnamed Section")])
                }
            }
            
            # Add doc_section to validation blocks
            validation_blocks.append(doc_section)
            
            # Add section to page's child_uuids
            page_block["child_uuids"].append(section_uuid)
            
            # Process child blocks (tables, code blocks)
            for child_uuid in section.get("child_uuids", []):
                child = next((b for b in file_blocks if b.get("uuid") == child_uuid), None)
                if child:
                    child_type = child.get("type", "")
                    
                    # Convert text blocks to sub_sections
                    if child_type == "text":
                        sub_section = {
                            "uuid": child.get("uuid", f"subsection-{len(validation_blocks)}"),
                            "id": child.get("id", f"subsection_{child_uuid}"),
                            "name": child.get("name", "Unnamed Subsection"),
                            "type": "sub_section",
                            "language": child.get("language", section["language"]),
                            "content": child.get("content", ""),
                            "file_path": file_path,
                            "parent_uuid": section_uuid,
                            "metadata": {
                                "language": child.get("language", section["language"]),
                                "is_embedded": child.get("metadata", {}).get("is_embedded", True)
                            }
                        }
                        validation_blocks.append(sub_section)
                    
                    # Create code blocks from content
                    if "```" in child.get("content", "") or section.get("content", ""):
                        code_uuid = f"{section_uuid}_code"
                        code_content = extract_code(child.get("content", "") or section.get("content", ""))
                        if code_content:
                            code_block = {
                                "uuid": code_uuid,
                                "id": f"{section['id']}_code",
                                "name": f"Code in {section['name']}",
                                "type": "code_block",
                                "language": "text",
                                "content": code_content,
                                "file_path": file_path,
                                "parent_uuid": section_uuid,
                                "metadata": {
                                    "source": "markdown",
                                    "language": "text"
                                }
                            }
                            validation_blocks.append(code_block)
                            
                            # Add code block to section's child_uuids if not already there
                            if code_uuid not in section.get("child_uuids", []):
                                section["child_uuids"].append(code_uuid)
                    
                    # Create table blocks from content
                    if "|" in child.get("content", "") or "|" in section.get("content", ""):
                        table_uuid = f"{section_uuid}_table"
                        table_content = extract_table(child.get("content", "") or section.get("content", ""))
                        if table_content:
                            table_block = {
                                "uuid": table_uuid,
                                "id": f"{section['id']}_table",
                                "name": f"Table in {section['name']}",
                                "type": "table",
                                "language": "text",
                                "content": table_content,
                                "file_path": file_path,
                                "parent_uuid": section_uuid,
                                "metadata": {
                                    "source": "markdown",
                                    "language": "text"
                                }
                            }
                            validation_blocks.append(table_block)
                            
                            # Add table to section's child_uuids if not already there
                            if table_uuid not in section.get("child_uuids", []):
                                section["child_uuids"].append(table_uuid)
    
    return validation_blocks


def extract_code(content: str) -> str:
    """Extract code from markdown content."""
    if "```" not in content:
        return ""
    
    # Simple extraction of first code block
    parts = content.split("```")
    if len(parts) < 3:
        return ""
    
    return parts[1].strip()


def extract_table(content: str) -> str:
    """Extract table from markdown content."""
    if "|" not in content:
        return ""
    
    # Find table lines
    table_lines = []
    in_table = False
    
    for line in content.split("\n"):
        if "|" in line:
            if not in_table:
                in_table = True
            table_lines.append(line.strip())
        elif in_table and not line.strip():
            break
    
    if not table_lines:
        return ""
    
    return "\n".join(table_lines)


def main():
    """Main function for testing."""
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python simple_adapter.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    # Load input data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Adapt blocks
    adapted_blocks = create_simple_validation_format(blocks)
    
    # Save output data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(adapted_blocks, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        sys.exit(1)
    
    logger.info(f"Successfully adapted blocks from {input_file} to {output_file}")
    logger.info(f"Original: {len(blocks)} blocks, Adapted: {len(adapted_blocks)} blocks")


if __name__ == "__main__":
    main()