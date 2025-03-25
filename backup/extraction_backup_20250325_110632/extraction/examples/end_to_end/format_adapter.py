#!/usr/bin/env python3
"""
Format Adapter for Documentation Extraction Validation.

This module provides functions to adapt extraction outputs to the expected
format for validation, ensuring consistent block types and relationships.
"""

import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("format_adapter")


def adapt_extraction_to_validation_format(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform extraction blocks to match validation expected format.
    
    Args:
        blocks: Original extraction blocks
        
    Returns:
        Transformed blocks with expected block types and relationships
    """
    # Return early if no blocks to process
    if not blocks:
        return []
        
    # Check if blocks is already in expected format
    doc_block_count = sum(1 for block in blocks if block.get("type") == "documentation")
    doc_page_count = sum(1 for block in blocks if block.get("type") == "doc_page")
    doc_section_count = sum(1 for block in blocks if block.get("type") == "doc_section")
    
    # If we already have documentation/doc_page/doc_section blocks, assume format is correct
    if doc_block_count > 0 and doc_page_count > 0 and doc_section_count > 0:
        return blocks
    
    # Create a simplified set of blocks with only the essential fields
    # This helps avoid infinite recursion due to circular references
    simplified_blocks = []
    for block in blocks:
        adapted_block = {
            "uuid": block.get("uuid", f"generated-uuid-{len(simplified_blocks)}"),
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "type": block.get("type", ""),
            "language": block.get("language", ""),
            "content": block.get("content", ""),
            "file_path": block.get("file_path", ""),
            "parent_uuid": block.get("parent_uuid", ""),
            "child_uuids": block.get("child_uuids", [])[:],  # Make a copy
            "metadata": block.get("metadata", {}).copy()
        }
        simplified_blocks.append(adapted_block)
    
    # Create mapping for UUID to block
    uuid_to_block = {block["uuid"]: block for block in simplified_blocks}
    
    # Process blocks to map types
    for block in simplified_blocks:
        block_type = block.get("type")
        
        # Map file blocks to documentation blocks
        if block_type == "file":
            if "markdown" in block.get("language", "").lower():
                block["type"] = "documentation"
                
                # Set source URL if not present
                if "source_url" not in block:
                    file_path = block.get("file_path", "")
                    if file_path:
                        block["source_url"] = f"file://{file_path}"
                
                # Add metadata if not present
                if "metadata" not in block:
                    block["metadata"] = {}
                
                block["metadata"]["doc_type"] = "markdown"
                
        # Map section blocks to doc_section blocks
        elif block_type == "section":
            block["type"] = "doc_section"
            
            # Add metadata if not present
            if "metadata" not in block:
                block["metadata"] = {}
                
            # Set section hierarchy if not present
            if "section_hierarchy" not in block["metadata"]:
                section_name = block.get("name", "")
                if section_name:
                    block["metadata"]["section_hierarchy"] = [section_name]
                    
            # Find parent and map to doc_page if parent is file
            parent_uuid = block.get("parent_uuid")
            if parent_uuid and parent_uuid in uuid_to_block:
                parent = uuid_to_block[parent_uuid]
                if parent.get("type") == "documentation":
                    # Create intermediate doc_page if needed
                    doc_page = {
                        "uuid": f"{parent_uuid}_page",
                        "id": f"{parent.get('id', '')}_page",
                        "name": f"{parent.get('name', '')} Page",
                        "type": "doc_page",
                        "language": parent.get("language", ""),
                        "content": f"Documentation page for {parent.get('name', '')}",
                        "file_path": parent.get("file_path", ""),
                        "parent_uuid": parent_uuid,
                        "child_uuids": [block["uuid"]],
                        "metadata": {
                            "language": parent.get("language", ""),
                            "doc_type": parent.get("metadata", {}).get("doc_type", "markdown"),
                            "source_url": parent.get("source_url", "")
                        }
                    }
                    
                    # Add doc_page to simplified blocks
                    simplified_blocks.append(doc_page)
                    
                    # Update parent's child_uuids
                    if parent_uuid in uuid_to_block:
                        parent = uuid_to_block[parent_uuid]
                        if "child_uuids" in parent:
                            # Replace the section with doc_page in parent's children
                            parent["child_uuids"] = [
                                doc_page["uuid"] if uuid == block["uuid"] else uuid 
                                for uuid in parent["child_uuids"]
                            ]
                    
                    # Update section's parent_uuid
                    block["parent_uuid"] = doc_page["uuid"]
        
        # Map text blocks to sub_section blocks
        elif block_type == "text":
            block["type"] = "sub_section"
    
    # Process child blocks (code blocks, tables, etc.)
    for block in simplified_blocks:
        # Check for tables in content
        content = block.get("content", "")
        child_uuids = block.get("child_uuids", [])
        
        # Check for table pattern in markdown content
        if "| --" in content or "|-" in content:
            # Create a table block
            table_uuid = f"{block.get('uuid', '')}_table"
            table_block = {
                "uuid": table_uuid,
                "id": f"{block.get('id', '')}_table",
                "name": f"Table in {block.get('name', '')}",
                "type": "table",
                "language": block.get("language", ""),
                "content": extract_table_content(content),
                "file_path": block.get("file_path", ""),
                "parent_uuid": block.get("uuid", ""),
                "metadata": {
                    "source": "markdown",
                    "language": block.get("language", "")
                }
            }
            
            # Add table to simplified blocks
            simplified_blocks.append(table_block)
            
            # Add table to parent's child_uuids if not already there
            if table_uuid not in child_uuids:
                child_uuids.append(table_uuid)
                block["child_uuids"] = child_uuids
        
        # Check for code blocks in content
        if "```" in content:
            # Create a code block
            code_uuid = f"{block.get('uuid', '')}_code"
            code_content, code_language = extract_code_content(content)
            
            if code_content:
                code_block = {
                    "uuid": code_uuid,
                    "id": f"{block.get('id', '')}_code",
                    "name": f"Code in {block.get('name', '')}",
                    "type": "code_block",
                    "language": code_language,
                    "content": code_content,
                    "file_path": block.get("file_path", ""),
                    "parent_uuid": block.get("uuid", ""),
                    "metadata": {
                        "source": "markdown",
                        "language": code_language
                    }
                }
                
                # Add code block to simplified blocks
                simplified_blocks.append(code_block)
                
                # Add code block to parent's child_uuids if not already there
                if code_uuid not in child_uuids:
                    child_uuids.append(code_uuid)
                    block["child_uuids"] = child_uuids
    
    return simplified_blocks


def extract_table_content(content: str) -> str:
    """Extract table content from markdown text."""
    table_lines = []
    capturing = False
    
    for line in content.split("\n"):
        if "|" in line:
            if not capturing:
                capturing = True
            table_lines.append(line.strip())
        elif capturing and not line.strip():
            break
    
    table_content = "\n".join(table_lines)
    
    # Create a simple dictionary format for the table content
    # This is a placeholder for more sophisticated table parsing
    return table_content


def extract_code_content(content: str) -> tuple:
    """Extract code content and language from markdown text."""
    code_blocks = content.split("```")
    
    if len(code_blocks) < 3:
        return "", "text"
    
    # Get the content after the first ```
    code_block = code_blocks[1]
    lines = code_block.split("\n")
    
    # First line might have the language
    language = "text"
    if lines and lines[0].strip():
        language = lines[0].strip()
        code_content = "\n".join(lines[1:])
    else:
        code_content = code_block
    
    return code_content.strip(), language


def main():
    """Main function for testing the format adapter."""
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 3:
        print("Usage: python format_adapter.py <input_file> <output_file>")
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
    adapted_blocks = adapt_extraction_to_validation_format(blocks)
    
    # Save output data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(adapted_blocks, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        sys.exit(1)
    
    logger.info(f"Successfully adapted blocks from {input_file} to {output_file}")
    logger.info(f"Adapted {len(blocks)} blocks to {len(adapted_blocks)} blocks")


if __name__ == "__main__":
    main()