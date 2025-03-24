#!/usr/bin/env python3
"""
Conversion Tool for Extraction Outputs.

This script converts raw extraction outputs to a format compatible with the 
validation framework requirements (deepseek format).
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("convert_for_validation")


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: Path) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def convert_to_validation_format(extraction_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert extraction data to the format expected by the validation framework.
    
    Args:
        extraction_data: The raw extraction data
        
    Returns:
        Converted data in deepseek format
    """
    deepseek_sections = []
    
    # First, identify key block types
    doc_blocks = [b for b in extraction_data if b.get("type") == "documentation"]
    doc_page_blocks = [b for b in extraction_data if b.get("type") == "doc_page"]
    section_blocks = [b for b in extraction_data if b.get("type") in ("section", "doc_section")]
    file_blocks = [b for b in extraction_data if b.get("type") == "file"]
    
    # Process hierarchically (documentation -> doc_page -> sections)
    processed_uuids = set()
    
    # Process documentation-docpage-section hierarchy
    for doc_block in doc_blocks:
        doc_uuid = doc_block.get("uuid", "")
        if doc_uuid in processed_uuids:
            continue
            
        processed_uuids.add(doc_uuid)
        child_uuids = doc_block.get("child_uuids", [])
        
        # Find child doc_pages
        for doc_page in doc_page_blocks:
            page_uuid = doc_page.get("uuid", "")
            if page_uuid in child_uuids:
                processed_uuids.add(page_uuid)
                
                # Find sections that are children of this doc_page
                page_child_uuids = doc_page.get("child_uuids", [])
                for section in section_blocks:
                    section_uuid = section.get("uuid", "")
                    if section_uuid in page_child_uuids:
                        deepseek_section = process_section(section, extraction_data)
                        if deepseek_section:
                            deepseek_sections.append(deepseek_section)
                            processed_uuids.add(section_uuid)
                
                # If the doc_page has no child sections, create a section from it
                if not any(u in processed_uuids for u in page_child_uuids):
                    deepseek_section = {
                        "uuid": page_uuid,
                        "title": doc_page.get("name", ""),
                        "content": doc_page.get("content", ""),
                        "section_hierarchy_depth": [doc_block.get("name", ""), doc_page.get("name", "")],
                        "images": [],
                        "tables": [],
                        "code": [],
                        "tests": []
                    }
                    extract_embedded_content(doc_page, deepseek_section)
                    deepseek_sections.append(deepseek_section)
    
    # Process standalone sections that weren't processed above
    for section in section_blocks:
        section_uuid = section.get("uuid", "")
        if section_uuid not in processed_uuids:
            deepseek_section = process_section(section, extraction_data)
            if deepseek_section:
                deepseek_sections.append(deepseek_section)
                processed_uuids.add(section_uuid)
    
    # If no sections found, create from file blocks
    if not deepseek_sections:
        for block in file_blocks:
            file_uuid = block.get("uuid", "")
            if file_uuid not in processed_uuids:
                deepseek_section = {
                    "uuid": file_uuid,
                    "title": block.get("name", ""),
                    "content": block.get("content", ""),
                    "section_hierarchy_depth": [block.get("name", "")],
                    "images": [],
                    "tables": [],
                    "code": [],
                    "tests": []
                }
                extract_embedded_content(block, deepseek_section)
                deepseek_sections.append(deepseek_section)
                processed_uuids.add(file_uuid)
    
    # Ensure all child blocks are accounted for by checking if there are any tables or code blocks
    # that weren't added to a section
    for block in extraction_data:
        block_uuid = block.get("uuid", "")
        parent_uuid = block.get("parent_uuid")
        block_type = block.get("type", "")
        
        if block_uuid not in processed_uuids and block_type in ("table", "code_block"):
            # Find parent section
            parent_section = None
            for section in deepseek_sections:
                if section["uuid"] == parent_uuid:
                    parent_section = section
                    break
            
            # If parent section found, add this block to it
            if parent_section:
                if block_type == "table":
                    table_content = parse_table_content(block.get("content", ""))
                    parent_section["tables"].append({
                        "uuid": block_uuid,
                        "content": table_content
                    })
                elif block_type == "code_block":
                    parent_section["code"].append({
                        "uuid": block_uuid,
                        "language": block.get("language", ""),
                        "content": block.get("content", "")
                    })
                processed_uuids.add(block_uuid)
    
    return deepseek_sections


def process_section(section: Dict[str, Any], all_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a section block into deepseek format, including its child elements."""
    section_uuid = section.get("uuid", "")
    parent_uuid = section.get("parent_uuid")
    
    # Determine section hierarchy
    section_hierarchy = section.get("metadata", {}).get("section_hierarchy", [])
    if not section_hierarchy:
        # Try to build hierarchy from parent chain
        hierarchy = [section.get("name", "")]
        current_parent = parent_uuid
        while current_parent:
            parent_block = next((b for b in all_blocks if b.get("uuid") == current_parent), None)
            if parent_block:
                parent_name = parent_block.get("name", "")
                if parent_name:
                    hierarchy.insert(0, parent_name)
                current_parent = parent_block.get("parent_uuid")
            else:
                break
        section_hierarchy = hierarchy
    
    # Create deepseek section
    deepseek_section = {
        "uuid": section_uuid,
        "title": section.get("name", ""),
        "content": section.get("content", ""),
        "section_hierarchy_depth": section_hierarchy,
        "images": [],
        "tables": [],
        "code": [],
        "tests": []
    }
    
    # Find and add child elements (tables, code blocks)
    child_uuids = section.get("child_uuids", [])
    for child_uuid in child_uuids:
        child_block = next((b for b in all_blocks if b.get("uuid") == child_uuid), None)
        if child_block:
            block_type = child_block.get("type", "")
            if block_type == "table":
                table_content = parse_table_content(child_block.get("content", ""))
                deepseek_section["tables"].append({
                    "uuid": child_uuid,
                    "content": table_content
                })
            elif block_type == "code_block":
                deepseek_section["code"].append({
                    "uuid": child_uuid,
                    "language": child_block.get("language", ""),
                    "content": child_block.get("content", "")
                })
    
    # Extract any embedded content from the section itself
    extract_embedded_content(section, deepseek_section)
    
    return deepseek_section


def extract_embedded_content(block: Dict[str, Any], deepseek_section: Dict[str, Any]) -> None:
    """Extract tables and code blocks from content text."""
    content = block.get("content", "")
    
    # Extract code blocks using markdown delimiters
    if "```" in content:
        code_blocks = content.split("```")
        for i in range(1, len(code_blocks), 2):
            if i < len(code_blocks):
                code_content = code_blocks[i]
                language = "text"
                if code_content and code_content.strip() and "\n" in code_content:
                    # Try to extract language from first line
                    first_line = code_content.split("\n")[0].strip()
                    if first_line:
                        language = first_line
                        code_content = "\n".join(code_content.split("\n")[1:])
                    
                    deepseek_section["code"].append({
                        "uuid": f"{block.get('uuid', '')}_code_{i}",
                        "language": language,
                        "content": code_content.strip()
                    })
    
    # Extract tables (basic markdown table detection)
    table_start = content.find("\n|")
    while table_start != -1:
        # Find the end of the table (next double newline or end of content)
        table_end = content.find("\n\n", table_start + 2)
        if table_end == -1:
            table_end = len(content)
        
        # Extract the table content
        table_content = content[table_start:table_end].strip()
        if table_content and "|" in table_content and "-|-" in table_content:
            # This looks like a markdown table
            try:
                # Basic parsing of markdown table
                rows = [line.strip() for line in table_content.split("\n") if line.strip()]
                if len(rows) >= 2:  # Need header row and separator row at minimum
                    # Parse header row
                    header_cells = [cell.strip() for cell in rows[0].split("|") if cell.strip()]
                    
                    # Skip separator row and parse data rows
                    data_rows = []
                    for row in rows[2:]:  # Skip header and separator
                        cells = [cell.strip() for cell in row.split("|") if cell.strip()]
                        if cells:
                            data_rows.append(cells)
                    
                    # Add table to deepseek section
                    if header_cells and data_rows:
                        deepseek_section["tables"].append({
                            "uuid": f"{block.get('uuid', '')}_table_{table_start}",
                            "content": {
                                "headers": header_cells,
                                "rows": data_rows
                            }
                        })
            except Exception as e:
                logger.warning(f"Failed to parse embedded table: {e}")
        
        # Find the next table
        table_start = content.find("\n|", table_end)


def parse_table_content(content: Any) -> Dict[str, Any]:
    """Parse table content into the expected format."""
    if not content:
        return {"headers": [], "rows": []}
    
    # If content is a string that looks like a dict, try to evaluate it
    if isinstance(content, str) and content.strip().startswith("{") and content.strip().endswith("}"):
        try:
            # Using json.loads is safer than eval
            import json
            content_dict = json.loads(content.replace("'", "\""))
            if isinstance(content_dict, dict):
                # Ensure required keys
                headers = content_dict.get("headers", [])
                rows = content_dict.get("rows", [])
                return {"headers": headers, "rows": rows}
        except Exception as e:
            logger.warning(f"Failed to parse table content as JSON: {e}")
    
    # If content is already a dict, ensure it has the right format
    if isinstance(content, dict):
        headers = content.get("headers", [])
        rows = content.get("rows", [])
        return {"headers": headers, "rows": rows}
    
    # Default empty structure
    return {"headers": [], "rows": []}


def main():
    """Main function for the conversion tool."""
    parser = argparse.ArgumentParser(description="Convert extraction outputs to validation format")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input extraction JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the converted output")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Load input data
    extraction_data = load_json_file(input_path)
    if not extraction_data:
        logger.error("Failed to load extraction data")
        sys.exit(1)
        
    # Convert to validation format
    converted_data = convert_to_validation_format(extraction_data)
    
    # Save converted data
    if not save_json_file(converted_data, output_path):
        logger.error("Failed to save converted data")
        sys.exit(1)
        
    logger.info(f"Successfully converted {input_path} to {output_path}")
    logger.info(f"Converted {len(extraction_data)} blocks to {len(converted_data)} sections")


if __name__ == "__main__":
    main()