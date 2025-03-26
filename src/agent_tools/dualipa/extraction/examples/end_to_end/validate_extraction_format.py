#!/usr/bin/env python3
"""
validate_extraction_format.py

This script validates the format of extraction blocks to ensure they follow
the expected schema for the DuaLipa extraction system.

Usage:
    python validate_extraction_format.py path/to/extraction_blocks.json

Example:
    python validate_extraction_format.py test_results/arangodb_blocks.json
"""

import sys
import json
import os
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union


def validate_block(block: Dict[str, Any], block_idx: int) -> List[str]:
    """
    Validate a single extraction block.
    
    Args:
        block: The block to validate
        block_idx: The index of the block in the array for error reporting
        
    Returns:
        A list of error messages, empty if the block is valid
    """
    errors = []
    
    # Check required fields
    required_fields = ["uuid", "title", "content"]
    for field in required_fields:
        if field not in block:
            errors.append(f"Block {block_idx} is missing required field: {field}")
    
    # Special checks for DeepSeek blocks
    if "section_hierarchy_depth" in block:
        if not isinstance(block["section_hierarchy_depth"], list):
            errors.append(f"Block {block_idx} has section_hierarchy_depth but it's not a list")
    
    # For tables, check if they have the right structure
    if "tables" in block and block["tables"]:
        for i, table in enumerate(block["tables"]):
            if not isinstance(table, dict):
                errors.append(f"Block {block_idx}, table {i} is not a dictionary")
                continue
                
            if "uuid" not in table:
                errors.append(f"Block {block_idx}, table {i} is missing uuid")
                
            if "content" not in table:
                errors.append(f"Block {block_idx}, table {i} is missing content")
            elif not isinstance(table["content"], dict):
                errors.append(f"Block {block_idx}, table {i} content is not a dictionary")
            else:
                if "headers" not in table["content"]:
                    errors.append(f"Block {block_idx}, table {i} content is missing headers")
                if "rows" not in table["content"]:
                    errors.append(f"Block {block_idx}, table {i} content is missing rows")
    
    # For code blocks, check if they have the right structure
    if "code" in block and block["code"]:
        for i, code in enumerate(block["code"]):
            if not isinstance(code, dict):
                errors.append(f"Block {block_idx}, code {i} is not a dictionary")
                continue
                
            if "uuid" not in code:
                errors.append(f"Block {block_idx}, code {i} is missing uuid")
                
            if "language" not in code:
                errors.append(f"Block {block_idx}, code {i} is missing language")
                
            if "content" not in code:
                errors.append(f"Block {block_idx}, code {i} is missing content")
    
    # For images, check if they have the right structure
    if "images" in block and block["images"]:
        for i, image in enumerate(block["images"]):
            if not isinstance(image, dict):
                errors.append(f"Block {block_idx}, image {i} is not a dictionary")
                continue
                
            if "uuid" not in image:
                errors.append(f"Block {block_idx}, image {i} is missing uuid")
                
            if "src" not in image:
                errors.append(f"Block {block_idx}, image {i} is missing src")
                
            if "alt" not in image:
                errors.append(f"Block {block_idx}, image {i} is missing alt")
    
    return errors


def validate_extraction_output(blocks: List[Dict[str, Any]], expected_format: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extraction output blocks against an expected format template.
    
    Args:
        blocks: A list of extraction blocks (dictionaries) to validate
        expected_format: A template/reference format to validate against
        
    Returns:
        Dictionary with validation results containing:
        - valid: Boolean indicating if validation passed
        - errors: List of error messages if validation failed
        - stats: Statistics about the blocks like counts by type, language, etc.
    """
    # Normalize blocks - handle both list and dict with blocks key
    if isinstance(blocks, dict) and "blocks" in blocks:
        blocks_list = blocks["blocks"]
    elif isinstance(blocks, list):
        blocks_list = blocks
    else:
        return {"valid": False, "errors": ["Input is neither a list of blocks nor a dictionary with 'blocks' key"], "stats": {}}
    
    errors = []
    stats = {
        "total_blocks": len(blocks_list),
        "block_types": {},
        "section_hierarchies": {},
    }
    
    # Verify blocks have consistent format
    if len(blocks_list) == 0:
        errors.append("No blocks found in input")
    else:
        # Check for DeepSeek format (list of blocks with section_hierarchy_depth)
        if any("section_hierarchy_depth" in block for block in blocks_list):
            # Count hierarchies
            for block in blocks_list:
                if "section_hierarchy_depth" in block:
                    hierarchy_level = len(block["section_hierarchy_depth"])
                    stats["section_hierarchies"][hierarchy_level] = stats["section_hierarchies"].get(hierarchy_level, 0) + 1
            
            # Validate each block
            for idx, block in enumerate(blocks_list):
                block_errors = validate_block(block, idx)
                errors.extend(block_errors)
        else:
            errors.append("Blocks do not have DeepSeek format (missing section_hierarchy_depth)")
    
    # Verify expected format
    if expected_format:
        # Expected format is a list of blocks with section_hierarchy_depth
        if isinstance(expected_format, list) and all("section_hierarchy_depth" in block for block in expected_format):
            # Count how many blocks by level of hierarchy in expected format
            expected_hierarchies = {}
            for block in expected_format:
                hierarchy_level = len(block["section_hierarchy_depth"])
                expected_hierarchies[hierarchy_level] = expected_hierarchies.get(hierarchy_level, 0) + 1
            
            # Compare actual vs expected hierarchy counts
            actual_hierarchies = stats["section_hierarchies"]
            for level, count in expected_hierarchies.items():
                actual_count = actual_hierarchies.get(level, 0)
                if actual_count < count:
                    errors.append(f"Too few blocks with hierarchy level {level}: {actual_count} (expected at least {count})")
        
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": stats
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate extraction blocks format")
    parser.add_argument("blocks_file", help="Path to the JSON file containing extraction blocks")
    args = parser.parse_args()
    
    blocks_file = Path(args.blocks_file)
    if not blocks_file.exists():
        print(f"Error: File not found: {blocks_file}")
        sys.exit(1)
    
    # Load blocks
    try:
        with open(blocks_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
    except Exception as e:
        print(f"Error loading blocks file: {e}")
        sys.exit(1)
    
    # Load expected format
    expected_format_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/samples/deepseek_markdown_extraction_example.json")
    try:
        with open(expected_format_path, 'r', encoding='utf-8') as f:
            expected_format = json.load(f)
    except Exception as e:
        print(f"Error loading expected format: {e}")
        expected_format = None
    
    # Validate against expected format
    results = validate_extraction_output(blocks, expected_format)
    
    # Print results
    if results["valid"]:
        print(f"✅ Validation passed: {blocks_file} contains valid extraction blocks")
        print(f"📋 Statistics: {results['stats']}")
        sys.exit(0)
    else:
        print(f"❌ Validation failed: {blocks_file} contains {len(results['errors'])} errors")
        for error in results["errors"]:
            print(f"  - {error}")
        sys.exit(1)