#!/usr/bin/env python3
"""
Format Validation Script for Markdown Extraction.

This script validates that markdown extraction outputs conform to the expected format
by comparing against a sample template. It can be used as a standalone validation tool
or integrated into test pipelines.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_extraction_format")


def load_json_file(file_path: Path) -> Any:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def validate_section_structure(section: Dict[str, Any], expected_section: Dict[str, Any]) -> List[str]:
    """
    Validate that a section follows the expected format.
    
    Args:
        section: The section to validate
        expected_section: A template section with the expected format
        
    Returns:
        List of error messages, empty if valid
    """
    errors = []
    
    # Check required keys
    required_keys = {
        "uuid", "title", "content", "section_hierarchy_depth",
        "images", "tables", "code", "tests"
    }
    
    missing_keys = required_keys - set(section.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    # Check data types
    if not isinstance(section.get("uuid", ""), str):
        errors.append("uuid must be a string")
        
    if not isinstance(section.get("title", ""), str):
        errors.append("title must be a string")
        
    if not isinstance(section.get("content", ""), str):
        errors.append("content must be a string")
        
    if not isinstance(section.get("section_hierarchy_depth", []), list):
        errors.append("section_hierarchy_depth must be a list")
    else:
        for item in section.get("section_hierarchy_depth", []):
            if not isinstance(item, str):
                errors.append("section_hierarchy_depth items must be strings")
                break
    
    # Check children arrays
    for child_type in ["images", "tables", "code", "tests"]:
        if not isinstance(section.get(child_type, []), list):
            errors.append(f"{child_type} must be a list")
            continue
        
        # Validate each child if there are any
        children = section.get(child_type, [])
        if children and child_type in expected_section:
            expected_children = expected_section[child_type]
            if expected_children:
                expected_child = expected_children[0]
                for i, child in enumerate(children):
                    child_errors = validate_child_structure(child, expected_child, child_type)
                    for error in child_errors:
                        errors.append(f"{child_type}[{i}]: {error}")
    
    return errors


def validate_child_structure(child: Dict[str, Any], expected_child: Dict[str, Any], child_type: str) -> List[str]:
    """
    Validate that a child element follows the expected format.
    
    Args:
        child: The child element to validate
        expected_child: A template child with the expected format
        child_type: Type of child element
        
    Returns:
        List of error messages, empty if valid
    """
    errors = []
    
    # Check required keys
    required_keys = {"uuid"}
    
    # Child-specific requirements
    if child_type == "images":
        required_keys.update({"src", "alt"})
    elif child_type == "tables":
        required_keys.add("content")
    elif child_type == "code":
        required_keys.update({"language", "content"})
    
    missing_keys = required_keys - set(child.keys())
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")
    
    # Check table content structure
    if child_type == "tables" and "content" in child:
        table_content = child["content"]
        if not isinstance(table_content, dict):
            errors.append("table content must be an object")
        else:
            if "headers" not in table_content:
                errors.append("table content missing headers array")
            elif not isinstance(table_content["headers"], list):
                errors.append("table headers must be an array")
                
            if "rows" not in table_content:
                errors.append("table content missing rows array")
            elif not isinstance(table_content["rows"], list):
                errors.append("table rows must be an array")
            else:
                for i, row in enumerate(table_content["rows"]):
                    if not isinstance(row, list):
                        errors.append(f"table row {i} must be an array")
    
    return errors


def validate_extraction_output(output: List[Dict[str, Any]], expected_format: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate the extraction output against expected format.
    
    Args:
        output: The extraction output to validate
        expected_format: A template output with the expected format
        
    Returns:
        Dictionary with validation results
    """
    if not output:
        return {
            "valid": False,
            "errors": ["Empty output"]
        }
    
    if not expected_format:
        return {
            "valid": False,
            "errors": ["Empty expected format"]
        }
    
    all_errors = []
    
    # Find a template section with tables, code, and images if possible
    template_section = None
    for section in expected_format:
        has_tables = bool(section.get("tables"))
        has_code = bool(section.get("code"))
        has_images = bool(section.get("images"))
        
        if has_tables or has_code or has_images:
            template_section = section
            break
    
    # Use the first section as template if no better one found
    if template_section is None:
        template_section = expected_format[0]
    
    # Validate each section
    for i, section in enumerate(output):
        section_errors = validate_section_structure(section, template_section)
        if section_errors:
            all_errors.append(f"Section {i} ({section.get('title', 'unnamed')}): {', '.join(section_errors)}")
    
    # Success if no errors
    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors,
        "stats": {
            "total_sections": len(output),
            "tables": sum(len(section.get("tables", [])) for section in output),
            "code_blocks": sum(len(section.get("code", [])) for section in output),
            "images": sum(len(section.get("images", [])) for section in output)
        }
    }


def get_deepseek_extraction(repo_path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Extract deepseek.md and return the formatted output.
    
    Args:
        repo_path: Path to the repository containing deepseek.md
        
    Returns:
        Formatted extraction output or None if extraction fails
    """
    try:
        # Add the parent directory to the path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import extraction functions
        from extraction_blocks import extract_all_blocks
        from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
        
        # Extract all blocks from the repository
        blocks = extract_all_blocks(repo_path)
        
        # Convert blocks to QA-compatible format
        qa_blocks = create_qa_compatible_blocks(blocks)
        
        # Create output
        output = create_qa_compatible_output(qa_blocks)
        
        return output
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate markdown extraction format")
    parser.add_argument("--input", type=str, help="Path to extraction output JSON file")
    parser.add_argument("--expected", type=str, 
                        default="/home/grahama/workspace/experiments/agent_tools/test_repos/samples/deepseek_markdown_extraction_example.json",
                        help="Path to expected format JSON template file")
    parser.add_argument("--extract", type=str, help="Extract and validate from repository path")
    args = parser.parse_args()
    
    # Load expected format
    expected_format = load_json_file(Path(args.expected))
    if expected_format is None:
        logger.error(f"Failed to load expected format from {args.expected}")
        sys.exit(1)
    
    # Either load input file or perform extraction
    output = None
    if args.input:
        output = load_json_file(Path(args.input))
        if output is None:
            logger.error(f"Failed to load input from {args.input}")
            sys.exit(1)
    elif args.extract:
        repo_path = Path(args.extract)
        if not repo_path.exists() or not repo_path.is_dir():
            logger.error(f"Repository path not found: {repo_path}")
            sys.exit(1)
            
        logger.info(f"Extracting from repository: {repo_path}")
        output = get_deepseek_extraction(repo_path)
        if output is None:
            logger.error("Extraction failed")
            sys.exit(1)
    else:
        logger.error("Either --input or --extract must be specified")
        sys.exit(1)
    
    # Validate output
    logger.info("Validating extraction output")
    results = validate_extraction_output(output, expected_format)
    
    # Print results
    if results["valid"]:
        logger.info("✅ Validation successful")
        logger.info(f"Statistics: {results['stats']}")
    else:
        logger.error("❌ Validation failed")
        for error in results["errors"]:
            logger.error(f"  - {error}")
        logger.info(f"Statistics: {results['stats']}")
        sys.exit(1)
    
    # Success
    sys.exit(0)


if __name__ == "__main__":
    main()