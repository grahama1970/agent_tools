#!/usr/bin/env python3
"""
Validation Module for DuaLipa.

This module provides functions for validating the output of the extraction process
to ensure it meets the requirements of the QA generation module. It checks for
required fields, consistent relationships, and proper formatting.

Key Functions:
- validate_qa_output: Validate that the output meets QA module requirements

Dependencies:
- typing: For type hints (https://docs.python.org/3/library/typing.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)

Examples:
    >>> output = create_qa_compatible_output(blocks)
    >>> is_valid = validate_qa_output(output)
    >>> print(f"Output validation {'successful' if is_valid else 'failed'}")
    Output validation successful
"""

from typing import Dict, Any, List, Union
import logging

# Setup logging
logger = logging.getLogger("extraction.validation")


def validate_qa_output(output: Any) -> bool:
    """Validate that the output meets QA module requirements.
    
    This function checks that the extraction output includes all required fields
    and maintains consistent relationships between sections.
    
    Args:
        output: The complete output dictionary or list to validate
        
    Returns:
        True if validation passes, False if it fails
        
    Example:
        >>> blocks = extract_all_blocks(Path('./test_repos/python-sample'))
        >>> output = create_qa_compatible_output(create_qa_compatible_blocks(blocks))
        >>> validate_qa_output(output)
        True
    """
    logger.info("Validating QA-compatible output")
    
    # Check for deepseek format - a list of sections
    if isinstance(output, list):
        logger.info("Validating deepseek format output")
        
        # Check for empty list
        if not output:
            logger.error("Empty deepseek format output")
            return False
        
        # Check first section for required fields
        required_fields = ["uuid", "title", "content", "section_hierarchy_depth"]
        for field in required_fields:
            if field not in output[0]:
                logger.error(f"Missing required field in deepseek format: {field}")
                return False
        
        # Validate section contents
        for i, section in enumerate(output):
            # Check required nested elements
            for element_type in ["images", "tables", "code"]:
                if element_type not in section:
                    logger.error(f"Section {i} missing element type: {element_type}")
                    return False
                
                # Check element content if any exist
                elements = section.get(element_type, [])
                if elements:
                    if element_type == "images":
                        if "src" not in elements[0] or "alt" not in elements[0]:
                            logger.error(f"Section {i} has invalid image format")
                            return False
                    elif element_type == "tables":
                        if "content" not in elements[0]:
                            logger.error(f"Section {i} has invalid table format")
                            return False
                    elif element_type == "code":
                        if "language" not in elements[0] or "content" not in elements[0]:
                            logger.error(f"Section {i} has invalid code format")
                            return False
        
        # If we got here, the deepseek format is valid
        logger.info("Deepseek format validation successful")
        return True
    
    # Standard format validation
    if not isinstance(output, dict):
        logger.error(f"Output is neither a dict nor a list: {type(output)}")
        return False
    
    # Check required top-level fields
    required_top_fields = ["sections", "extraction_metadata", "section_relationships"]
    for field in required_top_fields:
        if field not in output:
            logger.error(f"Missing required top-level field: {field}")
            return False
    
    # Check required metadata fields
    required_metadata_fields = ["model_used", "timestamp", "version", "statistics"]
    for field in required_metadata_fields:
        if field not in output.get("extraction_metadata", {}):
            logger.error(f"Missing required metadata field: {field}")
            return False
    
    # Check if sections exist
    if not output["sections"]:
        logger.error("No sections in output")
        return False
    
    # Check required fields in each section
    required_section_fields = [
        "uuid", "id", "type", "language", "content", "metadata", 
        "extraction_focus", "summary_instructions", "breadcrumb",
        "parent_uuid", "child_uuids"
    ]
    
    section_validation_errors = 0
    for i, section in enumerate(output["sections"]):
        for field in required_section_fields:
            if field not in section:
                logger.error(f"Section {i} missing required field: {field}")
                section_validation_errors += 1
        
        # Check metadata fields
        required_metadata_fields = ["language", "source_file"]
        for field in required_metadata_fields:
            if field not in section.get("metadata", {}):
                logger.error(f"Section {i} metadata missing required field: {field}")
                section_validation_errors += 1
    
    # If there are validation errors, report them in the statistics
    if section_validation_errors > 0:
        if "statistics" in output.get("extraction_metadata", {}):
            output["extraction_metadata"]["statistics"]["validation_errors"] = section_validation_errors
        logger.error(f"Found {section_validation_errors} validation errors in sections")
        return False
    
    # Verify relationship data
    relationship_types = ["parent_child", "imports", "inheritance"]
    for rel_type in relationship_types:
        if rel_type not in output.get("section_relationships", {}):
            logger.error(f"Missing relationship type: {rel_type}")
            return False
    
    # Perform consistency checks
    consistency_errors = 0
    
    # Check UUID consistency
    all_uuids = {section.get("uuid") for section in output["sections"] if section.get("uuid")}
    
    # Check parent_child relationships
    for uuid, rel_data in output.get("section_relationships", {}).get("parent_child", {}).items():
        # Check that parent exists
        parent_uuid = rel_data.get("parent")
        if parent_uuid and parent_uuid not in all_uuids:
            logger.error(f"Reference to non-existent parent UUID: {parent_uuid}")
            consistency_errors += 1
        
        # Check that children exist
        for child_uuid in rel_data.get("children", []):
            if child_uuid not in all_uuids:
                logger.error(f"Reference to non-existent child UUID: {child_uuid}")
                consistency_errors += 1
    
    # If there are consistency errors, report them
    if consistency_errors > 0:
        if "statistics" in output.get("extraction_metadata", {}):
            output["extraction_metadata"]["statistics"]["verification_errors"] = consistency_errors
        logger.error(f"Found {consistency_errors} consistency errors in output")
        return False
    
    logger.info("Output validation successful")
    return True