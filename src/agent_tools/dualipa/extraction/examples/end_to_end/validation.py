#!/usr/bin/env python3
"""
Validation Module for DuaLipa Documentation Extraction.

This module provides comprehensive validation functions for testing documentation
extraction outputs against expected formats. It supports both structure and content
validation following Test-Driven Development (TDD) principles.

Key Functions:
- validate_qa_output: Validate that the output meets QA module requirements
- validate_structure: Validate the hierarchical structure of blocks
- validate_content_against_expected: Validate semantic content against expected values
- validate_extraction_result: Comprehensive validation of both structure and content

Dependencies:
- typing: For type hints (https://docs.python.org/3/library/typing.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)
- json: For working with JSON data (https://docs.python.org/3/library/json.html)

Examples:
    >>> output = create_qa_compatible_output(blocks)
    >>> is_valid = validate_qa_output(output)
    >>> print(f"Output validation {'successful' if is_valid else 'failed'}")
    Output validation successful
    
    >>> extraction_result = extract_documentation(url)
    >>> expected_format = load_expected_format("expected_format.json")
    >>> validation_result = validate_extraction_result(extraction_result, expected_format)
    >>> print(f"Validation {'passed' if validation_result['valid'] else 'failed'} with score {validation_result['score']}")
    Validation passed with score 92.5
"""

from typing import Dict, Any, List, Union, Set, Optional, Tuple
import logging
import json
import os
import re
from pathlib import Path

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


def validate_structure(blocks: List[Dict[str, Any]], expected_format: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the hierarchical structure of extracted blocks.
    
    This function checks that the extracted blocks maintain the expected 
    hierarchical structure, including parent-child relationships, block types,
    and ordering.
    
    Args:
        blocks: List of extracted blocks
        expected_format: Dictionary containing expected structural information
        
    Returns:
        Dictionary with validation results including:
        - valid (bool): True if validation passed
        - score (float): Percentage of structure validation checks that passed
        - errors (list): List of validation errors
        - successes (list): List of validation successes
        
    Example:
        >>> blocks = extract_all_blocks('documentation/example')
        >>> expected = load_json('expected_format.json')
        >>> results = validate_structure(blocks, expected)
        >>> print(f"Structure validation: {results['valid']} with score {results['score']}%")
        Structure validation: True with score 95.5%
    """
    logger.info("Validating block structure")
    
    results = {
        "valid": False,
        "score": 0.0,
        "errors": [],
        "successes": [],
        "total_checks": 0,
        "passed_checks": 0
    }
    
    if not blocks:
        results["errors"].append("No blocks to validate")
        return results
    
    if not expected_format or not isinstance(expected_format, dict):
        results["errors"].append("Invalid expected format")
        return results
    
    # Get the expected structure
    expected_structure = expected_format.get("expected_structure", {})
    if not expected_structure:
        results["errors"].append("No expected structure defined")
        return results
    
    # Collect all block UUIDs
    all_uuids = {block.get("uuid") for block in blocks if "uuid" in block}
    
    # Check for required block types
    required_block_types = expected_structure.get("required_block_types", [])
    
    # Define type mappings for flexibility
    type_mappings = {
        "documentation": ["documentation", "file"],
        "doc_page": ["doc_page", "file"],
        "doc_section": ["doc_section", "section"],
        "code_block": ["code_block", "code"],
        "table": ["table"]
    }
    
    for block_type in required_block_types:
        results["total_checks"] += 1
        
        # Check for the block type or any of its acceptable alternatives
        equivalent_types = type_mappings.get(block_type, [block_type])
        
        blocks_of_equivalent_types = [
            b for b in blocks 
            if b.get("type") in equivalent_types
        ]
        
        if blocks_of_equivalent_types:
            results["successes"].append(f"Found required block type: {block_type} (or equivalent)")
            results["passed_checks"] += 1
        else:
            results["errors"].append(f"Missing required block type: {block_type}")
    
    # Check for the correct order and hierarchy of block types
    expected_hierarchy = expected_structure.get("hierarchy", [])
    if expected_hierarchy:
        for hierarchy_level in expected_hierarchy:
            parent_type = hierarchy_level.get("parent_type")
            child_types = hierarchy_level.get("child_types", [])
            
            # Check each block of parent_type or equivalent
            equivalent_parent_types = type_mappings.get(parent_type, [parent_type])
            parents = [b for b in blocks if b.get("type") in equivalent_parent_types]
            results["total_checks"] += 1
            
            if not parents and parent_type in required_block_types:
                results["errors"].append(f"Missing parent block type: {parent_type}")
            elif parents:
                results["successes"].append(f"Found parent block type: {parent_type} (or equivalent)")
                results["passed_checks"] += 1
                
                # Check parent-child relationships
                for parent in parents:
                    parent_uuid = parent.get("uuid")
                    child_uuids = parent.get("child_uuids", [])
                    
                    # Check that children exist
                    for child_uuid in child_uuids:
                        results["total_checks"] += 1
                        if child_uuid in all_uuids:
                            results["passed_checks"] += 1
                        else:
                            results["errors"].append(f"Reference to non-existent child UUID: {child_uuid}")
                    
                    # Check that children have correct types or equivalent types
                    children = [b for b in blocks if b.get("uuid") in child_uuids]
                    for child in children:
                        child_type = child.get("type")
                        results["total_checks"] += 1
                        
                        # Check if child type is directly in allowed types
                        direct_match = child_type in child_types
                        
                        # Check if child type is equivalent to any allowed type
                        equivalent_match = False
                        for allowed_type in child_types:
                            equivalent_types = type_mappings.get(allowed_type, [allowed_type])
                            if child_type in equivalent_types:
                                equivalent_match = True
                                break
                        
                        if direct_match or equivalent_match:
                            results["passed_checks"] += 1
                            results["successes"].append(f"Child {child.get('uuid')} has correct type: {child_type}")
                        else:
                            results["errors"].append(f"Child {child.get('uuid')} has incorrect type: {child_type}, expected one of {child_types}")
                    
                    # Check bidirectional references (children point back to parent)
                    for child in children:
                        results["total_checks"] += 1
                        if child.get("parent_uuid") == parent_uuid:
                            results["passed_checks"] += 1
                        else:
                            results["errors"].append(f"Child {child.get('uuid')} does not reference parent {parent_uuid}")
    
    # Check metadata consistency
    metadata_checks = expected_structure.get("metadata_checks", [])
    for check in metadata_checks:
        field_path = check.get("field")
        requirement = check.get("requirement")
        
        if field_path and requirement:
            for block in blocks:
                # Navigate the field path (e.g., "metadata.language")
                value = block
                for part in field_path.split('.'):
                    if part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    results["total_checks"] += 1
                    if requirement == "not_empty" and value:
                        results["passed_checks"] += 1
                    elif requirement == "uuid_format" and isinstance(value, str) and len(value) > 8:
                        results["passed_checks"] += 1
                    elif requirement.startswith("one_of:"):
                        allowed_values = requirement.split(":", 1)[1].split(",")
                        if value in allowed_values:
                            results["passed_checks"] += 1
                        else:
                            results["errors"].append(f"Block {block.get('uuid')} field {field_path} has invalid value: {value}, expected one of {allowed_values}")
                    else:
                        results["errors"].append(f"Block {block.get('uuid')} field {field_path} failed check: {requirement}")
    
    # Calculate final score and result
    if results["total_checks"] > 0:
        results["score"] = round((results["passed_checks"] / results["total_checks"]) * 100, 1)
    else:
        # If there are no checks (empty expected structure), consider it valid with 100% score
        results["score"] = 100.0
        results["passed_checks"] = 0
        results["total_checks"] = 0
    
    # Structure validation passes if score is above threshold (default 75%)
    threshold = expected_structure.get("validation_threshold", 75)
    results["valid"] = results["score"] >= threshold
    
    if results["valid"]:
        logger.info(f"Structure validation passed with score {results['score']}%")
    else:
        logger.warning(f"Structure validation failed with score {results['score']}% (threshold: {threshold}%)")
        for error in results["errors"]:
            logger.warning(f"  - {error}")
    
    return results


def validate_content_against_expected(
    blocks: List[Dict[str, Any]], 
    expected_content: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate the semantic content of extracted blocks against expected values.
    
    This function checks that the extracted blocks contain the expected semantic
    content as defined in the expected_content dictionary, including function names,
    parameters, descriptions, examples, etc.
    
    Args:
        blocks: List of extracted blocks
        expected_content: Dictionary describing expected content
        
    Returns:
        Dictionary with validation results including:
        - valid (bool): True if validation passed
        - score (float): Percentage of content validation checks that passed
        - errors (list): List of validation errors
        - successes (list): List of validation successes
        
    Example:
        >>> blocks = extract_all_blocks('documentation/example')
        >>> expected = load_json('expected_format.json')["expected_content_validation"]
        >>> results = validate_content_against_expected(blocks, expected)
        >>> print(f"Content validation: {results['valid']} with score {results['score']}%")
        Content validation: True with score 92.3%
    """
    logger.info("Validating content against expected values")
    
    results = {
        "valid": False,
        "score": 0.0,
        "errors": [],
        "successes": [],
        "total_checks": 0,
        "passed_checks": 0
    }
    
    if not blocks:
        results["errors"].append("No blocks to validate")
        return results
    
    if not expected_content or not isinstance(expected_content, dict):
        results["errors"].append("Invalid expected content format")
        return results
    
    # Combine all block content for easier searching
    all_content = "\n".join([block.get("content", "") for block in blocks])
    all_code = "\n".join([cb.get("content", "") for block in blocks 
                         for cb in block.get("code", [])])
    
    # Check for function presence
    function_name = expected_content.get("function_name")
    if function_name:
        results["total_checks"] += 1
        if function_name in all_content:
            results["successes"].append(f"Found function name: {function_name}")
            results["passed_checks"] += 1
        else:
            results["errors"].append(f"Missing function name: {function_name}")
    
    # Check for function purpose
    function_purpose = expected_content.get("function_purpose")
    if function_purpose:
        results["total_checks"] += 1
        if any(purpose in all_content for purpose in function_purpose):
            purposes_found = [purpose for purpose in function_purpose if purpose in all_content]
            results["successes"].append(f"Found function purpose: {', '.join(purposes_found)}")
            results["passed_checks"] += 1
        else:
            results["errors"].append(f"Missing function purpose, expected one of: {function_purpose}")
    
    # Check for parameters
    parameters = expected_content.get("parameters", [])
    for param in parameters:
        param_name = param.get("name")
        param_type = param.get("type")
        param_description = param.get("description", [])
        
        if param_name:
            results["total_checks"] += 1
            if param_name in all_content:
                results["successes"].append(f"Found parameter: {param_name}")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing parameter: {param_name}")
        
        if param_type:
            results["total_checks"] += 1
            # Check for param_type near param_name
            pattern = f"{param_name}.*?{param_type}|{param_type}.*?{param_name}"
            if re.search(pattern, all_content, re.DOTALL | re.IGNORECASE):
                results["successes"].append(f"Found parameter type: {param_type} for {param_name}")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing parameter type: {param_type} for {param_name}")
        
        if param_description:
            results["total_checks"] += 1
            if any(desc in all_content for desc in param_description):
                descriptions_found = [desc for desc in param_description if desc in all_content]
                results["successes"].append(f"Found parameter description for {param_name}: {', '.join(descriptions_found)}")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing parameter description for {param_name}, expected one of: {param_description}")
    
    # Check for return type
    return_type = expected_content.get("return_type")
    if return_type:
        results["total_checks"] += 1
        returns_pattern = r"returns?:.*?" + re.escape(return_type)
        if re.search(returns_pattern, all_content, re.DOTALL | re.IGNORECASE) or return_type in all_content:
            results["successes"].append(f"Found return type: {return_type}")
            results["passed_checks"] += 1
        else:
            results["errors"].append(f"Missing return type: {return_type}")
    
    # Check for examples
    examples = expected_content.get("examples", [])
    for example in examples:
        example_code = example.get("code")
        example_output = example.get("output")
        
        if example_code:
            results["total_checks"] += 1
            # Look for the example code in code blocks
            example_found = example_code in all_code or example_code in all_content
            if example_found:
                results["successes"].append(f"Found example code: {example_code[:30]}...")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing example code: {example_code[:30]}...")
        
        if example_output:
            results["total_checks"] += 1
            if example_output in all_content:
                results["successes"].append(f"Found example output: {example_output[:30]}...")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing example output: {example_output[:30]}...")
    
    # Check for required keywords
    required_keywords = expected_content.get("required_keywords", [])
    for keyword in required_keywords:
        results["total_checks"] += 1
        if keyword in all_content:
            results["successes"].append(f"Found required keyword: {keyword}")
            results["passed_checks"] += 1
        else:
            results["errors"].append(f"Missing required keyword: {keyword}")
    
    # Calculate final score and result
    if results["total_checks"] > 0:
        results["score"] = round((results["passed_checks"] / results["total_checks"]) * 100, 1)
    
    # Content validation passes if score is above threshold (default 85%)
    threshold = expected_content.get("validation_threshold", 85)
    results["valid"] = results["score"] >= threshold
    
    if results["valid"]:
        logger.info(f"Content validation passed with score {results['score']}%")
    else:
        logger.warning(f"Content validation failed with score {results['score']}% (threshold: {threshold}%)")
        for error in results["errors"]:
            logger.warning(f"  - {error}")
    
    return results


def validate_markdown_and_html_structure(blocks: List[Dict[str, Any]], expected_format: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the structure of an extraction specifically comparing markdown and HTML outputs.
    This function is particularly useful for ensuring extraction consistency across different source formats.
    
    Args:
        blocks: List of extracted blocks
        expected_format: Dictionary describing expected structure
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating structural consistency between markdown and HTML extraction")
    
    results = {
        "valid": False,
        "score": 0.0,
        "errors": [],
        "successes": [],
        "total_checks": 0,
        "passed_checks": 0
    }
    
    if not blocks:
        results["errors"].append("No blocks to validate")
        return results
    
    if not expected_format or not isinstance(expected_format, dict):
        results["errors"].append("Invalid expected format")
        return results
    
    # Get expected structure checks
    structure_checks = expected_format.get("structure_consistency", {})
    
    # Check required root blocks (skip if list is empty or empty structure checks)
    required_roots = structure_checks.get("required_root_blocks", [])
    skip_root_checks = structure_checks.get("skip_root_checks", False)
    
    if required_roots and not skip_root_checks:
        # Define type mappings for flexibility
        type_mappings = {
            "documentation": ["documentation", "file"],
            "doc_page": ["doc_page", "file"],
            "doc_section": ["doc_section", "section"],
            "code_block": ["code_block", "code"],
            "table": ["table"]
        }
        
        for root_type in required_roots:
            results["total_checks"] += 1
            
            # Check for equivalent root types
            equivalent_types = type_mappings.get(root_type, [root_type])
            root_blocks = [b for b in blocks if b.get("type") in equivalent_types and not b.get("parent_uuid")]
            
            if root_blocks:
                results["successes"].append(f"Found required root block type: {root_type} (or equivalent)")
                results["passed_checks"] += 1
            else:
                results["errors"].append(f"Missing required root block type: {root_type}")
    
    # Check hierarchical relationships
    hierarchical_types = structure_checks.get("hierarchical_types", [])
    for hierarchy in hierarchical_types:
        parent_type = hierarchy.get("parent")
        child_types = hierarchy.get("children", [])
        
        if parent_type and child_types:
            # Check for equivalent parent types
            equivalent_types = type_mappings.get(parent_type, [parent_type])
            parent_blocks = [b for b in blocks if b.get("type") in equivalent_types]
            
            for parent in parent_blocks:
                parent_uuid = parent.get("uuid")
                child_uuids = parent.get("child_uuids", [])
                
                # Check that parent has children
                results["total_checks"] += 1
                if child_uuids:
                    results["successes"].append(f"Parent {parent_uuid} has children")
                    results["passed_checks"] += 1
                else:
                    results["errors"].append(f"Parent {parent_uuid} has no children")
                    continue
                
                # Check child types
                child_blocks = [b for b in blocks if b.get("uuid") in child_uuids]
                for child in child_blocks:
                    results["total_checks"] += 1
                    child_type = child.get("type")
                    if child_type in child_types:
                        results["successes"].append(f"Child {child.get('uuid')} has valid type: {child_type}")
                        results["passed_checks"] += 1
                    else:
                        results["errors"].append(f"Child {child.get('uuid')} has invalid type: {child_type}, expected one of {child_types}")
                
                # Check bidirectional references
                for child in child_blocks:
                    results["total_checks"] += 1
                    if child.get("parent_uuid") == parent_uuid:
                        results["successes"].append(f"Child {child.get('uuid')} correctly references parent {parent_uuid}")
                        results["passed_checks"] += 1
                    else:
                        results["errors"].append(f"Child {child.get('uuid')} does not reference parent {parent_uuid}")
    
    # Check for file paths in metadata (skip if requested)
    skip_metadata = structure_checks.get("skip_metadata_checks", False)
    if not skip_metadata:
        results["total_checks"] += 1
        files_with_paths = [b for b in blocks if "file_path" in b.get("metadata", {})]
        if files_with_paths:
            results["successes"].append(f"Found file paths in {len(files_with_paths)} blocks")
            results["passed_checks"] += 1
        else:
            results["errors"].append("No blocks have file paths in metadata")
    
    # Calculate final score and result
    if results["total_checks"] > 0:
        results["score"] = round((results["passed_checks"] / results["total_checks"]) * 100, 1)
    else:
        # If there are no checks (empty expected structure), consider it valid with 100% score
        results["score"] = 100.0
        results["passed_checks"] = 0
        results["total_checks"] = 0
    
    # Validation passes if score is above threshold (default 75%)
    threshold = structure_checks.get("validation_threshold", 75)
    results["valid"] = results["score"] >= threshold
    
    if results["valid"]:
        logger.info(f"Structure consistency validation passed with score {results['score']}%")
    else:
        logger.warning(f"Structure consistency validation failed with score {results['score']}% (threshold: {threshold}%)")
        for error in results["errors"]:
            logger.warning(f"  - {error}")
    
    return results


def validate_extraction_result(
    extraction_result: Any, 
    expected_format: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive validation of both structure and content of extraction results.
    This is the main validation function that combines all other validation functions.
    
    Args:
        extraction_result: The extraction result to validate (could be dict or list)
        expected_format: Dictionary describing expected structure and content
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> extraction_result = extract_documentation('https://example.com/docs')
        >>> expected_format = load_json_file('expected_format.json')
        >>> validation = validate_extraction_result(extraction_result, expected_format)
        >>> print(f"Validation {'passed' if validation['valid'] else 'failed'} with score {validation['overall_score']}%")
        Validation passed with score 88.5%
    """
    logger.info("Performing comprehensive validation of extraction results")
    
    results = {
        "valid": False,
        "overall_score": 0.0,
        "structure_validation": None,
        "content_validation": None,
        "format_validation": None
    }
    
    # Ensure we have blocks to validate
    blocks = []
    if isinstance(extraction_result, list):
        blocks = extraction_result
    elif isinstance(extraction_result, dict) and "sections" in extraction_result:
        blocks = extraction_result["sections"]
    else:
        logger.error("Invalid extraction result format")
        return results
    
    # Check basic QA compatibility
    format_valid = validate_qa_output(extraction_result)
    results["format_validation"] = {"valid": format_valid}
    
    # Structure validation
    if "expected_structure" in expected_format:
        structure_results = validate_structure(blocks, expected_format)
        results["structure_validation"] = structure_results
    
    # Content validation
    if "expected_content_validation" in expected_format:
        content_results = validate_content_against_expected(
            blocks, expected_format["expected_content_validation"])
        results["content_validation"] = content_results
    
    # If both HTML and markdown extraction is expected, validate consistency
    if "structure_consistency" in expected_format:
        consistency_results = validate_markdown_and_html_structure(blocks, expected_format)
        results["structure_consistency"] = consistency_results
    
    # Calculate overall score
    scores = []
    if "structure_validation" in results and results["structure_validation"]:
        scores.append(results["structure_validation"]["score"])
    
    if "content_validation" in results and results["content_validation"]:
        scores.append(results["content_validation"]["score"])
    
    if "structure_consistency" in results and results["structure_consistency"]:
        scores.append(results["structure_consistency"]["score"])
    
    if scores:
        results["overall_score"] = round(sum(scores) / len(scores), 1)
    
    # Determine overall validity
    results["valid"] = (
        (not "structure_validation" in results or results["structure_validation"].get("valid", False)) and
        (not "content_validation" in results or results["content_validation"].get("valid", False)) and
        (not "structure_consistency" in results or results["structure_consistency"].get("valid", False)) and
        results["format_validation"].get("valid", False)
    )
    
    if results["valid"]:
        logger.info(f"Extraction validation passed with overall score {results['overall_score']}%")
    else:
        logger.warning(f"Extraction validation failed with overall score {results['overall_score']}%")
    
    return results


def load_expected_format(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load expected format from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing expected format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading expected format: {e}")
        return {}


def save_validation_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save validation results to a JSON file.
    
    Args:
        results: Validation results dictionary
        output_path: Path to save the results
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")


def validate_extraction(extraction_result: Any, expected_format_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Validate extraction results against an expected format.
    
    Args:
        extraction_result: The extraction result to validate (could be dict or list)
        expected_format_path: Optional path to expected format JSON file
        
    Returns:
        Dictionary with validation results
        
    Example:
        >>> extraction_result = extract_repository('./test_repos/python-sample')
        >>> validation = validate_extraction(extraction_result, 'expected_format.json')
        >>> print(f"Validation {'passed' if validation['valid'] else 'failed'}")
        Validation passed
    """
    logger.info("Validating extraction results")
    
    # First perform basic QA output validation
    format_valid = validate_qa_output(extraction_result)
    
    # If expected format is provided, perform more detailed validation
    if expected_format_path:
        expected_format = load_expected_format(expected_format_path)
        return validate_extraction_result(extraction_result, expected_format)
    
    # Return basic validation results if no expected format provided
    return {
        "valid": format_valid,
        "overall_score": 100.0 if format_valid else 0.0,
        "format_validation": {"valid": format_valid}
    }