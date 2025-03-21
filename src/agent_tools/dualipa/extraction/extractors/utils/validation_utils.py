"""
Block validation utilities for DuaLipa.

This module handles validation of code block formats and metadata,
ensuring consistency and completeness of extracted blocks.

Key Features:
1. Block format validation
2. Metadata validation
3. Required field checking
4. Type validation

Dependencies:
- loguru: For logging

Related Files:
- language_utils.py: Used for language validation
- verification_utils.py: Used after validation
"""

import uuid
from typing import Dict, Any, List, Set, Optional
from loguru import logger

from .language_utils import get_language_info, normalize_language

# Required fields for all blocks
REQUIRED_FIELDS = {
    "uuid": str,
    "type": str,
    "content": str,
    "metadata": dict
}

# Required metadata fields
REQUIRED_METADATA = {
    "source_file": str,
    "line_start": int,
    "line_end": int,
    "language": str
}

# Valid block types
VALID_BLOCK_TYPES = {
    "function",
    "class",
    "method",
    "interface",
    "section",
    "code",
    "script"
}

def validate_block_format(block: Dict[str, Any]) -> List[str]:
    """
    Validate format of a code block.
    
    Args:
        block: Code block dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in block:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(block[field], field_type):
            errors.append(f"Invalid type for {field}: expected {field_type.__name__}")
            
    # Check metadata
    if "metadata" in block:
        metadata_errors = validate_metadata(block["metadata"])
        errors.extend(metadata_errors)
        
    # Validate UUID format
    if "uuid" in block:
        try:
            uuid.UUID(block["uuid"])
        except ValueError:
            errors.append("Invalid UUID format")
            
    # Validate block type
    if "type" in block and block["type"] not in VALID_BLOCK_TYPES:
        errors.append(f"Invalid block type: {block['type']}")
        
    # Validate content
    if "content" in block and not block["content"].strip():
        errors.append("Empty block content")
        
    return errors

def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    Validate block metadata.
    
    Args:
        metadata: Block metadata dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required metadata fields
    for field, field_type in REQUIRED_METADATA.items():
        if field not in metadata:
            errors.append(f"Missing required metadata: {field}")
        elif not isinstance(metadata[field], field_type):
            errors.append(f"Invalid type for metadata {field}: expected {field_type.__name__}")
            
    # Validate line numbers
    if "line_start" in metadata and "line_end" in metadata:
        if metadata["line_start"] > metadata["line_end"]:
            errors.append("Invalid line numbers: start > end")
            
    # Validate language
    if "language" in metadata:
        language = normalize_language(metadata["language"])
        if not get_language_info(language):
            errors.append(f"Unsupported language: {metadata['language']}")
            
    return errors

def validate_block_consistency(blocks: List[Dict[str, Any]]) -> List[str]:
    """
    Validate consistency across multiple blocks.
    
    Args:
        blocks: List of code blocks
        
    Returns:
        List of consistency errors (empty if valid)
    """
    errors = []
    uuids = set()
    
    for block in blocks:
        # Check UUID uniqueness
        if "uuid" in block:
            if block["uuid"] in uuids:
                errors.append(f"Duplicate UUID: {block['uuid']}")
            uuids.add(block["uuid"])
            
        # Validate individual block
        block_errors = validate_block_format(block)
        if block_errors:
            errors.append(f"Block {block.get('uuid', 'unknown')}: {', '.join(block_errors)}")
            
    return errors

def validate_block(block: Dict[str, Any]) -> bool:
    """
    Validate a code block and standardize its format for QA module compatibility.
    
    Args:
        block: Code block dictionary
        
    Returns:
        True if valid, False otherwise
        
    Example Input:
        {
            "type": "function",
            "name": "add",
            "content": "function add(a, b) { return a + b; }",
            "line_start": 1,
            "line_end": 1,
            "metadata": {
                "language": "javascript",
                "file": "math.js"
            }
        }
        
    Example Output:
        True  # With standardized fields added/normalized
    """
    try:
        # Check required fields (using QA module's expected format)
        required = ["type", "name", "content"]
        if not all(field in block for field in required):
            logger.warning(f"Missing required field in block: {', '.join([f for f in required if f not in block])}")
            return False
        
        # Ensure block has a unique ID (uuid)
        if "uuid" not in block:
            # Handle blocks with 'id' field instead of 'uuid'
            if "id" in block:
                block["uuid"] = block["id"]
            else:
                # Add a new UUID if neither exists
                block["uuid"] = str(uuid.uuid4())
        
        # Standardize metadata dictionary
        if "metadata" not in block:
            # Create metadata from block fields
            metadata = {}
            
            # Handle line number variations
            if "line_start" in block:
                metadata["line_start"] = block["line_start"]
            elif "start_line" in block:
                metadata["line_start"] = block["start_line"]
                
            if "line_end" in block:
                metadata["line_end"] = block["line_end"]
            elif "end_line" in block:
                metadata["line_end"] = block["end_line"]
            
            # Check for language in block
            if "language" in block:
                metadata["language"] = block["language"]
                
            # Standardize source file path
            if "file" in block:
                metadata["source_file"] = block["file"]
            elif "path" in block:
                metadata["source_file"] = block["path"]
            elif "source_file" in block:
                metadata["source_file"] = block["source_file"]
                
            # Add imports if available
            if "imports" in block:
                metadata["imports"] = block["imports"]
                
            # Set metadata
            block["metadata"] = metadata
        else:
            # Ensure metadata has required fields
            metadata = block["metadata"]
            
            # Standardize metadata field names
            if "file" in metadata and "source_file" not in metadata:
                metadata["source_file"] = metadata["file"]
                
            if "path" in metadata and "source_file" not in metadata:
                metadata["source_file"] = metadata["path"]
                
            # Standardize line numbers
            if "start_line" in metadata and "line_start" not in metadata:
                metadata["line_start"] = metadata["start_line"]
                
            if "end_line" in metadata and "line_end" not in metadata:
                metadata["line_end"] = metadata["end_line"]
                
            # Ensure line numbers exist
            if "line_start" not in metadata:
                if "line_start" in block:
                    metadata["line_start"] = block["line_start"]
                elif "start_line" in block:
                    metadata["line_start"] = block["start_line"]
                    
            if "line_end" not in metadata:
                if "line_end" in block:
                    metadata["line_end"] = block["line_end"]
                elif "end_line" in block:
                    metadata["line_end"] = block["end_line"]
                
            # Ensure language is set
            if "language" not in metadata and "language" in block:
                metadata["language"] = block["language"]
        
        # Check content
        if not block["content"] or len(block["content"].strip()) == 0:
            logger.warning(f"Empty content in block: {block.get('name', 'unknown')}")
            return False
        
        # Check line numbers if present
        line_start_in_block = "line_start" in block
        line_end_in_block = "line_end" in block
        line_start_in_metadata = "line_start" in block.get("metadata", {})
        line_end_in_metadata = "line_end" in block.get("metadata", {})
        
        if (line_start_in_block and line_end_in_block) and block["line_start"] > block["line_end"]:
            logger.warning(f"Invalid line numbers in block: {block.get('name', 'unknown')}")
            return False
            
        if (line_start_in_metadata and line_end_in_metadata) and block["metadata"]["line_start"] > block["metadata"]["line_end"]:
            logger.warning(f"Invalid line numbers in block metadata: {block.get('name', 'unknown')}")
            return False
        
        # Make sure necessary top-level fields exist for QA compatibility
        if "type" not in block and "type" in block.get("metadata", {}):
            block["type"] = block["metadata"]["type"]
            
        if "name" not in block and "name" in block.get("metadata", {}):
            block["name"] = block["metadata"]["name"]
            
        if "language" not in block and "language" in block.get("metadata", {}):
            block["language"] = block["metadata"]["language"]
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating block: {e}")
        return False

def usage_example() -> None:
    """Example usage of validation utilities."""
    # Example valid block
    valid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "hello",
        "content": "def hello(): print('Hello')",
        "metadata": {
            "source_file": "script.py",
            "line_start": 1,
            "line_end": 2,
            "language": "python"
        }
    }
    
    # Example invalid block
    invalid_block = {
        "uuid": "not-a-uuid",
        "type": "invalid_type",
        "content": "",
        "metadata": {
            "source_file": "script.py",
            "line_start": 5,
            "line_end": 3,  # Invalid: start > end
            "language": "unknown"
        }
    }
    
    # Validate blocks
    print("Valid block validation:")
    errors = validate_block_format(valid_block)
    if not errors:
        print("No errors found")
    else:
        print("\n".join(errors))
        
    print("\nInvalid block validation:")
    errors = validate_block_format(invalid_block)
    print("\n".join(errors))
    
    # Check consistency
    print("\nBlock consistency validation:")
    blocks = [valid_block, invalid_block]
    errors = validate_block_consistency(blocks)
    print("\n".join(errors))
    
    # Test validate_block function
    print("\nValidate block with standardization:")
    block_to_standardize = {
        "type": "function",
        "name": "add",
        "content": "function add(a, b) { return a + b; }",
        "line_start": 1,
        "line_end": 1,
        "file": "math.js",
        "language": "javascript"
    }
    
    is_valid = validate_block(block_to_standardize)
    print(f"Valid: {is_valid}")
    print("UUID added:", "uuid" in block_to_standardize)
    print("Metadata created:", "metadata" in block_to_standardize)
    if "metadata" in block_to_standardize:
        print("source_file standardized:", "source_file" in block_to_standardize["metadata"])
        print("line_start in metadata:", "line_start" in block_to_standardize["metadata"]) 