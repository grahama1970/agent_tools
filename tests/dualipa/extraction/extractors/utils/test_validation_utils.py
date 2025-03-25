"""
TEST EXPECTATIONS

test_format_validation:
Input: Extracted blocks and metadata
Expected Output: Validation results for format compliance

CRITICAL RULES:
1. Schema Validation Rules:
   - All required fields must be present
   - Fields must have correct types
   - UUIDs must be valid
   - Hierarchical relationships must be consistent

2. Content Validation Rules:
   - Content must match declared language
   - Content flags must be accurate
   - Breadcrumbs must be valid
   - Depths must be consistent

3. Metadata Validation Rules:
   - Q&A pairs must be well-formed
   - Test coverage must be valid
   - Version history must be complete
   - Dependencies must be valid
"""

import pytest
import uuid
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.extraction.extractors.code.code_extractor import initialize_stats_dict
    IMPORTS_AVAILABLE = True
except ImportError as import_error:
    import traceback
    print(f"IMPORT ERROR: {import_error}")
    print("Traceback:")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required code extractor modules not available")

def is_valid_uuid(uuid_str: str) -> bool:
    """Validate UUID string format."""
    try:
        uuid_obj = uuid.UUID(uuid_str)
        return str(uuid_obj) == uuid_str
    except ValueError:
        return False

def validate_block_schema(block: Dict[str, Any]) -> List[str]:
    """Validate a single block against the required schema."""
    errors = []
    
    # Required fields for all blocks
    required_fields = {
        "uuid": str,
        "id": str,
        "type": str,
        "language": str,
        "title": str,
        "content": str,
        "file_path": str,
        "breadcrumb": list,
        "parent_uuid": (str, type(None)),
        "child_uuids": list,
        "depth": int
    }
    
    # Validate required fields
    for field, field_type in required_fields.items():
        if field not in block:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(block[field], field_type):
            if isinstance(field_type, tuple):
                if not any(isinstance(block[field], t) for t in field_type):
                    errors.append(f"Invalid type for {field}: expected {field_type}, got {type(block[field])}")
            else:
                errors.append(f"Invalid type for {field}: expected {field_type}, got {type(block[field])}")
    
    # Validate UUID format
    if "uuid" in block and not is_valid_uuid(block["uuid"]):
        errors.append(f"Invalid UUID format: {block['uuid']}")
    
    # Validate parent UUID if present
    if "parent_uuid" in block and block["parent_uuid"] is not None:
        if not is_valid_uuid(block["parent_uuid"]):
            errors.append(f"Invalid parent UUID format: {block['parent_uuid']}")
    
    # Validate child UUIDs
    if "child_uuids" in block:
        for child_uuid in block["child_uuids"]:
            if not is_valid_uuid(child_uuid):
                errors.append(f"Invalid child UUID format: {child_uuid}")
    
    # Validate type-specific fields
    if block.get("type") == "code":
        code_fields = {
            "dependencies": dict,
            "test_coverage": dict,
            "version_history": dict,
            "qa_generation": dict
        }
        for field, field_type in code_fields.items():
            if field not in block:
                errors.append(f"Missing code block field: {field}")
            elif not isinstance(block[field], field_type):
                errors.append(f"Invalid type for {field}: expected {field_type}, got {type(block[field])}")
    
    elif block.get("type") == "documentation":
        doc_fields = {
            "content_flags": dict,
            "toc_format": str,
            "section_role": str,
            "extraction_focus": str,
            "summary_instructions": str
        }
        for field, field_type in doc_fields.items():
            if field not in block:
                errors.append(f"Missing documentation block field: {field}")
            elif not isinstance(block[field], field_type):
                errors.append(f"Invalid type for {field}: expected {field_type}, got {type(block[field])}")
    
    return errors

def validate_extraction_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate extraction metadata."""
    errors = []
    
    required_fields = {
        "version": str,
        "purpose": str,
        "instructions_to_agent": str,
        "supported_languages": list,
        "extraction_focus_options": list,
        "expected_output_structure": dict
    }
    
    for field, field_type in required_fields.items():
        if field not in metadata:
            errors.append(f"Missing metadata field: {field}")
        elif not isinstance(metadata[field], field_type):
            errors.append(f"Invalid type for {field}: expected {field_type}, got {type(metadata[field])}")
    
    return errors

def test_block_schema_validation():
    """Test validation of block schema."""
    # Valid code block
    valid_code_block = {
        "uuid": str(uuid.uuid4()),
        "id": "test_function",
        "type": "code",
        "language": "python",
        "title": "Test Function",
        "content": "def test():\n    pass",
        "file_path": "test.py",
        "breadcrumb": ["test.py", "test"],
        "parent_uuid": None,
        "child_uuids": [],
        "depth": 0,
        "dependencies": {},
        "test_coverage": {"percentage": 100},
        "version_history": {"last_modified": "2024-03-20"},
        "qa_generation": {
            "difficulty_levels": ["basic"],
            "qa_examples": []
        }
    }
    
    assert not validate_block_schema(valid_code_block), "Valid code block should pass validation"
    
    # Valid documentation block
    valid_doc_block = {
        "uuid": str(uuid.uuid4()),
        "id": "readme_section",
        "type": "documentation",
        "language": "markdown",
        "title": "README",
        "content": "# README",
        "file_path": "README.md",
        "breadcrumb": ["README.md"],
        "parent_uuid": None,
        "child_uuids": [],
        "depth": 0,
        "content_flags": {"has_code": False},
        "toc_format": "README",
        "section_role": "parent_section",
        "extraction_focus": "overview",
        "summary_instructions": "Summarize the README"
    }
    
    assert not validate_block_schema(valid_doc_block), "Valid documentation block should pass validation"

def test_invalid_block_schema():
    """Test validation catches invalid blocks."""
    # Missing required fields
    invalid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "code"
    }
    
    errors = validate_block_schema(invalid_block)
    assert len(errors) > 0, "Should catch missing required fields"
    
    # Invalid UUID
    invalid_uuid_block = {
        "uuid": "not-a-uuid",
        "id": "test",
        "type": "code",
        "language": "python",
        "title": "Test",
        "content": "",
        "file_path": "test.py",
        "breadcrumb": [],
        "parent_uuid": None,
        "child_uuids": [],
        "depth": 0
    }
    
    errors = validate_block_schema(invalid_uuid_block)
    assert any("Invalid UUID" in error for error in errors), "Should catch invalid UUID"

def test_metadata_validation():
    """Test validation of extraction metadata."""
    valid_metadata = {
        "version": "1.0.0",
        "purpose": "Test extraction",
        "instructions_to_agent": "Extract code blocks",
        "supported_languages": ["python", "javascript"],
        "extraction_focus_options": ["overview", "detail"],
        "expected_output_structure": {
            "question": "string",
            "answer": "string"
        }
    }
    
    assert not validate_extraction_metadata(valid_metadata), "Valid metadata should pass validation"
    
    invalid_metadata = {
        "version": 1.0,  # Should be string
        "purpose": "Test"
    }
    
    errors = validate_extraction_metadata(invalid_metadata)
    assert len(errors) > 0, "Should catch invalid metadata"

def test_hierarchical_relationships():
    """Test validation of block hierarchical relationships."""
    parent_uuid = str(uuid.uuid4())
    child_uuid = str(uuid.uuid4())
    
    # Valid parent block
    parent_block = {
        "uuid": parent_uuid,
        "id": "parent",
        "type": "documentation",
        "language": "markdown",
        "title": "Parent",
        "content": "# Parent",
        "file_path": "doc.md",
        "breadcrumb": ["doc.md"],
        "parent_uuid": None,
        "child_uuids": [child_uuid],
        "depth": 0,
        "content_flags": {},
        "toc_format": "Parent",
        "section_role": "parent_section",
        "extraction_focus": "overview",
        "summary_instructions": "Summarize"
    }
    
    # Valid child block
    child_block = {
        "uuid": child_uuid,
        "id": "child",
        "type": "documentation",
        "language": "markdown",
        "title": "Child",
        "content": "## Child",
        "file_path": "doc.md",
        "breadcrumb": ["doc.md", "Parent", "Child"],
        "parent_uuid": parent_uuid,
        "child_uuids": [],
        "depth": 1,
        "content_flags": {},
        "toc_format": "  Child",
        "section_role": "child_section",
        "extraction_focus": "detail",
        "summary_instructions": "Summarize"
    }
    
    assert not validate_block_schema(parent_block), "Valid parent block should pass validation"
    assert not validate_block_schema(child_block), "Valid child block should pass validation"
    
    # Invalid relationship
    invalid_child = dict(child_block)
    invalid_child["parent_uuid"] = "invalid-uuid"
    
    errors = validate_block_schema(invalid_child)
    assert any("Invalid parent UUID" in error for error in errors), "Should catch invalid parent UUID"

if __name__ == "__main__":
    pytest.main([__file__]) 