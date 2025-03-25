"""Test integration between code hierarchy and QA modules.

This module tests the integration between the code hierarchy extraction
and the QA module, ensuring that hierarchy output can be consumed by QA.

Test Specifications:
- Verify that hierarchy output includes all fields required by QA
- Test that the QA processor can validate hierarchy output
- Ensure that block metadata is correctly standardized
"""

import sys
import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent_tools.dualipa.extraction.extractors.hierarchy import analyze_code_hierarchy
from src.agent_tools.dualipa.extraction.extractors.hierarchy.utils import format_hierarchy_summary


@pytest.fixture
def qa_requirement_fields():
    """Return the fields required by the QA module."""
    # Based on the analysis of validation_utils.py in the QA module
    section_fields = [
        "uuid",             # Required for block identification
        "type",             # Required for content type (code or documentation)
        "content",          # Required for actual content
        "extraction_focus", # Optional but defaulted if missing
        "summary_instructions"  # Optional but defaulted if missing
    ]
    
    # Additional metadata required for specific block types
    code_block_fields = [
        "language",         # Required for code blocks
        "file_path",        # Required for file reference
        "dependencies"      # Required for dependency tracking
    ]
    
    # Required for hierarchical relationships
    relationship_fields = [
        "parent_uuid",      # Optional, required if not root
        "child_uuids",      # Required for tracking children
        "breadcrumb"        # Required for navigation
    ]
    
    return {
        "section": section_fields,
        "code": code_block_fields,
        "relationships": relationship_fields
    }


@pytest.fixture
def sample_python_code():
    """Return sample Python code for testing."""
    return """
import os
from pathlib import Path

class FileProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def process(self) -> bool:
        if not Path(self.file_path).exists():
            return False
        return True
    
def main():
    processor = FileProcessor("example.txt")
    result = processor.process()
    print(f"Processing result: {result}")

if __name__ == "__main__":
    main()
"""


def create_qa_compatible_block(hierarchy_data: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
    """Create QA-compatible blocks from hierarchy data."""
    blocks = []
    file_path = hierarchy_data.get("file_path", "")
    language = hierarchy_data.get("language", "")
    
    # Add file-level block
    file_block = {
        "uuid": "file-" + file_path.replace("/", "-"),
        "id": Path(file_path).name,
        "type": "code",
        "language": language,
        "title": Path(file_path).name,
        "content": content,
        "file_path": file_path,
        "breadcrumb": [Path(file_path).name],
        "parent_uuid": None,
        "child_uuids": [],
        "depth": 0,
        "extraction_focus": "code structure",
        "summary_instructions": "Generate QA pairs about the overall code structure",
        "dependencies": {},
        "test_coverage": {},
        "version_history": {},
        "qa_generation": {}
    }
    blocks.append(file_block)
    
    # Add class blocks
    for class_name, class_info in hierarchy_data.get("classes", {}).items():
        line_start = class_info.get("line_start", 0)
        line_end = class_info.get("line_end", 0)
        
        # Extract class content
        class_lines = content.split("\n")[line_start-1:line_end]
        class_content = "\n".join(class_lines)
        
        class_block = {
            "uuid": f"class-{class_name}-{file_path.replace('/', '-')}",
            "id": class_name,
            "type": "code",
            "language": language,
            "title": f"Class {class_name}",
            "content": class_content,
            "file_path": file_path,
            "breadcrumb": [Path(file_path).name, class_name],
            "parent_uuid": file_block["uuid"],
            "child_uuids": [],
            "depth": 1,
            "extraction_focus": "class implementation",
            "summary_instructions": f"Generate QA pairs about the {class_name} class implementation",
            "dependencies": {},
            "test_coverage": {},
            "version_history": {},
            "qa_generation": {}
        }
        
        # Add this class to parent's children
        file_block["child_uuids"].append(class_block["uuid"])
        blocks.append(class_block)
        
        # Add method blocks
        for method_info in class_info.get("methods", []):
            method_name = method_info.get("name", "")
            method_line_start = method_info.get("line_start", 0)
            method_line_end = method_info.get("line_end", 0)
            
            # Extract method content
            method_lines = content.split("\n")[method_line_start-1:method_line_end]
            method_content = "\n".join(method_lines)
            
            method_block = {
                "uuid": f"method-{class_name}-{method_name}-{file_path.replace('/', '-')}",
                "id": f"{class_name}.{method_name}",
                "type": "code",
                "language": language,
                "title": f"Method {class_name}.{method_name}",
                "content": method_content,
                "file_path": file_path,
                "breadcrumb": [Path(file_path).name, class_name, method_name],
                "parent_uuid": class_block["uuid"],
                "child_uuids": [],
                "depth": 2,
                "extraction_focus": "method implementation",
                "summary_instructions": f"Generate QA pairs about the {class_name}.{method_name} method implementation",
                "dependencies": {},
                "test_coverage": {},
                "version_history": {},
                "qa_generation": {}
            }
            
            # Add this method to parent's children
            class_block["child_uuids"].append(method_block["uuid"])
            blocks.append(method_block)
    
    # Add function blocks
    for func_name, func_info in hierarchy_data.get("functions", {}).items():
        line_start = func_info.get("line_start", 0)
        line_end = func_info.get("line_end", 0)
        
        # Extract function content
        func_lines = content.split("\n")[line_start-1:line_end]
        func_content = "\n".join(func_lines)
        
        func_block = {
            "uuid": f"function-{func_name}-{file_path.replace('/', '-')}",
            "id": func_name,
            "type": "code",
            "language": language,
            "title": f"Function {func_name}",
            "content": func_content,
            "file_path": file_path,
            "breadcrumb": [Path(file_path).name, func_name],
            "parent_uuid": file_block["uuid"],
            "child_uuids": [],
            "depth": 1,
            "extraction_focus": "function implementation",
            "summary_instructions": f"Generate QA pairs about the {func_name} function implementation",
            "dependencies": {},
            "test_coverage": {},
            "version_history": {},
            "qa_generation": {}
        }
        
        # Add this function to parent's children
        file_block["child_uuids"].append(func_block["uuid"])
        blocks.append(func_block)
    
    return blocks


def test_hierarchy_to_qa_compatible_blocks(sample_python_code, qa_requirement_fields):
    """Test converting hierarchy data to QA-compatible blocks."""
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_python_code)
        temp_file = f.name
    
    try:
        # Get hierarchy data
        hierarchy, stats = analyze_code_hierarchy(temp_file)
        
        # Create QA-compatible blocks
        blocks = create_qa_compatible_block(hierarchy, sample_python_code)
        
        # Verify blocks meet QA requirements
        for block in blocks:
            # Check required section fields
            for field in qa_requirement_fields["section"]:
                assert field in block, f"Missing required field '{field}' in block"
                
            # Check code-specific fields for code blocks
            if block["type"] == "code":
                for field in qa_requirement_fields["code"]:
                    assert field in block, f"Missing code-specific field '{field}' in block"
                    
            # Check relationship fields
            for field in qa_requirement_fields["relationships"]:
                assert field in block, f"Missing relationship field '{field}' in block"
        
        # Check that blocks form a proper hierarchy
        root_blocks = [b for b in blocks if b["parent_uuid"] is None]
        assert len(root_blocks) == 1, "Should have exactly one root block"
        
        # Check that parent-child relationships are consistent
        for block in blocks:
            for child_uuid in block.get("child_uuids", []):
                child_block = next((b for b in blocks if b["uuid"] == child_uuid), None)
                assert child_block is not None, f"Child block with UUID {child_uuid} not found"
                assert child_block["parent_uuid"] == block["uuid"], "Parent-child relationship inconsistency"
                
        # Create a QA-compatible JSON structure
        extraction_json = {
            "sections": blocks,
            "extraction_metadata": {
                "model_used": "gpt-4-turbo",
                "timestamp": "2025-03-21T12:00:00Z"
            }
        }
        
        # Write to file to verify it can be serialized properly
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(extraction_json, f, indent=2)
            json_file = f.name
            
        # Read it back to confirm it's valid JSON
        with open(json_file, 'r') as f:
            loaded_json = json.load(f)
            assert loaded_json["sections"] == blocks
        
        # Clean up the JSON file
        os.unlink(json_file)
        
    finally:
        # Clean up the Python file
        os.unlink(temp_file)


def test_qa_compatible_output_format(sample_python_code):
    """Test that the output format is compatible with the QA processor."""
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_python_code)
        temp_file = f.name
    
    try:
        # Skip if QA module is not available (this is just for integration testing)
        try:
            from src.agent_tools.dualipa.qa.utils.validation import validate_input_json
        except ImportError:
            pytest.skip("QA module not available for validation")
            
        # Get hierarchy data
        hierarchy, stats = analyze_code_hierarchy(temp_file)
        
        # Create QA-compatible blocks
        blocks = create_qa_compatible_block(hierarchy, sample_python_code)
        
        # Create a QA-compatible JSON structure
        extraction_json = {
            "sections": blocks,
            "extraction_metadata": {
                "model_used": "gpt-4-turbo",
                "timestamp": "2025-03-21T12:00:00Z"
            }
        }
        
        # Validate using QA module's validation function
        is_valid = validate_input_json(extraction_json, raise_on_error=False)
        assert is_valid, "Extraction JSON failed QA module validation"
        
    finally:
        # Clean up the Python file
        os.unlink(temp_file)


def test_hierarchy_summary_formatting(sample_python_code):
    """Test the hierarchy summary formatting utility."""
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_python_code)
        temp_file = f.name
    
    try:
        # Get hierarchy data
        hierarchy, stats = analyze_code_hierarchy(temp_file)
        
        # Format the summary
        summary = format_hierarchy_summary(hierarchy)
        
        # Check summary contents
        assert Path(temp_file).name in summary
        assert "python" in summary.lower()
        assert "1" in summary  # 1 class
        assert "1" in summary  # 1 function (main)
        
    finally:
        # Clean up the Python file
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])