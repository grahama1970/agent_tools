"""
Tests for block verification functionality.

This module tests the verification of extracted code blocks to ensure
they meet quality standards and contain valid code.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import pytest

from agent_tools.dualipa.code_extractor import (
    extract_repository,
    _extract_python_blocks,
    _extract_markdown_blocks,
    _verify_code_block
)

# Path to test resources
RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "src" / "agent_tools" / "dualipa" / "resources" / "templates"

@pytest.fixture
def python_code_block():
    """Sample Python code block."""
    return {
        "id": "func_1",
        "language": "python",
        "content": """def hello_world():
    \"\"\"Say hello to the world.\"\"\"
    return "Hello, World!"
""",
        "path": "sample.py",
        "start_line": 1,
        "end_line": 3,
        "type": "function"
    }

@pytest.fixture
def invalid_python_block():
    """Invalid Python code block with syntax error."""
    return {
        "id": "invalid_1",
        "language": "python",
        "content": """def broken_function()
    \"\"\"Missing colon in function definition.\"\"\"
    return "This won't work!"
""",
        "path": "broken.py",
        "start_line": 1,
        "end_line": 3,
        "type": "function"
    }

def test_verify_valid_python_block(python_code_block):
    """Test verification of a valid Python code block."""
    result = _verify_code_block(python_code_block)
    assert result["is_valid"] is True
    assert "errors" not in result or len(result["errors"]) == 0

def test_verify_invalid_python_block(invalid_python_block):
    """Test verification of an invalid Python code block."""
    result = _verify_code_block(invalid_python_block)
    assert result["is_valid"] is False
    assert "errors" in result
    assert len(result["errors"]) > 0

def test_block_verification_in_extraction():
    """Test block verification during extraction process."""
    # Get the path to sample Python file
    sample_file = RESOURCES_DIR / "sample_python.py"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks from the sample file with verification
        stats = extract_repository(
            source=str(sample_file),
            output_path=temp_dir,
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True,
            verify_blocks=True
        )
        
        # Check verification statistics
        assert "verified_blocks" in stats
        assert stats["verified_blocks"] > 0
        
        # Read the blocks from the output directory
        blocks_path = Path(temp_dir) / "blocks" / "python"
        assert blocks_path.exists()
        
        # At least one block should have verification metadata
        block_files = list(blocks_path.glob("*.json"))
        assert len(block_files) > 0
        
        # Check verification metadata in at least one block
        with open(block_files[0], "r") as f:
            block_data = json.load(f)
            assert "verification" in block_data
            assert "is_valid" in block_data["verification"]

def test_verification_reject_empty_blocks():
    """Test that verification rejects empty or too-short blocks."""
    empty_block = {
        "id": "empty_1",
        "language": "python",
        "content": "",
        "path": "empty.py",
        "start_line": 1,
        "end_line": 1,
        "type": "function"
    }
    
    short_block = {
        "id": "short_1",
        "language": "python",
        "content": "# Just a comment",
        "path": "short.py",
        "start_line": 1,
        "end_line": 1,
        "type": "function"
    }
    
    # Verify empty block
    empty_result = _verify_code_block(empty_block)
    assert empty_result["is_valid"] is False
    assert "empty" in " ".join(empty_result["errors"]).lower()
    
    # Verify short block
    short_result = _verify_code_block(short_block)
    assert short_result["is_valid"] is False
    assert "short" in " ".join(short_result["errors"]).lower() or "trivial" in " ".join(short_result["errors"]).lower() 