"""
Tests for output examples and formatting.

This module tests the output examples and formatting capabilities
ensuring that the code extraction results are properly formatted
and match expected output structures.
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
    format_output_as_json,
    format_output_as_md,
    format_output_as_html
)

# Path to test resources
RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "src" / "agent_tools" / "dualipa" / "resources" / "templates"

@pytest.fixture
def sample_extraction_results():
    """Sample extraction results."""
    return {
        "blocks": [
            {
                "id": "func_1",
                "language": "python",
                "content": "def hello_world():\n    \"\"\"Say hello to the world.\"\"\"\n    return \"Hello, World!\"",
                "path": "sample.py",
                "start_line": 1,
                "end_line": 3,
                "type": "function"
            },
            {
                "id": "class_1",
                "language": "python",
                "content": "class TestClass:\n    \"\"\"A test class.\"\"\"\n    def __init__(self):\n        self.value = 42",
                "path": "sample.py",
                "start_line": 5,
                "end_line": 8,
                "type": "class"
            }
        ],
        "stats": {
            "total_files": 1,
            "code_files": 1,
            "documentation_files": 0,
            "code_blocks": 2,
            "languages": {"python": 1}
        }
    }

def test_json_output_format(sample_extraction_results):
    """Test JSON output formatting."""
    # Format results as JSON
    json_output = format_output_as_json(sample_extraction_results)
    
    # Parse the JSON to verify it's valid
    parsed = json.loads(json_output)
    
    # Verify the structure matches the input
    assert "blocks" in parsed
    assert len(parsed["blocks"]) == len(sample_extraction_results["blocks"])
    assert "stats" in parsed
    
    # Verify specific fields are preserved
    assert parsed["blocks"][0]["id"] == "func_1"
    assert parsed["blocks"][0]["language"] == "python"
    assert parsed["blocks"][0]["type"] == "function"
    assert parsed["stats"]["code_blocks"] == 2

def test_markdown_output_format(sample_extraction_results):
    """Test Markdown output formatting."""
    # Format results as Markdown
    md_output = format_output_as_md(sample_extraction_results)
    
    # Verify markdown contains expected sections
    assert "# Extraction Results" in md_output
    assert "## Statistics" in md_output
    assert "## Code Blocks" in md_output
    
    # Check for code block formatting
    assert "```python" in md_output
    assert "def hello_world()" in md_output
    assert "class TestClass" in md_output
    
    # Verify stats are included
    assert "Total Files: 1" in md_output
    assert "Code Blocks: 2" in md_output

def test_html_output_format(sample_extraction_results):
    """Test HTML output formatting."""
    # Format results as HTML
    html_output = format_output_as_html(sample_extraction_results)
    
    # Verify HTML structure
    assert "<!DOCTYPE html>" in html_output
    assert "<html" in html_output
    assert "<head" in html_output
    assert "<body" in html_output
    
    # Check for content sections
    assert "<h1>Extraction Results</h1>" in html_output
    assert "<h2>Statistics</h2>" in html_output
    assert "<h2>Code Blocks</h2>" in html_output
    
    # Verify code blocks are properly formatted
    assert "<pre><code class=\"language-python\">" in html_output
    assert "def hello_world()" in html_output
    assert "class TestClass" in html_output

def test_output_format_from_extraction():
    """Test output formatting with real extraction results."""
    # Get the path to sample Python file
    sample_file = RESOURCES_DIR / "sample_python.py"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist"
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract code blocks
        temp_dir_path = Path(temp_dir)
        stats = extract_repository(
            source=str(sample_file),
            output_path=str(temp_dir_path),
            extract_documentation=False,
            extract_code=True,
            extract_blocks=True
        )
        
        # Check that the extraction produced results
        assert stats["code_blocks"] > 0
        
        # Check for output files
        json_path = temp_dir_path / "extraction_stats.json"
        assert json_path.exists()
        
        # Load and validate the JSON output
        with open(json_path, "r") as f:
            json_data = json.load(f)
            assert "total_files" in json_data
            assert "code_blocks" in json_data
            assert json_data["code_blocks"] > 0
            
        # Check that we can create alternative formats
        md_path = temp_dir_path / "extraction_results.md"
        with open(md_path, "w") as f:
            f.write(format_output_as_md({
                "blocks": [], 
                "stats": stats
            }))
        assert md_path.exists()
        
        # Read back the markdown and verify basic structure
        with open(md_path, "r") as f:
            md_content = f.read()
            assert "# Extraction Results" in md_content
            assert "## Statistics" in md_content 