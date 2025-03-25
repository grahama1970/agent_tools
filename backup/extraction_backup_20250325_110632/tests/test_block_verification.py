"""
Tests for code block verification functionality with real-world repositories.

These tests verify the validation and verification of code blocks extracted
from real-world repositories.
"""

import os
import sys
import tempfile
import uuid
import textwrap
from pathlib import Path
import pytest

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"

# Test if the repositories exist
HAS_TEST_REPOS = RUST_ANALYZER_PATH.exists() and REACT_PATH.exists()
if not HAS_TEST_REPOS:
    print(f"Warning: Test repositories not found at: {RUST_ANALYZER_PATH}, {REACT_PATH}")
    print("Some tests will be skipped")

# Import the required modules
try:
    from agent_tools.dualipa.extraction.extractors.code.code_extractor import (
        extract_code_blocks,
        validate_block,
        verify_block
    )
    from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats
    
    HAS_DEPENDENCIES = True
except ImportError as e:
    # Instead of silently skipping, fail loudly with a clear error message
    raise ImportError(f"Required verification modules not available: {e}. Fix the dependencies to run these tests.")

# Skip all tests if dependencies are not available
# pytestmark = pytest.mark.skipif(

def test_python_block_verification():
    """Test verification of Python code blocks."""
    # Create a valid Python block
    valid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "example_function",
        "content": "def example_function():\n    return 42\n",
        "line_start": 1,
        "line_end": 2,
        "metadata": {
            "language": "python",
            "file": "test.py",
            "imports": [],
            "exports": []
        }
    }
    
    # Create an invalid Python block
    invalid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "invalid_function",
        "content": "def invalid_function():\n    return }\n",  # Invalid syntax
        "line_start": 1,
        "line_end": 2,
        "metadata": {
            "language": "python",
            "file": "test.py",
            "imports": [],
            "exports": []
        }
    }
    
    # Test verification
    assert validate_block(valid_block), "Valid Python block should pass validation"
    assert verify_block(valid_block), "Valid Python block should pass verification"
    assert not verify_block(invalid_block), "Invalid Python block should fail verification"
    
    print("Python block verification tests passed!")

def test_javascript_block_verification():
    """Test verification of JavaScript code blocks."""
    # Create a valid JavaScript block
    valid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "exampleFunction",
        "content": "function exampleFunction() {\n  return 42;\n}\n",
        "line_start": 1,
        "line_end": 3,
        "metadata": {
            "language": "javascript",
            "file": "test.js",
            "imports": [],
            "exports": []
        }
    }
    
    # Create a valid React component block
    react_block = {
        "uuid": str(uuid.uuid4()),
        "type": "react_component",
        "name": "ExampleComponent",
        "content": textwrap.dedent('''
            import React from 'react';
            
            export function ExampleComponent() {
                return <div>Hello World</div>;
            }
        ''').strip(),
        "line_start": 1,
        "line_end": 5,
        "metadata": {
            "language": "javascript",
            "file": "test.jsx",
            "framework": "react",
            "imports": ["import React from 'react'"],
            "exports": ["export function ExampleComponent"]
        }
    }
    
    # Create an invalid JavaScript block
    invalid_block = {
        "uuid": str(uuid.uuid4()),
        "type": "function",
        "name": "invalidFunction",
        "content": "function invalidFunction() {\n  return }\n",  # Missing semicolon
        "line_start": 1,
        "line_end": 2,
        "metadata": {
            "language": "javascript",
            "file": "test.js",
            "imports": [],
            "exports": []
        }
    }
    
    # Test verification
    assert validate_block(valid_block), "Valid JavaScript block should pass validation"
    assert verify_block(valid_block), "Valid JavaScript block should pass verification"
    assert validate_block(react_block), "Valid React component should pass validation"
    assert verify_block(react_block), "Valid React component should pass verification"
    assert not verify_block(invalid_block), "Invalid JavaScript block should fail verification"
    
    print("JavaScript block verification tests passed!")

def test_block_extraction():
    """Test extraction and verification of code blocks."""
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as output_dir:
        output_path = Path(output_dir)
        
        # Test Python extraction
        python_content = textwrap.dedent('''
            def example_function():
                return 42

            class ExampleClass:
                def method(self):
                    pass
        ''').strip()
        
        # Write Python content to file
        python_file = output_path / "test.py"
        with open(python_file, "w") as f:
            f.write(python_content)
        
        # Extract Python blocks
        blocks = extract_code_blocks(str(python_file), output_path)
        assert len(blocks) > 0, "Should extract Python blocks"
        
        # Test JavaScript/React extraction
        js_content = textwrap.dedent('''
            import React from 'react';

            export function ExampleComponent() {
                return <div>Hello World</div>;
            }

            class ExampleClass {
                method() {
                    return true;
                }
            }
        ''').strip()
        
        # Write JavaScript content to file
        js_file = output_path / "test.jsx"
        with open(js_file, "w") as f:
            f.write(js_content)
        
        # Extract JavaScript blocks
        blocks = extract_code_blocks(str(js_file), output_path)
        assert len(blocks) > 0, "Should extract JavaScript blocks"
        
        # Verify each block
        for block in blocks:
            assert validate_block(block), f"Block {block['name']} should pass validation"
            assert verify_block(block), f"Block {block['name']} should pass verification"
            assert "uuid" in block, "Block should have UUID"
            assert "metadata" in block, "Block should have metadata"
            assert "language" in block["metadata"], "Block should have language in metadata"
            assert "imports" in block["metadata"], "Block should have imports in metadata"
            assert "exports" in block["metadata"], "Block should have exports in metadata"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 