"""
Simple focused test for Python code extraction.

This test verifies that Python code blocks are correctly extracted using AST.
"""

import os
import tempfile
from pathlib import Path

# Direct import of only what we need
from agent_tools.dualipa.code_extractor import _extract_python_blocks

# Get path to sample files
# Update path since tests have been moved
RESOURCES_DIR = Path(__file__).parent.parent.parent.parent / "src" / "agent_tools" / "dualipa" / "resources" / "templates"
SAMPLE_PYTHON = RESOURCES_DIR / "sample_python.py"

def test_python_extraction_direct():
    """Test Python extraction directly without other components."""
    # Verify sample file exists
    assert SAMPLE_PYTHON.exists(), f"Sample file {SAMPLE_PYTHON} not found"
    
    # Read the content
    with open(SAMPLE_PYTHON, 'r') as f:
        content = f.read()
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Initialize stats dictionary
        stats = {"code_blocks": 0, "errors": []}
        
        # Call the function directly
        num_blocks = _extract_python_blocks(SAMPLE_PYTHON, content, output_dir, stats)
        
        # Verify blocks were extracted
        assert num_blocks > 0, "No Python blocks were extracted"
        assert stats["code_blocks"] > 0, "Code blocks counter not updated"
        print(f"Extracted {num_blocks} Python blocks")
        
        # Check output files
        blocks_dir = output_dir / "blocks" / "code" / "python"
        assert blocks_dir.exists(), "Python blocks directory not created"
        
        # Count files
        block_files = list(blocks_dir.glob("*.py"))
        assert len(block_files) == num_blocks, "Number of files doesn't match expected"
        
        # Check file contents - should have expected functions
        found_greet = False
        found_calculator = False
        
        for block_file in block_files:
            with open(block_file, 'r') as f:
                content = f.read()
                if "def greet" in content:
                    found_greet = True
                if "class Calculator" in content:
                    found_calculator = True
        
        assert found_greet, "Function 'greet' not found in extracted blocks"
        assert found_calculator, "Class 'Calculator' not found in extracted blocks"
        
        # Verify there are no errors
        assert not stats["errors"], f"Extraction produced errors: {stats['errors']}"

if __name__ == "__main__":
    # Run the test directly
    try:
        test_python_extraction_direct()
        print("✅ Test passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}") 