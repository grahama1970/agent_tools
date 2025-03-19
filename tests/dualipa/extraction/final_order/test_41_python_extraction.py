"""
TEST EXPECTATIONS

test_extract_python_blocks:
Input: Python file with functions and classes
Expected Output:
{
    "code_blocks": > 0,
    "total_files": 1,
    "file_blocks": {
        "example.py": [
            {
                "block_type": "function",
                "name": "simple_function",
                "content": "def simple_function()..."
            },
            {
                "block_type": "class",
                "name": "SimpleClass",
                "content": "class SimpleClass..."
            }
        ]
    }
}

CRITICAL RULES:
1. Block Extraction Rules:
   - Each block must have a block_type
   - Each block must have a name
   - Each block must have content
   - Each block must preserve original formatting

2. Stats Tracking Rules:
   - Track total files processed
   - Track blocks per file
   - Track languages encountered
   - Track errors during extraction

3. Output File Rules:
   - All blocks must be written to output directory
   - All paths must be relative to output directory
   - Block files must have .py extension
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.code_extractor import (
        _extract_python_blocks,
        initialize_stats_dict
    )
    IMPORTS_AVAILABLE = True
except ImportError as import_error:
    import traceback
    print(f"IMPORT ERROR: {import_error}")
    print("Traceback:")
    traceback.print_exc()
    IMPORTS_AVAILABLE = False

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required code extractor modules not available")

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    # Clean up after the test
    shutil.rmtree(dir_path)

@pytest.fixture
def python_file_fixture(temp_dir):
    """Create a Python file with known functions and classes for testing."""
    py_file = Path(temp_dir) / "example.py"
    content = """
import os
import sys
from typing import List, Dict, Any

def simple_function():
    \"\"\"A simple function that returns a string.\"\"\"
    return "This is a simple function"

def function_with_args(arg1: str, arg2: int = 0) -> str:
    \"\"\"
    A function with arguments and type hints.
    
    Args:
        arg1: A string argument
        arg2: An integer argument with default value
        
    Returns:
        A formatted string
    \"\"\"
    return f"{arg1}: {arg2}"

class SimpleClass:
    \"\"\"A simple class with methods.\"\"\"
    
    def __init__(self, name: str):
        \"\"\"Initialize with a name.\"\"\"
        self.name = name
    
    def get_name(self) -> str:
        \"\"\"Return the name.\"\"\"
        return self.name
    
    def set_name(self, name: str) -> None:
        \"\"\"Set the name.\"\"\"
        self.name = name

# A global variable
GLOBAL_CONSTANT = "This is a global constant"

if __name__ == "__main__":
    simple_function()
"""
    
    py_file.write_text(content)
    return py_file

def test_extract_python_blocks(python_file_fixture, temp_dir):
    """Test extracting Python blocks."""
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=python_file_fixture, output_dir=output_dir)
    
    content = python_file_fixture.read_text()
    
    num_blocks = _extract_python_blocks(python_file_fixture, content, output_dir, stats)
    
    # Verify block count
    assert num_blocks > 0, "No blocks were extracted"
    assert stats["code_blocks"] == num_blocks, "Stats code_blocks count doesn't match returned count"
    
    # Verify file tracking
    assert stats["total_files"] == 1, "Total file count should be 1"
    assert str(python_file_fixture) in stats["file_blocks"], "File not in file_blocks dictionary"
    
    # Verify extracted blocks
    blocks = stats["file_blocks"][str(python_file_fixture)]
    assert len(blocks) > 0, "No blocks were extracted"
    
    # Verify function blocks
    function_blocks = [b for b in blocks if b.get("block_type") == "function"]
    assert len(function_blocks) >= 2, "Expected at least 2 functions"
    function_names = {b.get("name", "") for b in function_blocks}
    assert "simple_function" in function_names, "simple_function not found"
    assert "function_with_args" in function_names, "function_with_args not found"
    
    # Verify class blocks
    class_blocks = [b for b in blocks if b.get("block_type") == "class"]
    assert len(class_blocks) == 1, "Expected 1 class"
    assert class_blocks[0].get("name") == "SimpleClass", "SimpleClass not found"
    
    # Verify output files
    blocks_dir = output_dir / "blocks" / "code" / "python"
    assert blocks_dir.exists(), "Blocks directory not created"
    block_files = list(blocks_dir.glob("*.py"))
    assert len(block_files) == num_blocks, "Number of block files doesn't match block count"
    
    # Verify each block has required fields
    for block in blocks:
        assert "block_type" in block, f"Block missing block_type: {block}"
        assert "name" in block, f"Block missing name: {block}"
        assert "content" in block, f"Block missing content: {block}"
        
        # Verify block file exists
        output_file = block.get("output_file")
        assert output_file, f"Block missing output_file: {block}"
        assert Path(output_file).exists(), f"Output file does not exist: {output_file}"
        
        # Verify content formatting
        content = Path(output_file).read_text()
        assert content.strip(), "Block file is empty"
        assert not content.startswith("\n"), "Block has leading newline"
        assert not content.endswith("\n\n"), "Block has multiple trailing newlines"

def test_extract_python_script_level(temp_dir):
    """Test extraction of script-level Python code."""
    script_file = Path(temp_dir) / "script.py"
    script_content = """#!/usr/bin/env python3
import sys

# Process command line arguments
args = sys.argv[1:]
for arg in args:
    print(f"Processing: {arg}")

if __name__ == "__main__":
    print("Running script")
"""
    script_file.write_text(script_content)
    
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=script_file, output_dir=output_dir)
    
    num_blocks = _extract_python_blocks(script_file, script_content, output_dir, stats)
    
    # Script-level code should be extracted as one block
    assert num_blocks == 1, "Script-level code should be extracted as one block"
    assert stats["code_blocks"] == 1, "Stats should count script-level block"
    
    blocks = stats["file_blocks"][str(script_file)]
    assert len(blocks) == 1, "Expected one script-level block"
    
    block = blocks[0]
    assert block["block_type"] == "script", "Block should be marked as script type"
    assert "content" in block, "Block should contain the script content"
    
    # Verify the block file
    output_file = Path(block["output_file"])
    assert output_file.exists(), "Script block file should exist"
    content = output_file.read_text()
    assert "#!/usr/bin/env python3" in content, "Should preserve shebang line"
    assert "if __name__ == \"__main__\":" in content, "Should preserve main guard"

def test_extract_python_with_decorators(temp_dir):
    """Test extraction of Python code with decorators."""
    decorated_file = Path(temp_dir) / "decorated.py"
    decorated_content = """
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def decorated_function():
    \"\"\"A decorated function.\"\"\"
    return "Hello from decorated function"

class DecoratedClass:
    @property
    def name(self):
        return self._name
        
    @name.setter
    def name(self, value):
        self._name = value
"""
    decorated_file.write_text(decorated_content)
    
    output_dir = Path(temp_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    stats = initialize_stats_dict(source=decorated_file, output_dir=output_dir)
    
    num_blocks = _extract_python_blocks(decorated_file, decorated_content, output_dir, stats)
    
    # Verify decorator handling
    blocks = stats["file_blocks"][str(decorated_file)]
    
    # Find decorated function
    decorated_funcs = [b for b in blocks if b["name"] == "decorated_function"]
    assert len(decorated_funcs) == 1, "Decorated function not found"
    dec_func = decorated_funcs[0]
    assert "@my_decorator" in dec_func["content"], "Decorator not preserved in function"
    
    # Find decorated class
    decorated_classes = [b for b in blocks if b["name"] == "DecoratedClass"]
    assert len(decorated_classes) == 1, "Decorated class not found"
    dec_class = decorated_classes[0]
    assert "@property" in dec_class["content"], "Property decorator not preserved"
    assert "@name.setter" in dec_class["content"], "Setter decorator not preserved"

if __name__ == "__main__":
    pytest.main([__file__]) 