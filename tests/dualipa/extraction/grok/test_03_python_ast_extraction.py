"""
Tests for Python code extraction using AST.
Depends on: Repository operations working to provide test files.
"""

import os
import pytest
import tempfile
from pathlib import Path
from agent_tools.dualipa.code_extractor import _extract_python_blocks
from agent_tools.dualipa.utils import initialize_stats_dict

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)

def test_extract_python_blocks(temp_dir):
    """Test Python block extraction from a simple file."""
    code = """
def test_func():
    return "Hello"
class TestClass:
    def method(self):
        pass
"""
    file_path = temp_dir / "test.py"
    with open(file_path, "w") as f:
        f.write(code)
    stats = initialize_stats_dict()
    num_blocks = _extract_python_blocks(file_path, code, temp_dir, stats)
    assert num_blocks == 2, f"Expected 2 blocks, got {num_blocks}"
    assert stats["extraction"]["blocks"]["total"] == 2, "Stats not updated"
    blocks_dir = temp_dir / "blocks" / "code" / "python"
    assert blocks_dir.exists(), "Blocks directory not created"
    assert len(list(blocks_dir.glob("*.py"))) == 2, "Block files not created"