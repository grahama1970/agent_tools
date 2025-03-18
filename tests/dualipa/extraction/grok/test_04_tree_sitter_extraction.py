"""
Tests for Tree-sitter-based extraction for supported languages (JS/TS).
Depends on: Repository operations to provide files, Python AST working as a baseline.
"""

import pytest
import tempfile
from pathlib import Path
from agent_tools.dualipa.code_extractor import _extract_js_ts_blocks
from agent_tools.dualipa.utils import initialize_stats_dict

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)

def test_extract_javascript_blocks(temp_dir):
    """Test JavaScript block extraction."""
    code = """
function greet(name) {
    return `Hello, ${name}`;
}
class Person {
    constructor(name) {
        this.name = name;
    }
}
"""
    file_path = temp_dir / "test.js"
    with open(file_path, "w") as f:
        f.write(code)
    stats = initialize_stats_dict()
    num_blocks = _extract_js_ts_blocks(file_path, code, temp_dir, stats, "javascript")
    assert num_blocks >= 2, f"Expected at least 2 blocks, got {num_blocks}"
    blocks_dir = temp_dir / "blocks" / "code" / "javascript"
    assert blocks_dir.exists(), "Blocks directory not created"
    assert len(list(blocks_dir.glob("*.js"))) >= 2, "Block files not created"