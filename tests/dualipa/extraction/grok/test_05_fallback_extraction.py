"""
Tests for fallback extraction for unsupported languages.
Depends on: Tree-sitter failing gracefully for unsupported languages.
"""

import pytest
import tempfile
from pathlib import Path
from agent_tools.dualipa.code_extractor import _extract_generic_blocks
from agent_tools.dualipa.utils import initialize_stats_dict

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)

def test_extract_generic_blocks(temp_dir):
    """Test generic block extraction for an unsupported language."""
    code = """
function hello() {
    print("Hello");
}
"""
    file_path = temp_dir / "test.xyz"
    with open(file_path, "w") as f:
        f.write(code)
    stats = initialize_stats_dict()
    num_blocks = _extract_generic_blocks(file_path, code, temp_dir, stats, "xyz")
    assert isinstance(num_blocks, int), "Should return a number"
    assert len(stats["errors"]) == 0, "No errors should occur"
    blocks_dir = temp_dir / "blocks" / "code" / "xyz"
    if blocks_dir.exists():
        assert len(list(blocks_dir.glob("*.xyz"))) > 0, "Block files should be created if extracted"