"""
TEST EXPECTATIONS

test_extract_repository:
Input: Repository path with mixed content
Expected Output:
{
    "code_blocks": > 0,
    "total_files": > 0,
    "file_blocks": {
        "file1.py": [...],
        "file2.js": [...],
        "doc.md": [...]
    }
}

CRITICAL RULES:
1. Repository Extraction Rules:
   - Must handle multiple file types
   - Must process nested directories
   - Must skip binary files
   - Must skip ignored files/directories

2. Stats Tracking Rules:
   - Track total files processed
   - Track files by language
   - Track blocks by language
   - Track errors by file

3. Output File Rules:
   - Must create output directory if not exists
   - Must maintain directory structure
   - Must handle file name collisions
   - Must clean output directory before extraction
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys
import shutil

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.code_extractor import (
        extract_repository,
        initialize_stats_dict,
        _is_code_file,
        _extract_generic_blocks,
        _save_stats_to_json
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
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    yield tmp_path
    shutil.rmtree(tmp_path)

@pytest.fixture
def stats_dict():
    """Initialize a stats dictionary."""
    return {
        "total_files": 0,
        "code_files": 0,
        "documentation_files": 0,
        "code_blocks": 0,
        "doc_blocks": 0,
        "languages": {},
        "file_types": {},
        "file_blocks": {},
        "errors": []
    }

@pytest.fixture
def mixed_repo_fixture(temp_dir):
    """Create a repository with mixed content for testing."""
    repo_dir = Path(temp_dir) / "test_repo"
    repo_dir.mkdir(exist_ok=True)
    
    # Create Python files
    python_dir = repo_dir / "python"
    python_dir.mkdir(exist_ok=True)
    
    with open(python_dir / "main.py", "w") as f:
        f.write("""
def main():
    print("Hello from Python!")

if __name__ == "__main__":
    main()
""")
    
    with open(python_dir / "utils.py", "w") as f:
        f.write("""
def greet(name: str) -> str:
    return f"Hello, {name}!"
""")
    
    # Create JavaScript files
    js_dir = repo_dir / "javascript"
    js_dir.mkdir(exist_ok=True)
    
    with open(js_dir / "app.js", "w") as f:
        f.write("""
function init() {
    console.log("App initialized");
}

init();
""")
    
    with open(js_dir / "utils.js", "w") as f:
        f.write("""
function greet(name) {
    return `Hello, ${name}!`;
}

module.exports = { greet };
""")
    
    # Create documentation
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    with open(docs_dir / "README.md", "w") as f:
        f.write("""# Test Repository

## Python Examples

```python
def example():
    return "Hello"
```

## JavaScript Examples

```javascript
function example() {
    return "Hello";
}
```
""")
    
    # Create binary and ignored files
    with open(repo_dir / "binary.bin", "wb") as f:
        f.write(b"\x00\x01\x02\x03")
    
    with open(repo_dir / ".gitignore", "w") as f:
        f.write("*.bin\n")
    
    return repo_dir

def test_extract_repository(mixed_repo_fixture, temp_dir):
    """Test extracting code from a repository with mixed content."""
    output_dir = Path(temp_dir) / "output"
    
    stats = extract_repository(mixed_repo_fixture, output_dir)
    
    # Verify stats structure
    assert isinstance(stats, dict), "Stats should be a dictionary"
    assert "code_blocks" in stats, "Stats missing code_blocks count"
    assert "total_files" in stats, "Stats missing total_files count"
    assert "file_blocks" in stats, "Stats missing file_blocks dictionary"
    
    # Verify file counts
    assert stats["total_files"] > 0, "No files were processed"
    assert len(stats["file_blocks"]) > 0, "No files were extracted"
    
    # Verify Python files
    python_files = [f for f in stats["file_blocks"].keys() if str(f).endswith(".py")]
    assert len(python_files) >= 2, "Expected at least 2 Python files"
    
    # Verify JavaScript files
    js_files = [f for f in stats["file_blocks"].keys() if str(f).endswith(".js")]
    assert len(js_files) >= 2, "Expected at least 2 JavaScript files"
    
    # Verify Markdown files
    md_files = [f for f in stats["file_blocks"].keys() if str(f).endswith(".md")]
    assert len(md_files) >= 1, "Expected at least 1 Markdown file"
    
    # Verify output directory structure
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "blocks" / "code" / "python").exists(), "Python output directory not created"
    assert (output_dir / "blocks" / "code" / "javascript").exists(), "JavaScript output directory not created"
    assert (output_dir / "blocks" / "sections").exists(), "Sections output directory not created"
    
    # Verify binary files are skipped
    binary_files = [f for f in stats["file_blocks"].keys() if str(f).endswith(".bin")]
    assert len(binary_files) == 0, "Binary files should be skipped"

def test_is_code_file():
    """Test code file detection."""
    assert _is_code_file("test.py") is True
    assert _is_code_file("test.js") is True
    assert _is_code_file("test.ts") is True
    assert _is_code_file("test.md") is False
    assert _is_code_file("test.txt") is False

def test_extract_generic_blocks(temp_dir, stats_dict):
    """Test generic block extraction from a Python file."""
    # Create a test file
    test_file = temp_dir / "test.py"
    content = """def function1():
    print("Hello")

def function2():
    print("World")

# Some comment
x = 1
y = 2
"""
    test_file.write_text(content)
    
    # Extract blocks
    blocks = _extract_generic_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict,
        "python"
    )
    
    assert blocks > 0
    assert stats_dict["code_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check block files were created
    blocks_dir = temp_dir / "code_blocks" / "python"
    assert blocks_dir.exists()
    assert len(list(blocks_dir.glob("*.py"))) > 0

def test_save_stats(temp_dir, stats_dict):
    """Test saving stats to JSON."""
    stats_dict["total_files"] = 1
    stats_dict["code_files"] = 1
    
    _save_stats_to_json(stats_dict, str(temp_dir))
    
    stats_file = temp_dir / "extraction_stats.json"
    assert stats_file.exists()
    
    # Verify file contents
    import json
    with open(stats_file) as f:
        saved_stats = json.load(f)
    
    assert saved_stats["total_files"] == 1
    assert saved_stats["code_files"] == 1

def test_extract_generic_blocks_error_handling(temp_dir, stats_dict):
    """Test error handling in generic block extraction."""
    # Create an invalid file path
    invalid_file = temp_dir / "nonexistent.py"
    
    # Try to extract blocks
    blocks = _extract_generic_blocks(
        invalid_file,
        "some content",
        temp_dir,
        stats_dict,
        "python"
    )
    
    assert blocks == 0
    assert len(stats_dict["errors"]) > 0

def test_extract_repository_error_handling(temp_dir):
    """Test error handling during repository extraction."""
    repo_dir = Path(temp_dir) / "error_repo"
    repo_dir.mkdir(exist_ok=True)
    
    # Create a Python file with syntax error
    with open(repo_dir / "error.py", "w") as f:
        f.write("""
def broken_function()
    print("Missing colon")
""")
    
    # Create a JavaScript file with syntax error
    with open(repo_dir / "error.js", "w") as f:
        f.write("""
function broken_function( {
    console.log("Missing parenthesis");
}
""")
    
    output_dir = Path(temp_dir) / "output"
    
    stats = extract_repository(repo_dir, output_dir)
    
    # Verify error tracking
    assert "errors" in stats, "Stats should track errors"
    assert len(stats["errors"]) > 0, "Errors should be recorded"
    
    # Verify processing continues despite errors
    assert stats["total_files"] > 0, "Files should still be processed"
    assert len(stats["file_blocks"]) > 0, "Valid blocks should still be extracted"

def test_extract_repository_with_nested_structure(temp_dir):
    """Test extracting code from a deeply nested repository structure."""
    repo_dir = Path(temp_dir) / "nested_repo"
    repo_dir.mkdir(exist_ok=True)
    
    # Create nested directory structure
    current_dir = repo_dir
    for depth in range(5):
        current_dir = current_dir / f"level_{depth}"
        current_dir.mkdir(exist_ok=True)
        
        # Add a Python file
        with open(current_dir / f"file_{depth}.py", "w") as f:
            f.write(f"""
def function_{depth}():
    return "Level {depth}"
""")
        
        # Add a JavaScript file
        with open(current_dir / f"file_{depth}.js", "w") as f:
            f.write(f"""
function function_{depth}() {{
    return "Level {depth}";
}}
""")
    
    output_dir = Path(temp_dir) / "output"
    
    stats = extract_repository(repo_dir, output_dir)
    
    # Verify all levels are processed
    assert stats["total_files"] >= 10, "Expected at least 10 files (2 per level)"
    
    # Verify directory structure is maintained
    for depth in range(5):
        level_path = output_dir / "blocks" / "code"
        
        # Check Python files
        python_files = list((level_path / "python").glob(f"**/*level_{depth}*"))
        assert len(python_files) > 0, f"No Python files found at level {depth}"
        
        # Check JavaScript files
        js_files = list((level_path / "javascript").glob(f"**/*level_{depth}*"))
        assert len(js_files) > 0, f"No JavaScript files found at level {depth}"

if __name__ == "__main__":
    pytest.main([__file__]) 