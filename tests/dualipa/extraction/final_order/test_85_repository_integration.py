"""
TEST EXPECTATIONS

test_repository_integration:
Input: Repository with mixed content and nested structure
Expected Output:
{
    "code_blocks": > 0,
    "total_files": > 0,
    "file_blocks": {...},
    "errors": [],
    "stats": {
        "languages": {...},
        "file_types": {...},
        "block_types": {...}
    }
}

CRITICAL RULES:
1. Integration Rules:
   - Must handle all supported languages
   - Must process all file types correctly
   - Must maintain directory structure
   - Must track all statistics

2. Performance Rules:
   - Must handle large repositories
   - Must process files in parallel
   - Must manage memory efficiently
   - Must clean up temporary files

3. Error Handling Rules:
   - Must continue on file errors
   - Must log all errors
   - Must maintain partial results
   - Must clean up on failure
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agent_tools.dualipa.code_extractor import (
        extract_repository,
        initialize_stats_dict,
        _extract_js_ts_blocks,
        _extract_python_blocks,
        _extract_markdown_blocks
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
def complex_repo_fixture(temp_dir):
    """Create a complex repository structure for integration testing."""
    repo_dir = Path(temp_dir) / "complex_repo"
    repo_dir.mkdir(exist_ok=True)
    
    # Create nested directory structure
    for lang in ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "ruby"]:
        lang_dir = repo_dir / lang
        lang_dir.mkdir(exist_ok=True)
        
        # Create source directory
        src_dir = lang_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        # Create test directory
        test_dir = lang_dir / "tests"
        test_dir.mkdir(exist_ok=True)
        
        # Create docs directory
        docs_dir = lang_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
    
    # Add Python files
    with open(repo_dir / "python/src/main.py", "w") as f:
        f.write("""
def main():
    print("Hello from Python!")

if __name__ == "__main__":
    main()
""")
    
    with open(repo_dir / "python/tests/test_main.py", "w") as f:
        f.write("""
def test_main():
    assert True, "Test passes"
""")
    
    # Add JavaScript files
    with open(repo_dir / "javascript/src/app.js", "w") as f:
        f.write("""
function init() {
    console.log("App initialized");
}

init();
""")
    
    with open(repo_dir / "javascript/tests/app.test.js", "w") as f:
        f.write("""
test('app initialization', () => {
    expect(true).toBe(true);
});
""")
    
    # Add TypeScript files
    with open(repo_dir / "typescript/src/service.ts", "w") as f:
        f.write("""
interface Service {
    start(): void;
    stop(): void;
}

class AppService implements Service {
    start() {
        console.log("Service started");
    }
    
    stop() {
        console.log("Service stopped");
    }
}
""")
    
    # Add Java files
    with open(repo_dir / "java/src/Main.java", "w") as f:
        f.write("""
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from Java!");
    }
}
""")
    
    # Add C++ files
    with open(repo_dir / "cpp/src/main.cpp", "w") as f:
        f.write("""
#include <iostream>

int main() {
    std::cout << "Hello from C++!" << std::endl;
    return 0;
}
""")
    
    # Add Go files
    with open(repo_dir / "go/src/main.go", "w") as f:
        f.write("""
package main

import "fmt"

func main() {
    fmt.Println("Hello from Go!")
}
""")
    
    # Add Rust files
    with open(repo_dir / "rust/src/main.rs", "w") as f:
        f.write("""
fn main() {
    println!("Hello from Rust!");
}
""")
    
    # Add Ruby files
    with open(repo_dir / "ruby/src/main.rb", "w") as f:
        f.write("""
def main
  puts "Hello from Ruby!"
end

main if __FILE__ == $0
""")
    
    # Add documentation
    for lang in ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "ruby"]:
        with open(repo_dir / lang / "docs" / "README.md", "w") as f:
            f.write(f"""# {lang.title()} Project

## Overview

This is a sample {lang.title()} project.

## Code Examples

```{lang}
// Sample code here
```

## Testing

Run tests using the appropriate test runner.
""")
    
    # Add configuration files
    with open(repo_dir / ".gitignore", "w") as f:
        f.write("""
*.pyc
node_modules/
target/
build/
dist/
""")
    
    with open(repo_dir / "README.md", "w") as f:
        f.write("""# Complex Repository

This repository contains sample code in multiple languages.
""")
    
    return repo_dir

def test_repository_integration(complex_repo_fixture, temp_dir):
    """Test full repository integration with all supported languages."""
    output_dir = Path(temp_dir) / "output"
    
    # Extract code from repository
    stats = extract_repository(complex_repo_fixture, output_dir)
    
    # Verify stats structure
    assert isinstance(stats, dict), "Stats should be a dictionary"
    assert "code_blocks" in stats, "Stats missing code_blocks count"
    assert "total_files" in stats, "Stats missing total_files count"
    assert "file_blocks" in stats, "Stats missing file_blocks dictionary"
    assert "errors" in stats, "Stats missing errors list"
    
    # Verify file counts
    assert stats["total_files"] > 0, "No files were processed"
    assert len(stats["file_blocks"]) > 0, "No files were extracted"
    
    # Verify language support
    languages = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby"
    }
    
    for ext, lang in languages.items():
        files = [f for f in stats["file_blocks"].keys() if str(f).endswith(ext)]
        assert len(files) > 0, f"No {lang} files were processed"
    
    # Verify directory structure
    assert output_dir.exists(), "Output directory not created"
    for lang in languages.values():
        assert (output_dir / "blocks" / "code" / lang).exists(), f"{lang} output directory not created"
    assert (output_dir / "blocks" / "sections").exists(), "Sections output directory not created"
    
    # Verify block extraction
    for file_path, blocks in stats["file_blocks"].items():
        assert len(blocks) > 0, f"No blocks extracted from {file_path}"
        for block in blocks:
            assert "block_type" in block, f"Block missing block_type: {block}"
            assert "name" in block, f"Block missing name: {block}"
            assert "content" in block, f"Block missing content: {block}"
            assert "output_file" in block, f"Block missing output_file: {block}"
            
            # Verify block file exists
            assert Path(block["output_file"]).exists(), f"Block file does not exist: {block['output_file']}"

def test_repository_parallel_processing(complex_repo_fixture, temp_dir):
    """Test parallel processing of repository files."""
    output_dir = Path(temp_dir) / "output"
    
    def process_file(file_path):
        """Process a single file and return its blocks."""
        if not _is_code_file(file_path):
            return []
        
        with open(file_path) as f:
            content = f.read()
        
        stats = initialize_stats_dict(source=file_path, output_dir=output_dir)
        
        if file_path.endswith(".py"):
            return _extract_python_blocks(file_path, content, output_dir, stats)
        elif file_path.endswith((".js", ".ts")):
            return _extract_js_ts_blocks(file_path, content, output_dir, stats)
        elif file_path.endswith(".md"):
            return _extract_markdown_blocks(file_path, content, output_dir, stats)
        return []
    
    # Get all files in repository
    files = []
    for root, _, filenames in os.walk(complex_repo_fixture):
        for filename in filenames:
            files.append(Path(root) / filename)
    
    # Process files in parallel
    blocks = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, str(f)): f for f in files}
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_blocks = future.result()
                blocks.extend(file_blocks)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    assert len(blocks) > 0, "No blocks were extracted in parallel"

def test_repository_error_handling(temp_dir):
    """Test repository extraction error handling."""
    repo_dir = Path(temp_dir) / "error_repo"
    repo_dir.mkdir(exist_ok=True)
    
    # Create files with syntax errors
    files = {
        "error1.py": """
def broken_function()
    print("Missing colon")
""",
        "error2.js": """
function broken_function( {
    console.log("Missing parenthesis");
}
""",
        "valid.py": """
def valid_function():
    print("This is valid")
"""
    }
    
    for filename, content in files.items():
        with open(repo_dir / filename, "w") as f:
            f.write(content)
    
    output_dir = Path(temp_dir) / "output"
    
    # Extract code from repository
    stats = extract_repository(repo_dir, output_dir)
    
    # Verify error handling
    assert "errors" in stats, "Stats should track errors"
    assert len(stats["errors"]) > 0, "Errors should be recorded"
    assert stats["total_files"] == 3, "All files should be counted"
    assert len(stats["file_blocks"]) > 0, "Valid files should be processed"
    
    # Verify valid file was processed
    valid_files = [f for f in stats["file_blocks"].keys() if str(f).endswith("valid.py")]
    assert len(valid_files) == 1, "Valid file should be processed"
    
    # Verify error files were tracked
    error_files = [f for f in stats["errors"] if str(f).endswith((".py", ".js"))]
    assert len(error_files) == 2, "Error files should be tracked"

def test_repository_cleanup(complex_repo_fixture, temp_dir):
    """Test cleanup of temporary files during repository extraction."""
    output_dir = Path(temp_dir) / "output"
    temp_pattern = "*.tmp"
    
    # Extract code from repository
    extract_repository(complex_repo_fixture, output_dir)
    
    # Check for temporary files
    temp_files = []
    for root, _, files in os.walk(output_dir):
        temp_files.extend(Path(root).glob(temp_pattern))
    
    assert len(temp_files) == 0, "Temporary files should be cleaned up"
    
    # Verify only expected directories exist
    allowed_dirs = {"blocks", "code", "sections"}
    root_dirs = {d.name for d in output_dir.iterdir() if d.is_dir()}
    assert root_dirs.issubset(allowed_dirs), "Unexpected directories found"

if __name__ == "__main__":
    pytest.main([__file__]) 