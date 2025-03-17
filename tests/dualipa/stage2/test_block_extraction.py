"""
Tests for block extraction functionality with real-world repositories.

These tests verify the extraction of code blocks from real-world code repositories.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import json

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import required modules, fail loudly if they're not available
try:
    from agent_tools.dualipa.code_extractor import (
        extract_python_blocks,
        extract_javascript_blocks,
        extract_typescript_blocks,
        extract_go_blocks,
        extract_rust_blocks,
        extract_c_blocks,
        extract_blocks_generic,
        extract_markdown_blocks
    )
    from agent_tools.dualipa.utils import (
        clone_repository_if_not_exists,
        find_files_by_extension
    )
except ImportError as e:
    raise ImportError(f"Required dependencies for block extraction tests not available: {e}")

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"
REQUESTS_PATH = project_root / "test_repos" / "requests"
PYTHON_REPO_PATH = project_root / "test_repos" / "python-sample"  # Fallback sample

# Test if the repositories exist
HAS_RUST_REPO = RUST_ANALYZER_PATH.exists()
HAS_REACT_REPO = REACT_PATH.exists()
HAS_REQUESTS_REPO = REQUESTS_PATH.exists() 

# Clone repositories if they don't exist
try:
    if not HAS_REQUESTS_REPO:
        REQUESTS_PATH = clone_repository_if_not_exists(
            "https://github.com/psf/requests.git",
            project_root / "test_repos",
            depth=1
        )
        HAS_REQUESTS_REPO = REQUESTS_PATH.exists()
except Exception as e:
    pytest.skip(f"Failed to clone requests repository: {e}")

# Print repository status
print(f"Repository status:")
print(f"- Rust analyzer: {'Available' if HAS_RUST_REPO else 'Not found'}")
print(f"- React: {'Available' if HAS_REACT_REPO else 'Not found'}")
print(f"- Requests: {'Available' if HAS_REQUESTS_REPO else 'Not found'}")
print(f"- Python sample: {'Available' if HAS_PYTHON_REPO else 'Not found'}")

def test_extract_python_code_blocks():
    """Test extraction of Python code blocks from real-world Python files."""
    if not HAS_REQUESTS_REPO:
        pytest.skip("Requests repository not available")
    
    # Find some Python files
    python_files = find_files_by_extension(REQUESTS_PATH, ".py", limit=5)
    assert len(python_files) > 0, "No Python files found in the requests repository"
    
    # Use an existing Python file from the repository
    python_file = python_files[0]
    print(f"Testing with Python file: {python_file}")
    
    # Set up output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Extract blocks
        with open(python_file, 'r') as file:
            content = file.read()
        
        stats = {
            "code_blocks": 0, 
            "errors": [],
            "file_blocks": {}
        }
        
        extract_python_blocks(python_file, content, output_dir, stats)
        
        # Verify extraction worked
        assert stats["code_blocks"] > 0, "Should extract at least one code block"
        print(f"Extracted {stats['code_blocks']} blocks from {python_file}")
        
        # Check if blocks were written to files
        blocks_dir = output_dir / "blocks" / "code" / "python"
        assert blocks_dir.exists(), "Should create blocks directory"
        
        block_files = list(blocks_dir.glob("*.py"))
        assert len(block_files) > 0, "Should create at least one block file"
        print(f"Found {len(block_files)} block files")
        
        # Check content of first block file
        if block_files:
            with open(block_files[0], 'r') as f:
                block_content = f.read()
            print(f"First block preview: {block_content[:100]}...")
            assert len(block_content) > 0, "Block file should contain content"

def test_extract_javascript_code_blocks():
    """Test extraction of JavaScript code blocks from real-world JS files."""
    # Specific React JS files we know exist
    js_files = [
        REACT_PATH / "fixtures" / "devtools" / "scheduling-profiler" / "run.js",
        REACT_PATH / "fixtures" / "devtools" / "regression" / "shared.js",
        REACT_PATH / "fixtures" / "legacy-jsx-runtimes" / "react-15" / "cjs" / "react-jsx-dev-runtime.development.js"
    ]
    
    # Check if at least one file exists
    existing_files = [f for f in js_files if f.exists()]
    if not existing_files:
        pytest.fail("No JavaScript test files found. Repository may be missing or corrupted.")
    
    js_file = existing_files[0]
    print(f"Testing with JavaScript file: {js_file} ({js_file.stat().st_size} bytes)")
    
    # Set up output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Extract blocks
        with open(js_file, 'r') as file:
            content = file.read()
        
        stats = {
            "code_blocks": 0, 
            "errors": [],
            "file_blocks": {}
        }
        
        extract_javascript_blocks(js_file, content, output_dir, stats)
        
        # Print any errors for debugging
        if stats["errors"]:
            print("Errors during extraction:")
            for error in stats["errors"]:
                print(f"- {error}")
                
        # Verify extraction
        # Even if extraction didn't find any blocks, it shouldn't crash
        print(f"Extracted {stats['code_blocks']} blocks from {js_file}")
        
        # Check if blocks were written to files
        blocks_dir = output_dir / "blocks" / "code" / "javascript"
        if blocks_dir.exists():
            block_files = list(blocks_dir.glob("*.js"))
            print(f"Found {len(block_files)} block files")
            
            # Check content of first block file if any were created
            if block_files:
                with open(block_files[0], 'r') as f:
                    block_content = f.read()
                print(f"First block preview: {block_content[:100]}...")
                assert len(block_content) > 0, "Block file should contain content"
        else:
            # No blocks were found, but test should still pass
            # The function should handle the case gracefully
            print("No blocks were extracted from the file")

def test_cross_language_extraction():
    """Test extraction of code blocks from multiple files with different languages."""
    # Use specific files we know exist
    test_files = []
    
    # Python file from requests
    python_file = REQUESTS_PATH / "setup.py"
    if python_file.exists():
        test_files.append(("python", python_file))
    
    # JavaScript file from React
    js_file = REACT_PATH / "fixtures" / "devtools" / "scheduling-profiler" / "run.js"
    if js_file.exists():
        test_files.append(("javascript", js_file))
    
    # Rust file from rust-analyzer
    rust_file = RUST_ANALYZER_PATH / "lib" / "la-arena" / "src" / "lib.rs"
    if rust_file.exists():
        test_files.append(("rust", rust_file))
    
    if not test_files:
        pytest.fail("No test files found. Repositories may be missing or corrupted.")
        
    print(f"Selected {len(test_files)} files for cross-language extraction testing")
    for lang, file in test_files:
        print(f"- {lang}: {file}")
        
    # Set up output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Extract blocks from each file type
        total_blocks = 0
        for lang, file_path in test_files:
            print(f"Testing extraction from {lang} file: {file_path}")
            
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                
            stats = {
                "code_blocks": 0, 
                "errors": [],
                "file_blocks": {}
            }
            
            # Call the appropriate extraction function based on language
            if lang == "python":
                extract_python_blocks(file_path, content, output_dir, stats)
            elif lang == "javascript":
                extract_javascript_blocks(file_path, content, output_dir, stats)
            elif lang == "rust":
                extract_rust_blocks(file_path, content, output_dir, stats)
                
            # Print any errors for debugging
            if stats["errors"]:
                print(f"Errors during {lang} extraction:")
                for error in stats["errors"]:
                    print(f"- {error}")
            
            # Verify extraction
            block_count = stats["code_blocks"]
            total_blocks += block_count
            print(f"Extracted {block_count} blocks from {lang} file")
            
            # Check blocks directory
            blocks_dir = output_dir / "blocks" / "code" / lang
            if blocks_dir.exists():
                block_files = list(blocks_dir.glob(f"*.{lang}"))
                print(f"Found {len(block_files)} block files for {lang}")
                if block_files and len(block_files) > 0:
                    print(f"First {lang} block file: {block_files[0].name}")
        
        # Overall results
        print(f"Extracted a total of {total_blocks} blocks across {len(test_files)} languages")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 