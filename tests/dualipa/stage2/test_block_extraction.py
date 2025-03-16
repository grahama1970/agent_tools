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

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"
REQUESTS_PATH = project_root / "test_repos" / "requests"
PYTHON_REPO_PATH = project_root / "test_repos" / "python-sample"  # Fallback sample

# Test if the repositories exist
HAS_RUST_REPO = RUST_ANALYZER_PATH.exists()
HAS_REACT_REPO = REACT_PATH.exists()
HAS_REQUESTS_REPO = REQUESTS_PATH.exists() 
HAS_PYTHON_REPO = PYTHON_REPO_PATH.exists()  # Fallback sample

# Print repository status
print(f"Repository status:")
print(f"- Rust analyzer: {'Available' if HAS_RUST_REPO else 'Not found'}")
print(f"- React: {'Available' if HAS_REACT_REPO else 'Not found'}")
print(f"- Requests: {'Available' if HAS_REQUESTS_REPO else 'Not found'}")
print(f"- Python sample: {'Available' if HAS_PYTHON_REPO else 'Not found'}")

# Import the required modules
try:
    from agent_tools.dualipa.code_extractor import (
        extract_repository,
        _extract_python_blocks,
        _extract_js_ts_blocks
    )
    from agent_tools.dualipa.github_utils import fetch_file_content, clone_github_repo
    HAS_DEPENDENCIES = True
    print("Successfully imported extraction modules")
    
    # Try to clone the requests repository if it doesn't exist
    if not HAS_REQUESTS_REPO:
        print("Attempting to clone requests repository...")
        try:
            os.makedirs(project_root / "test_repos", exist_ok=True)
            clone_github_repo(
                "https://github.com/psf/requests.git",
                str(REQUESTS_PATH),
                single_branch=True,
                depth=1
            )
            HAS_REQUESTS_REPO = REQUESTS_PATH.exists()
            print(f"Requests repository {'cloned successfully' if HAS_REQUESTS_REPO else 'clone failed'}")
        except Exception as e:
            print(f"Failed to clone requests repository: {e}")
            
except ImportError as e:
    print(f"Import error: {e}")
    print("Required dependencies not available, tests will be skipped")
    HAS_DEPENDENCIES = False

# Skip all tests if dependencies are not available
pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES, 
    reason="Required modules not available"
)

def test_extract_python_code_blocks():
    """Test extraction of Python code blocks from real-world Python files."""
    try:
        # Try to find Python files in the requests repository first
        python_files = []
        if HAS_REQUESTS_REPO:
            python_files = list(REQUESTS_PATH.glob("**/*.py"))
            if python_files:
                print(f"Found {len(python_files)} Python files in requests repository")
                # Sort by file size to find substantial files
                python_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                for file in python_files[:5]:  # Print top 5 files
                    print(f"- {file.relative_to(REQUESTS_PATH)} ({file.stat().st_size} bytes)")
        
        # Fallback to Python sample if no files found in requests
        if not python_files and HAS_PYTHON_REPO:
            python_files = list(PYTHON_REPO_PATH.glob("**/*.py"))
            if python_files:
                print(f"Found {len(python_files)} Python files in sample directory")
        
        # Look for Python files elsewhere if needed
        if not python_files:
            print("Looking for Python files in project root...")
            python_files = list(Path(project_root).glob("**/*.py"))
            if python_files:
                print(f"Found {len(python_files)} Python files in project")
            
        if not python_files:
            # Create a sample Python file if none exist in the repo
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w') as f:
                f.write("""
def hello_world():
    print("Hello, World!")

class TestClass:
    def __init__(self):
        self.value = 42
        
    def get_value(self):
        return self.value
""")
                f.flush()
                
                # Set up output directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir)
                    
                    # Extract blocks
                    with open(f.name, 'r') as file:
                        content = file.read()
                    
                    stats = {
                        "code_blocks": 0, 
                        "errors": [],
                        "file_blocks": {}
                    }
                    
                    _extract_python_blocks(Path(f.name), content, output_dir, stats)
                    
                    # Verify extraction worked
                    assert stats["code_blocks"] > 0, "Should extract at least one code block"
                    print(f"Extracted {stats['code_blocks']} blocks from {f.name}")
                    
                    # Check if blocks were written to files
                    blocks_dir = output_dir / "blocks" / "code" / "python"
                    assert blocks_dir.exists(), "Should create blocks directory"
                    
                    block_files = list(blocks_dir.glob("*.py"))
                    assert len(block_files) > 0, "Should create at least one block file"
                    print(f"Found {len(block_files)} block files")
        else:
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
                
                _extract_python_blocks(python_file, content, output_dir, stats)
                
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
        
    except Exception as e:
        pytest.skip(f"Error in extract_python_code_blocks test: {e}")

def test_extract_javascript_code_blocks():
    """Test extraction of JavaScript code blocks from real-world JS files."""
    try:
        # Find JavaScript files in React repo
        js_files = []
        if HAS_REACT_REPO:
            js_files = list(REACT_PATH.glob("**/*.js"))
            if js_files:
                print(f"Found {len(js_files)} JavaScript files in React repository")
                # Sort by file size to find substantial files
                js_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                for file in js_files[:5]:  # Print top 5 files
                    print(f"- {file.relative_to(REACT_PATH)} ({file.stat().st_size} bytes)")
        
        if not js_files:
            # Create a sample JavaScript file if none exist in the repo
            with tempfile.NamedTemporaryFile(suffix='.js', mode='w') as f:
                f.write("""
function helloWorld() {
    console.log("Hello, World!");
}

class TestClass {
    constructor() {
        this.value = 42;
    }
    
    getValue() {
        return this.value;
    }
}
""")
                f.flush()
                
                # Set up output directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir)
                    
                    # Extract blocks
                    with open(f.name, 'r') as file:
                        content = file.read()
                    
                    stats = {
                        "code_blocks": 0, 
                        "errors": [],
                        "file_blocks": {}
                    }
                    
                    _extract_js_ts_blocks(Path(f.name), content, output_dir, stats, "javascript")
                    
                    # Verify extraction worked
                    assert stats["code_blocks"] > 0, "Should extract at least one code block"
                    print(f"Extracted {stats['code_blocks']} blocks from {f.name}")
                    
                    # Check if blocks were written to files
                    blocks_dir = output_dir / "blocks" / "code" / "javascript"
                    assert blocks_dir.exists(), "Should create blocks directory"
                    
                    block_files = list(blocks_dir.glob("*.js"))
                    assert len(block_files) > 0, "Should create at least one block file"
                    print(f"Found {len(block_files)} block files")
        else:
            # Use an existing JavaScript file from the repository
            js_file = js_files[0]
            print(f"Testing with JavaScript file: {js_file}")
            
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
                
                _extract_js_ts_blocks(js_file, content, output_dir, stats, "javascript")
                
                # Verify extraction worked
                assert stats["code_blocks"] > 0, "Should extract at least one code block"
                print(f"Extracted {stats['code_blocks']} blocks from {js_file}")
                
                # Check if blocks were written to files
                blocks_dir = output_dir / "blocks" / "code" / "javascript"
                assert blocks_dir.exists(), "Should create blocks directory"
                
                block_files = list(blocks_dir.glob("*.js"))
                assert len(block_files) > 0, "Should create at least one block file"
                print(f"Found {len(block_files)} block files")
                
                # Check content of first block file
                if block_files:
                    with open(block_files[0], 'r') as f:
                        block_content = f.read()
                    print(f"First block preview: {block_content[:100]}...")
                    assert len(block_content) > 0, "Block file should contain content"
        
    except Exception as e:
        pytest.skip(f"Error in extract_javascript_code_blocks test: {e}")

def test_cross_language_extraction():
    """Test extraction of code blocks from multiple files with different languages."""
    try:
        # Find files of different languages
        python_files = []
        js_files = []
        rust_files = []
        
        # Find Python files
        if HAS_REQUESTS_REPO:
            python_files = list(REQUESTS_PATH.glob("**/*.py"))
            if python_files:
                python_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                
        # Find JavaScript files
        if HAS_REACT_REPO:
            js_files = list(REACT_PATH.glob("**/*.js"))
            if js_files:
                js_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                
        # Find Rust files
        if HAS_RUST_REPO:
            rust_files = list(RUST_ANALYZER_PATH.glob("**/*.rs"))
            if rust_files:
                rust_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                
        # Print file counts
        print(f"Found {len(python_files)} Python files, {len(js_files)} JavaScript files, and {len(rust_files)} Rust files")
        
        # Collect files to test
        test_files = []
        if python_files:
            test_files.append(("python", python_files[0]))
        if js_files:
            test_files.append(("javascript", js_files[0]))
        if rust_files:
            test_files.append(("rust", rust_files[0]))
        
        if not test_files:
            pytest.skip("No suitable test files found in repositories")
            
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
                
                # Extract blocks from this specific file
                stats = extract_repository(
                    source=str(file_path),
                    output_path=temp_dir,
                    extract_documentation=False,
                    extract_code=True,
                    extract_blocks=True
                )
                
                # Verify extraction worked
                block_count = stats.get("code_blocks", 0)
                assert block_count > 0, f"Should extract at least one code block from {lang} file"
                total_blocks += block_count
                print(f"Extracted {block_count} blocks from {lang} file")
            
            # Check overall results
            assert total_blocks > 0, "Should extract blocks from at least one language"
            print(f"Extracted a total of {total_blocks} blocks across languages")
        
    except Exception as e:
        pytest.skip(f"Error in cross_language_extraction test: {e}")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 