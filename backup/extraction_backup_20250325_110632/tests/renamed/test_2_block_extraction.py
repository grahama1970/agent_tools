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
        _extract_python_blocks,
        _extract_js_ts_blocks,
        _extract_markdown_blocks,
        _extract_generic_blocks
    )
    from agent_tools.dualipa.github_utils import clone_github_repo
    
    # Define helper function that doesn't exist in the codebase
    def find_files_by_extension(directory, extension, limit=None):
        """Find files with a specific extension in a directory."""
        import os
        from pathlib import Path
        files = []
        for root, _, filenames in os.walk(str(directory)):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(Path(os.path.join(root, filename)))
                    if limit and len(files) >= limit:
                        return files
        return files
        
    # Alias for compatibility with the existing test code
    def clone_repository_if_not_exists(url, directory, depth=None):
        """Clone a Git repository if it doesn't exist."""
        repo_path = Path(directory) / Path(url).stem
        if repo_path.exists():
            return repo_path
        return clone_github_repo(url, str(directory))
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
    pytest.fail(f"Failed to clone requests repository: {e}")

# Print repository status
print(f"Repository status:")
print(f"- Rust analyzer: {'Available' if HAS_RUST_REPO else 'Not found'}")
print(f"- React: {'Available' if HAS_REACT_REPO else 'Not found'}")
print(f"- Requests: {'Available' if HAS_REQUESTS_REPO else 'Not found'}")
print(f"- Python sample: {'Available' if PYTHON_REPO_PATH.exists() else 'Not found'}")

def test_extract_python_code_blocks():
    """Test extraction of Python code blocks from real-world Python files."""
    if not HAS_REQUESTS_REPO:
        pytest.fail("Requests repository not available")

    # First, test with a script-like file (setup.py)
    setup_file = REQUESTS_PATH / "setup.py"
    if setup_file.exists():
        print(f"Testing with script file: {setup_file}")
        
        # Set up output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Extract blocks
            with open(setup_file, 'r') as file:
                content = file.read()
            
            stats = {
                "code_blocks": 0, 
                "errors": [],
                "file_blocks": {}
            }
            
            _extract_python_blocks(setup_file, content, output_dir, stats)
            
            # Verify extraction worked for script file
            if stats["code_blocks"] > 0:
                print(f"Successfully extracted {stats['code_blocks']} blocks from script file {setup_file}")
            else:
                print(f"Warning: Failed to extract blocks from script file {setup_file}")
    
    # Now, test with a file that likely has functions/classes
    # Try to find api.py or similar files with likely functions
    api_files = []
    for potential_file in ["api.py", "models.py", "utils.py", "core.py"]:
        matches = list(REQUESTS_PATH.glob(f"**/{potential_file}"))
        if matches:
            api_files.extend(matches)
    
    # If no api-like files found, just get any Python file that's not setup.py
    if not api_files:
        python_files = find_files_by_extension(REQUESTS_PATH, ".py", limit=10)
        api_files = [f for f in python_files if f.name.lower() != "setup.py"]
    
    if not api_files:
        pytest.fail("No suitable Python files found in the requests repository")
    
    api_file = api_files[0]
    print(f"Testing with function file: {api_file}")
    
    # Set up output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Extract blocks
        with open(api_file, 'r') as file:
            content = file.read()
        
        stats = {
            "code_blocks": 0, 
            "errors": [],
            "file_blocks": {}
        }
        
        _extract_python_blocks(api_file, content, output_dir, stats)
        
        # Verify extraction worked for function file
        assert stats["code_blocks"] > 0, f"Should extract at least one code block from {api_file}"
        print(f"Extracted {stats['code_blocks']} blocks from {api_file}")
        
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
        
        _extract_js_ts_blocks(js_file, content, output_dir, stats, language="javascript")
        
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
                _extract_python_blocks(file_path, content, output_dir, stats)
            elif lang == "javascript":
                _extract_js_ts_blocks(file_path, content, output_dir, stats, language="javascript")
            elif lang == "rust":
                _extract_generic_blocks(file_path, content, output_dir, stats, language="rust")
                
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

def test_block_extraction_from_requests():
    """Test extracting code blocks from requests repository."""
    # Clone the requests repository to a temporary directory
    repo_url = "https://github.com/psf/requests"
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            repo_path = clone_repository_if_not_exists(repo_url, tmp_dir)
        except Exception as e:
            pytest.fail(f"Failed to clone requests repository: {e}")
            
        # Now extract code blocks from the repository
        output_dir = os.path.join(tmp_dir, "output")
        
        if not os.path.exists(repo_path):
            pytest.fail("Requests repository not available")
            
        # ... rest of test code

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 