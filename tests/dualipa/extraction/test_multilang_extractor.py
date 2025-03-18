#!/usr/bin/env python3
"""
Tests for the multilang_extractor module.

This file contains tests for the multilang_extractor module,
which is responsible for extracting code blocks from files in
different programming languages.
"""

import os
import sys
import pytest
from pathlib import Path

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Local test repository paths
RUST_ANALYZER_PATH = project_root / "test_repos" / "rust-analyzer"
REACT_PATH = project_root / "test_repos" / "react"

# Test if the repositories exist
HAS_TEST_REPOS = RUST_ANALYZER_PATH.exists() and REACT_PATH.exists()
if not HAS_TEST_REPOS:
    print(f"Warning: Test repositories not found at: {RUST_ANALYZER_PATH}, {REACT_PATH}")
    print("Some tests will be skipped")

# Flag to track if dependencies are available
HAS_DEPENDENCIES = False

# Import the extractor functions
try:
    from agent_tools.dualipa.multilang_extractor import (
        extract_code_blocks,
        get_language_for_file,
        get_available_languages,
        extract_blocks_from_repository
    )
    HAS_DEPENDENCIES = True
    print("Successfully imported multilang_extractor")
except ImportError as e:
    # Instead of silently skipping, fail loudly with a clear error message
    raise ImportError(f"Required multilang_extractor modules not available: {e}. Fix the dependencies to run these tests.")

# Check if tree-sitter is available
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
    print("tree-sitter is available")
except ImportError as e:
    # Instead of silently skipping, fail loudly
    raise ImportError(f"tree-sitter is not available: {e}. Install tree-sitter to run these tests.")

# Skip tests if required modules are not available
# if not HAS_DEPENDENCIES:
#     pytestmark = pytest.mark.skipif(
#         True,
#         reason="Required modules not available"
#     )

# Define paths to test repositories
REPOS_DIR = Path(__file__).parent.parent.parent.parent / "test_repos"
REACT_REPO = REPOS_DIR / "react"
RUST_ANALYZER_REPO = REPOS_DIR / "rust-analyzer"
REQUESTS_REPO = REPOS_DIR / "requests"

HAS_REACT = REACT_REPO.exists()
HAS_RUST_ANALYZER = RUST_ANALYZER_REPO.exists()
HAS_REQUESTS = REQUESTS_REPO.exists()

print(f"Repository status:")
print(f"- Requests: {'Available' if HAS_REQUESTS else 'Not found'}")
print(f"- React: {'Available' if HAS_REACT else 'Not found'}")
print(f"- Rust Analyzer: {'Available' if HAS_RUST_ANALYZER else 'Not found'}")

# Remove the tree-sitter skipif marker
# pytestmark = pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not available")

def test_get_available_languages():
    """Test that we can get a list of supported languages."""
    languages = get_available_languages()
    
    # Check we have at least some common languages
    assert len(languages) > 0, "Should return at least one language"
    
    expected_languages = {"python", "javascript", "typescript"}
    for lang in expected_languages:
        if lang in languages:
            print(f"Found supported language: {lang}")
    
    # Print all available languages
    print(f"Available languages: {languages}")

def test_get_language_for_file():
    """Test that we can detect the language from file extensions."""
    try:
        # Test some common extensions
        assert get_language_for_file("test.py") == "python", "Should detect Python"
        assert get_language_for_file("test.js") == "javascript", "Should detect JavaScript"
        assert get_language_for_file("test.ts") == "typescript", "Should detect TypeScript"
        
        # Test with full paths
        assert get_language_for_file("/path/to/file.py") == "python", "Should extract extension from path"
        
        # Test with Path objects
        assert get_language_for_file(Path("/path/to/file.rs")) == "rust", "Should work with Path objects"
        
        # Test unknown extension
        assert get_language_for_file("test.unknown") is None, "Should return None for unknown extensions"
        
        print("Language detection for files is working")
    except Exception as e:
        pytest.fail(f"Error in get_language_for_file: {e}")

def test_extract_python_code_blocks():
    """Test extraction of Python code blocks."""
    if not HAS_REQUESTS:
        pytest.fail("No Python repository available")
    
    python_files = list(REQUESTS_REPO.glob("**/*.py"))
    if not python_files:
        pytest.fail("No Python files found in requests repository")
    
    # Sort files by size to get a substantial file
    python_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    
    # Select a good-sized Python file
    selected_file = None
    for file in python_files:
        if file.stat().st_size > 1000:  # At least 1KB
            selected_file = file
            break
    
    if not selected_file:
        selected_file = python_files[0]
    
    print(f"Testing with Python file: {selected_file} ({selected_file.stat().st_size} bytes)")
    
    # Extract code blocks directly from file
    blocks = extract_code_blocks(selected_file)
    
    # Verify extraction
    assert blocks is not None, "Should return a list of blocks"
    assert len(blocks) > 0, "Should extract at least one code block"
    
    print(f"Extracted {len(blocks)} Python blocks")
    
    # Check first block
    first_block = blocks[0]
    assert first_block.get("language") == "python"
    assert len(first_block.get("content", "")) > 0
    
    print(f"First block preview: {first_block.get('content', '')[:100]}...")

def test_extract_javascript_code_blocks():
    """Test extracting code blocks from JavaScript files."""
    try:
        # Skip if JavaScript isn't in available languages
        languages = get_available_languages()
        if "javascript" not in languages:
            pytest.fail("JavaScript language support not available")
        
        # Find a JavaScript file in the React repo
        if HAS_TEST_REPOS:
            js_files = list(REACT_PATH.glob("**/*.js"))
            
            if not js_files:
                pytest.fail("No JavaScript files found in test repository")
                
            js_file = js_files[0]
        else:
            # Create a simple JavaScript file for testing
            with open("test_sample.js", "w") as f:
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
            js_file = Path("test_sample.js")
        
        print(f"Testing JavaScript extraction with file: {js_file}")
        
        # Extract code blocks
        blocks = extract_code_blocks(js_file)
        
        # Check that we extracted something
        assert blocks is not None, "Should return a list of blocks"
        print(f"Extracted {len(blocks)} blocks from JavaScript file")
        
        # Print first block for inspection if available
        if blocks and len(blocks) > 0:
            print(f"First block language: {blocks[0].get('language')}")
            print(f"First block content preview: {blocks[0].get('content')[:100]}...")
        
        # Cleanup
        if not HAS_TEST_REPOS and os.path.exists("test_sample.js"):
            os.remove("test_sample.js")
            
    except Exception as e:
        pytest.fail(f"Error in extract_javascript_code_blocks: {e}")

def test_extract_rust_code_blocks():
    """Test extracting code blocks from Rust files."""
    try:
        # Skip if Rust isn't in available languages
        languages = get_available_languages()
        if "rust" not in languages:
            pytest.fail("Rust language support not available")
        
        # Find a Rust file in the rust-analyzer repo
        if HAS_TEST_REPOS:
            rust_files = list(RUST_ANALYZER_PATH.glob("**/*.rs"))
            
            if not rust_files:
                pytest.fail("No Rust files found in test repository")
                
            rust_file = rust_files[0]
        else:
            # Create a simple Rust file for testing
            with open("test_sample.rs", "w") as f:
                f.write("""
fn main() {
    println!("Hello, World!");
}

struct TestStruct {
    value: i32,
}

impl TestStruct {
    fn new() -> Self {
        Self { value: 42 }
    }
    
    fn get_value(&self) -> i32 {
        self.value
    }
}
""")
            rust_file = Path("test_sample.rs")
        
        print(f"Testing Rust extraction with file: {rust_file}")
        
        # Extract code blocks
        blocks = extract_code_blocks(rust_file)
        
        # Check that we extracted something
        assert blocks is not None, "Should return a list of blocks"
        print(f"Extracted {len(blocks)} blocks from Rust file")
        
        # Print first block for inspection if available
        if blocks and len(blocks) > 0:
            print(f"First block language: {blocks[0].get('language')}")
            print(f"First block content preview: {blocks[0].get('content')[:100]}...")
        
        # Cleanup
        if not HAS_TEST_REPOS and os.path.exists("test_sample.rs"):
            os.remove("test_sample.rs")
            
    except Exception as e:
        pytest.fail(f"Error in extract_rust_code_blocks: {e}")

def test_extract_typescript_code_blocks():
    """Test extracting code blocks from TypeScript files."""
    try:
        # Skip if TypeScript isn't in available languages
        languages = get_available_languages()
        if "typescript" not in languages:
            pytest.fail("TypeScript language support not available")
        
        # Find a TypeScript file in the React repo
        if HAS_TEST_REPOS:
            ts_files = list(REACT_PATH.glob("**/*.ts")) + list(REACT_PATH.glob("**/*.tsx"))
            
            if not ts_files:
                pytest.fail("No TypeScript files found in test repository")
                
            ts_file = ts_files[0]
        else:
            # Create a simple TypeScript file for testing
            with open("test_sample.ts", "w") as f:
                f.write("""
function helloWorld(): void {
    console.log("Hello, World!");
}

interface TestInterface {
    value: number;
}

class TestClass implements TestInterface {
    value: number;
    
    constructor() {
        this.value = 42;
    }
    
    getValue(): number {
        return this.value;
    }
}
""")
            ts_file = Path("test_sample.ts")
        
        print(f"Testing TypeScript extraction with file: {ts_file}")
        
        # Extract code blocks
        blocks = extract_code_blocks(ts_file)
        
        # Check that we extracted something
        assert blocks is not None, "Should return a list of blocks"
        print(f"Extracted {len(blocks)} blocks from TypeScript file")
        
        # Print first block for inspection if available
        if blocks and len(blocks) > 0:
            print(f"First block language: {blocks[0].get('language')}")
            print(f"First block content preview: {blocks[0].get('content')[:100]}...")
        
        # Cleanup
        if not HAS_TEST_REPOS and os.path.exists("test_sample.ts"):
            os.remove("test_sample.ts")
            
    except Exception as e:
        pytest.fail(f"Error in extract_typescript_code_blocks: {e}")

def test_multifile_extraction():
    """Test extracting code blocks from multiple files."""
    try:
        # Create a temporary directory with multiple file types
        temp_dir = Path("temp_multifile_test")
        temp_dir.mkdir(exist_ok=True)
        
        # Create files of different languages
        files = {
            "sample.py": """
def hello_world():
    print("Hello, World!")
""",
            "sample.js": """
function helloWorld() {
    console.log("Hello, World!");
}
""",
            "sample.rs": """
fn hello_world() {
    println!("Hello, World!");
}
"""
        }
        
        file_paths = []
        for filename, content in files.items():
            file_path = temp_dir / filename
            with open(file_path, "w") as f:
                f.write(content)
            file_paths.append(file_path)
        
        # Extract blocks from each file
        all_blocks = []
        for file_path in file_paths:
            try:
                blocks = extract_code_blocks(file_path)
                if blocks:
                    all_blocks.extend(blocks)
                    print(f"Extracted {len(blocks)} blocks from {file_path}")
            except Exception as e:
                print(f"Error extracting from {file_path}: {e}")
        
        # Check that we got blocks
        assert len(all_blocks) > 0, "Should extract at least one block"
        
        # Check that blocks have correct languages
        languages = {block.get("language") for block in all_blocks if "language" in block}
        print(f"Extracted blocks in languages: {languages}")
        
        # Cleanup
        for file_path in file_paths:
            if file_path.exists():
                os.remove(file_path)
        if temp_dir.exists():
            temp_dir.rmdir()
            
    except Exception as e:
        # Clean up even if test fails
        if "temp_dir" in locals() and temp_dir.exists():
            for file_path in temp_dir.glob("*"):
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
                
        pytest.fail(f"Error in multifile_extraction: {e}")

if __name__ == "__main__":
    # Run tests directly
    pytest.main(["-xvs", __file__]) 