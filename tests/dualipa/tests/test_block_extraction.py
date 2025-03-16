"""
Test for code_extractor.py block extraction functionality.

This tests the block extraction capabilities of the code_extractor module
to ensure it correctly extracts Python functions/classes, Markdown sections,
JavaScript/TypeScript blocks, and generic code blocks.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import pytest

# Remove path manipulation
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import the module to test
from agent_tools.dualipa.code_extractor import (
    extract_repository,
    _extract_python_blocks,
    _extract_markdown_blocks,
    _extract_js_ts_blocks,
    _extract_generic_blocks
)

# Test fixtures for various file types
@pytest.fixture
def python_content():
    """Sample Python content."""
    return '''"""Sample module for testing."""

def hello_world():
    """Say hello to the world."""
    return "Hello, World!"

class TestClass:
    """A test class."""
    
    def __init__(self, name):
        """Initialize with a name."""
        self.name = name
    
    def greet(self):
        """Return a greeting."""
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    print(hello_world())
    test = TestClass("Tester")
    print(test.greet())
'''

@pytest.fixture
def markdown_content():
    """Sample Markdown content."""
    return '''# Test Document

This is a test document for block extraction.

## Section 1

Content for section 1.

## Section 2

Content for section 2.

## Section 3

Content for section 3.
'''

@pytest.fixture
def javascript_content():
    """Sample JavaScript content."""
    return '''// Sample JavaScript file

function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }
    
    sayHello() {
        return `Hello, my name is ${this.name}!`;
    }
}

const add = (a, b) => {
    return a + b;
};

console.log(greet("World"));
'''

def test_python_block_extraction(tmp_path, python_content):
    """Test Python block extraction."""
    # Set up
    file_path = tmp_path / "test.py"
    with open(file_path, "w") as f:
        f.write(python_content)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    stats = {"code_blocks": 0, "errors": []}
    
    # Execute
    block_count = _extract_python_blocks(file_path, python_content, output_dir, stats)
    
    # Verify
    assert block_count > 0, "Should extract at least one block"
    assert stats["code_blocks"] > 0, "Should update the statistics"
    
    # Check that blocks directory exists
    blocks_dir = output_dir / "blocks" / "code" / "python"
    assert blocks_dir.exists(), "Should create the blocks directory"
    
    # Check that block files exist
    block_files = list(blocks_dir.glob("*"))
    assert len(block_files) > 0, "Should create block files"
    
    # Check that blocks have the correct content
    has_hello_world = False
    has_test_class = False
    
    for block_file in block_files:
        with open(block_file, "r") as f:
            content = f.read()
            if "hello_world" in content and "Say hello to the world" in content:
                has_hello_world = True
            if "TestClass" in content and "A test class" in content:
                has_test_class = True
    
    assert has_hello_world, "Should extract the hello_world function"
    assert has_test_class, "Should extract the TestClass class"

def test_markdown_block_extraction(tmp_path, markdown_content):
    """Test Markdown block extraction."""
    # Set up
    file_path = tmp_path / "test.md"
    with open(file_path, "w") as f:
        f.write(markdown_content)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    stats = {"doc_blocks": 0, "errors": []}
    
    # Execute
    block_count = _extract_markdown_blocks(file_path, markdown_content, output_dir, stats)
    
    # Verify
    assert block_count > 0, "Should extract at least one block"
    assert stats["doc_blocks"] > 0, "Should update the statistics"
    
    # Check that blocks directory exists
    blocks_dir = output_dir / "blocks" / "docs" / "markdown"
    assert blocks_dir.exists(), "Should create the blocks directory"
    
    # Check that block files exist
    block_files = list(blocks_dir.glob("*"))
    assert len(block_files) > 0, "Should create block files"
    
    # Verify at least 3 sections were extracted (for the sample content)
    assert len(block_files) >= 3, "Should extract all sections"
    
    # Check that blocks have the correct content
    section_count = 0
    for block_file in block_files:
        with open(block_file, "r") as f:
            content = f.read()
            if "Section" in content:
                section_count += 1
    
    assert section_count >= 3, "Should extract all sections with their content"

def test_javascript_block_extraction(tmp_path, javascript_content):
    """Test JavaScript block extraction."""
    # Set up
    file_path = tmp_path / "test.js"
    with open(file_path, "w") as f:
        f.write(javascript_content)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    stats = {"code_blocks": 0, "errors": []}
    
    # Execute
    block_count = _extract_js_ts_blocks(file_path, javascript_content, output_dir, stats, "javascript")
    
    # Verify
    assert block_count > 0, "Should extract at least one block"
    assert stats["code_blocks"] > 0, "Should update the statistics"
    
    # Check that blocks directory exists
    blocks_dir = output_dir / "blocks" / "code" / "javascript"
    assert blocks_dir.exists(), "Should create the blocks directory"
    
    # Check that block files exist
    block_files = list(blocks_dir.glob("*"))
    assert len(block_files) > 0, "Should create block files"

def test_generic_block_extraction(tmp_path):
    """Test generic block extraction."""
    # Sample content with blocks separated by double newlines
    content = """Block 1
This is the first block.

Block 2
This is the second block.

Block 3
This is the third block."""
    
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write(content)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    stats = {"code_blocks": 0, "errors": []}
    
    # Execute
    block_count = _extract_generic_blocks(file_path, content, output_dir, stats, "text")
    
    # Verify
    assert block_count > 0, "Should extract at least one block"
    assert stats["code_blocks"] > 0, "Should update the statistics"
    
    # Check that blocks directory exists
    blocks_dir = output_dir / "blocks" / "code" / "text"
    assert blocks_dir.exists(), "Should create the blocks directory"
    
    # Check that block files exist
    block_files = list(blocks_dir.glob("*"))
    assert len(block_files) > 0, "Should create block files"
    
    # Verify that the correct number of blocks were extracted
    assert len(block_files) == 3, "Should extract all 3 blocks"

def test_full_extraction_process(tmp_path):
    """Test the full extraction process with blocks."""
    # Create a small repository structure
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    # Create a Python file
    with open(repo_dir / "main.py", "w") as f:
        f.write('''"""Main module."""

def main():
    """Main function."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')
    
    # Create a Markdown file
    with open(repo_dir / "README.md", "w") as f:
        f.write('''# Test Repository

This is a test repository.

## Usage

Run `main.py` to see the output.
''')
    
    # Create a JavaScript file
    with open(repo_dir / "script.js", "w") as f:
        f.write('''// Script file

function hello() {
    console.log("Hello from JavaScript!");
}

hello();
''')
    
    # Execute extraction
    output_dir = tmp_path / "output"
    stats = extract_repository(
        str(repo_dir),
        str(output_dir),
        extract_documentation=True,
        extract_code=True,
        extract_blocks=True
    )
    
    # Verify
    assert stats["total_files"] > 0, "Should process files"
    assert stats["code_files"] > 0, "Should extract code files"
    assert stats["documentation_files"] > 0, "Should extract documentation files"
    assert stats["code_blocks"] > 0, "Should extract code blocks"
    assert stats["doc_blocks"] > 0, "Should extract documentation blocks"
    
    # Check that directories exist
    assert (output_dir / "code").exists(), "Should create code directory"
    assert (output_dir / "docs").exists(), "Should create docs directory"
    assert (output_dir / "blocks" / "code").exists(), "Should create code blocks directory"
    assert (output_dir / "blocks" / "docs").exists(), "Should create docs blocks directory"
    
    # Check for specific files/blocks
    python_files = list((output_dir / "code").glob("**/main.py*"))
    assert len(python_files) > 0, "Should extract the Python file"
    
    md_files = list((output_dir / "docs").glob("**/README.md*"))
    assert len(md_files) > 0, "Should extract the Markdown file"
    
    js_files = list((output_dir / "code").glob("**/script.js*"))
    assert len(js_files) > 0, "Should extract the JavaScript file"
    
    python_blocks = list((output_dir / "blocks" / "code").glob("**/python/*"))
    assert len(python_blocks) > 0, "Should extract Python blocks"
    
    md_blocks = list((output_dir / "blocks" / "docs").glob("**/markdown/*"))
    assert len(md_blocks) > 0, "Should extract Markdown blocks"

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 