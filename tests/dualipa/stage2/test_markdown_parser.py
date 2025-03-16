"""
Test the markdown_parser module.

These tests verify that the markdown parser correctly extracts sections,
code blocks, and other content from markdown files.

Official Documentation References:
- markdown-it-py: https://markdown-it-py.readthedocs.io/
- mistune: https://mistune.readthedocs.io/
- pytest: https://docs.pytest.org/
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Remove path manipulation
# current_dir = Path(__file__).parent
# parent_dir = current_dir.parent
# if str(parent_dir) not in sys.path:
#     sys.path.append(str(parent_dir))

# Import directly from the package
try:
    from agent_tools.dualipa.markdown_parser import (
        extract_sections_from_markdown,
        extract_code_blocks,
        process_markdown_file,
        get_markdown_files,
        MARKDOWN_IT_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Skipping markdown_parser tests: {e}", allow_module_level=True)


def test_extract_sections_from_markdown():
    """Test that sections are correctly extracted from markdown content."""
    # Test markdown content with headers
    markdown_content = """# Title

This is an introduction.

## Section 1

This is section 1 content.

### Subsection 1.1

This is subsection 1.1 content.

## Section 2

This is section 2 content.
"""

    sections = extract_sections_from_markdown(markdown_content)

    # Verify that sections were extracted correctly
    assert len(sections) >= 4  # Title, Section 1, Subsection 1.1, Section 2
    
    # Verify the structure of sections (now using a dictionary format)
    assert "Title" in sections
    assert "This is an introduction" in sections["Title"]
    
    assert "Section 1" in sections
    assert "This is section 1 content" in sections["Section 1"]
    
    assert "Section 2" in sections
    assert "This is section 2 content" in sections["Section 2"]
    
    assert "Subsection 1.1" in sections
    assert "This is subsection 1.1 content" in sections["Subsection 1.1"]


def test_extract_code_blocks():
    """Test that code blocks are correctly extracted from markdown content."""
    # Test markdown content with code blocks
    markdown_content = """# Title

Here's a Python code block:

```python
def hello():
    print("Hello, world!")
```

And a JavaScript code block:

```javascript
function hello() {
    console.log("Hello, world!");
}
```

And a block without language specification:

```
This is a generic code block
```
"""
    
    code_blocks = extract_code_blocks(markdown_content)
    
    # Verify that code blocks were extracted correctly
    assert len(code_blocks) == 3
    
    # Verify the Python code block
    python_block = next((b for b in code_blocks if b.get("language") == "python"), None)
    assert python_block is not None
    assert "def hello():" in python_block["content"]
    assert 'print("Hello, world!")' in python_block["content"]
    
    # Verify the JavaScript code block
    js_block = next((b for b in code_blocks if b.get("language") == "javascript"), None)
    assert js_block is not None
    assert "function hello() {" in js_block["content"]
    assert 'console.log("Hello, world!");' in js_block["content"]
    
    # Verify the generic code block
    generic_block = next((b for b in code_blocks if b.get("language") is None or b.get("language") == ""), None)
    assert generic_block is not None
    assert "This is a generic code block" in generic_block["content"]


def test_process_markdown_file():
    """Test that markdown files are processed correctly."""
    # Create a temporary markdown file
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
        temp_path = temp_file.name
        markdown_content = """# Test Markdown
        
This is a test markdown file.

## Section 1

This is section 1 content.

```python
def test():
    return "Hello"
```

## Section 2

This is section 2 content.
"""
        temp_file.write(markdown_content.encode("utf-8"))
    
    try:
        # Process the markdown file
        result = process_markdown_file(temp_path)
        
        # Verify the result structure
        assert "content" in result
        assert "sections" in result
        assert "code_blocks" in result
        
        # Verify the content
        assert "# Test Markdown" in result["content"]
        
        # Verify the sections
        assert len(result["sections"]) >= 3  # Title, Section 1, Section 2
        
        # Verify the code blocks
        assert len(result["code_blocks"]) == 1
        assert "def test():" in result["code_blocks"][0]["content"]
        assert "return \"Hello\"" in result["code_blocks"][0]["content"]
    finally:
        # Clean up
        os.unlink(temp_path)


def test_get_markdown_files():
    """Test that markdown files are correctly found in a directory."""
    # Create a temporary directory with markdown and non-markdown files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create markdown files
        with open(os.path.join(temp_dir, "file1.md"), "w") as f:
            f.write("# File 1")
        with open(os.path.join(temp_dir, "file2.md"), "w") as f:
            f.write("# File 2")
        
        # Create a non-markdown file
        with open(os.path.join(temp_dir, "file3.txt"), "w") as f:
            f.write("This is not a markdown file")
        
        # Create a subdirectory with a markdown file
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "file4.md"), "w") as f:
            f.write("# File 4")
        
        # Get markdown files
        markdown_files = get_markdown_files(temp_dir)
        
        # Verify that only markdown files were found
        assert len(markdown_files) == 3
        file_names = [os.path.basename(f) for f in markdown_files]
        assert "file1.md" in file_names
        assert "file2.md" in file_names
        assert "file4.md" in file_names
        assert "file3.txt" not in file_names
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def test_markdown_parser_availability():
    """Test that at least one markdown parser is available."""
    try:
        import markdown_it
        assert MARKDOWN_IT_AVAILABLE is True
    except ImportError:
        try:
            import mistune
            assert MARKDOWN_IT_AVAILABLE is False
        except ImportError:
            pytest.skip("No markdown parser available")


def test_fallback_to_regex_parsing():
    """Test that regex parsing is used as a fallback when no parser is available."""
    # This is a bit tricky to test since we need to simulate missing parsers
    # For now, we'll just check that sections can be extracted regardless of parser
    
    markdown_content = """# Title
    
This is an introduction.

## Section 1

This is section 1 content.
"""
    
    # Even without advanced parsers, we should get at least basic sections
    sections = extract_sections_from_markdown(markdown_content)
    assert len(sections) >= 2  # Title, Section 1 


def test_extract_indented_code_blocks():
    """Test extracting indented code blocks from markdown content."""
    # Markdown content with indented code blocks
    markdown_content = """# Test Markdown

This is a test markdown file with indented code blocks.

## Python Example

    # Python code block
    def hello():
        print("Hello, World!")

## JavaScript Example

    // JavaScript code block
    function greet() {
        console.log("Hello, World!");
    }
"""
    
    # Extract code blocks
    code_blocks = extract_code_blocks(markdown_content)
    
    # Verify code blocks were extracted
    assert len(code_blocks) == 2
    
    # Check if Python block was detected correctly
    python_block_found = False
    for lang, content in code_blocks.items():
        if lang == "python" or lang.startswith("python_"):
            python_block_found = True
            assert "def hello():" in content
            assert "print(\"Hello, World!\")" in content
    
    assert python_block_found, "Python code block was not detected"
    
    # Check if JavaScript block was detected correctly
    js_block_found = False
    for lang, content in code_blocks.items():
        if lang == "javascript" or lang.startswith("javascript_"):
            js_block_found = True
            assert "function greet()" in content
            assert "console.log(\"Hello, World!\");" in content
    
    assert js_block_found, "JavaScript code block was not detected" 