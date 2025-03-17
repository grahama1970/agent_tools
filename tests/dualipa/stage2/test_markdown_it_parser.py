import pytest
import tempfile
import os
import json
from pathlib import Path
from textwrap import dedent

from src.agent_tools.dualipa.markdown_it_parser import (
    MARKDOWN_IT_AVAILABLE,
    markdown_to_hierarchical_json,
    extract_code_blocks,
    extract_content_blocks,
    build_section_hierarchy,
    process_markdown_file,
    get_flattened_markdown_content,
    flatten_hierarchy
)

# Sample markdown content for testing
SAMPLE_MARKDOWN = """# Test Document

This is a test paragraph with some content.

## Section 1

This is content in section 1.

```python
def hello_world():
    print("Hello, World!")
```

### Nested Section 1.1

This is content in a nested section.

![Sample Image](https://example.com/image.png "Image Title")

## Section 2

This is content in section 2.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

- List item 1
- List item 2
- List item 3
"""

@pytest.fixture
def markdown_file():
    """Create a temporary markdown file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as f:
        f.write(SAMPLE_MARKDOWN)
        file_path = f.name
    
    yield file_path
    
    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)


def test_markdown_it_availability():
    """Test if markdown-it is available."""
    try:
        import markdown_it
        assert True, "markdown-it-py is available"
    except ImportError:
        pytest.fail("markdown-it-py is not installed - this test requires it to run")


def test_extract_code_blocks():
    """Test extraction of code blocks from markdown."""
    try:
        import markdown_it
    except ImportError:
        pytest.fail("markdown-it-py not available - install it to run this test")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Check code blocks
    code_blocks = result["code_blocks"]
    assert len(code_blocks) == 1
    
    # Verify python code block
    python_block = code_blocks[0]
    assert python_block["language"] == "python"
    assert "def hello_world()" in python_block["content"]
    assert "print(\"Hello, World!\")" in python_block["content"]
    assert "token_count" in python_block
    assert python_block["token_count"] > 0
    assert "metadata" in python_block
    assert "token_count" in python_block["metadata"]


def test_extract_images():
    """Test extraction of images from markdown."""
    try:
        import markdown_it
    except ImportError:
        pytest.fail("markdown-it-py not available - install it to run this test")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Find image blocks in the hierarchy
    image_blocks = []
    for section_name, section in result["document"]["hierarchy"].items():
        for block in section["content"]:
            if block.get("type") == "image":
                image_blocks.append(block)
        
        # Check subsections
        for subsection_name, subsection in section.get("subsections", {}).items():
            for block in subsection.get("content", []):
                if block.get("type") == "image":
                    image_blocks.append(block)
                    
            # Check nested subsections (level 3)
            for nested_name, nested_section in subsection.get("subsections", {}).items():
                for block in nested_section.get("content", []):
                    if block.get("type") == "image":
                        image_blocks.append(block)
    
    # Verify image blocks
    assert len(image_blocks) == 1
    image = image_blocks[0]
    assert image["type"] == "image"
    assert image["src"] == "https://example.com/image.png"
    assert image["alt"] == "Sample Image"
    assert image["title"] == "Image Title"
    assert "token_count" in image
    assert "metadata" in image
    assert "src" in image["metadata"]


def test_extract_tables():
    """Test extraction of tables from markdown."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Find table blocks in the hierarchy
    table_blocks = []
    for section_name, section in result["document"]["hierarchy"].items():
        for block in section["content"]:
            if block.get("type") == "table":
                table_blocks.append(block)
        
        # Check subsections
        for subsection_name, subsection in section.get("subsections", {}).items():
            for block in subsection.get("content", []):
                if block.get("type") == "table":
                    table_blocks.append(block)
    
    # Verify table blocks
    assert len(table_blocks) == 1
    table = table_blocks[0]
    assert table["type"] == "table"
    assert len(table["header"]) == 3
    assert table["header"] == ["Header 1", "Header 2", "Header 3"]
    assert len(table["rows"]) == 2
    assert table["rows"][0] == ["Cell 1", "Cell 2", "Cell 3"]
    assert table["rows"][1] == ["Cell 4", "Cell 5", "Cell 6"]
    assert "token_count" in table
    assert table["token_count"] > 0
    assert "metadata" in table
    assert "row_count" in table["metadata"]
    assert table["metadata"]["row_count"] == 2
    assert "column_count" in table["metadata"]
    assert table["metadata"]["column_count"] == 3


def test_extract_lists():
    """Test extraction of lists from markdown."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Find list blocks in the hierarchy
    list_blocks = []
    for section_name, section in result["document"]["hierarchy"].items():
        for block in section["content"]:
            if block.get("type") == "list":
                list_blocks.append(block)
        
        # Check subsections
        for subsection_name, subsection in section.get("subsections", {}).items():
            for block in subsection.get("content", []):
                if block.get("type") == "list":
                    list_blocks.append(block)
    
    # Verify list blocks
    assert len(list_blocks) == 1
    list_block = list_blocks[0]
    assert list_block["type"] == "list"
    assert list_block["list_type"] == "unordered"
    assert len(list_block["items"]) == 3
    assert "List item 1" in list_block["items"]
    assert "List item 2" in list_block["items"]
    assert "List item 3" in list_block["items"]
    assert "token_count" in list_block
    assert list_block["token_count"] > 0
    assert "metadata" in list_block
    assert "item_count" in list_block["metadata"]
    assert list_block["metadata"]["item_count"] == 3


def test_section_hierarchy():
    """Test building of section hierarchy from markdown."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Check hierarchy structure
    hierarchy = result["document"]["hierarchy"]
    
    # Verify top-level section
    assert "Test Document" in hierarchy
    assert hierarchy["Test Document"]["level"] == 1
    assert "token_count" in hierarchy["Test Document"]
    assert hierarchy["Test Document"]["token_count"] > 0
    assert "metadata" in hierarchy["Test Document"]
    assert "total_token_count_with_subsections" in hierarchy["Test Document"]["metadata"]
    
    # Verify subsections
    assert "Section 1" in hierarchy["Test Document"]["subsections"]
    assert hierarchy["Test Document"]["subsections"]["Section 1"]["level"] == 2
    assert "token_count" in hierarchy["Test Document"]["subsections"]["Section 1"]
    
    # Verify nested subsections
    assert "Nested Section 1.1" in hierarchy["Test Document"]["subsections"]["Section 1"]["subsections"]
    assert hierarchy["Test Document"]["subsections"]["Section 1"]["subsections"]["Nested Section 1.1"]["level"] == 3
    assert "token_count" in hierarchy["Test Document"]["subsections"]["Section 1"]["subsections"]["Nested Section 1.1"]
    
    # Verify another subsection
    assert "Section 2" in hierarchy["Test Document"]["subsections"]
    assert hierarchy["Test Document"]["subsections"]["Section 2"]["level"] == 2
    assert "token_count" in hierarchy["Test Document"]["subsections"]["Section 2"]
    
    # Verify hierarchy token counts
    assert hierarchy["Test Document"]["metadata"]["total_token_count_with_subsections"] > 0
    
    # The total token count with subsections should be the sum of all section token counts
    total_from_sections = hierarchy["Test Document"]["token_count"]
    for subsection_name, subsection in hierarchy["Test Document"]["subsections"].items():
        total_from_sections += subsection["token_count"]
        for nested_name, nested_section in subsection.get("subsections", {}).items():
            total_from_sections += nested_section["token_count"]
    
    # The top-level total should include all subsections
    assert hierarchy["Test Document"]["metadata"]["total_token_count_with_subsections"] == total_from_sections


def test_file_metadata():
    """Test file metadata extraction."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(SAMPLE_MARKDOWN)
        file_path = f.name
    
    try:
        result = process_markdown_file(file_path)
        
        # Check file metadata
        file_info = result["document"]["file_info"]
        assert file_info["path"] == file_path
        assert file_info["filename"] == os.path.basename(file_path)
        assert file_info["extension"] == ".md"
        assert file_info["directory"] == str(Path(file_path).parent)
        assert file_info["line_count"] > 0
        assert "token_count" in file_info
        assert file_info["token_count"] > 0
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)


def test_flatten_hierarchy():
    """Test flattening of section hierarchy."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process markdown
    result = markdown_to_hierarchical_json(SAMPLE_MARKDOWN)
    
    # Flatten hierarchy
    flat_blocks = flatten_hierarchy(result["document"]["hierarchy"])
    
    # Verify flat blocks
    assert len(flat_blocks) > 0
    
    # Check section paths in flat blocks
    section_paths = set()
    for block in flat_blocks:
        assert "section" in block
        assert "section_level" in block
        assert "metadata" in block
        assert "section_path" in block["metadata"]
        assert "section_level" in block["metadata"]
        section_paths.add(block["section"])
    
    # Verify expected section paths
    assert "Test Document" in section_paths
    assert "Test Document > Section 1" in section_paths
    assert "Test Document > Section 1 > Nested Section 1.1" in section_paths
    assert "Test Document > Section 2" in section_paths


def test_process_markdown_file(markdown_file):
    """Test processing a markdown file from disk."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Process the file
    result = process_markdown_file(markdown_file)
    
    # Basic validation
    assert "document" in result
    assert "file_info" in result["document"]
    assert "hierarchy" in result["document"]
    assert "code_blocks" in result
    
    # Check file info
    file_info = result["document"]["file_info"]
    assert file_info["path"] == markdown_file
    assert "token_count" in file_info
    assert file_info["token_count"] > 0
    
    # Verify that the token count in file_info matches the content
    hierarchy = result["document"]["hierarchy"]
    total_section_tokens = hierarchy["Test Document"]["metadata"]["total_token_count_with_subsections"]
    assert total_section_tokens > 0
    
    # The file token count may differ slightly from the section token count
    # due to extra spaces, newlines, etc.
    assert abs(file_info["token_count"] - total_section_tokens) < file_info["token_count"] * 0.1


def test_get_flattened_markdown_content(markdown_file):
    """Test getting flattened markdown content."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Get flattened content
    result = get_flattened_markdown_content(markdown_file)
    
    # Verify result
    assert "document" in result
    assert "flat_blocks" in result["document"]
    assert "hierarchy" in result["document"]
    
    # Check flat blocks
    flat_blocks = result["document"]["flat_blocks"]
    assert len(flat_blocks) > 0
    
    # Count elements by type
    type_counts = {}
    for block in flat_blocks:
        assert "type" in block
        assert "section" in block
        assert "section_level" in block
        
        block_type = block["type"]
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    
    # Verify expected block types and counts
    assert type_counts.get("paragraph", 0) > 0
    assert type_counts.get("code", 0) > 0
    assert type_counts.get("image", 0) > 0
    assert type_counts.get("table", 0) > 0
    assert type_counts.get("list", 0) > 0


def test_token_counting_accuracy():
    """Test the accuracy of token counting."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Simple test markdown with known token counts
    simple_markdown = """# Hello World
    
This is a simple paragraph with exactly 10 tokens.

```python
# This code block has 7 tokens
print("Hello World")
```

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |
"""
    
    # Process markdown
    result = markdown_to_hierarchical_json(simple_markdown)
    
    # Check document token count
    file_info = result["document"]["file_info"]
    assert "token_count" in file_info
    assert file_info["token_count"] > 20  # Should be more than 20 tokens
    
    # Find the paragraph and verify token count
    paragraph_found = False
    code_block_found = False
    
    for section_name, section in result["document"]["hierarchy"].items():
        for block in section["content"]:
            if block["type"] == "paragraph" and "simple paragraph" in block["content"]:
                paragraph_found = True
                # Allow for some flexibility in token counting implementations
                assert 8 <= block["token_count"] <= 12
            
            if block["type"] == "code" and "python" in block["language"]:
                code_block_found = True
                assert 5 <= block["token_count"] <= 9
    
    assert paragraph_found, "Paragraph with known token count was not found"
    assert code_block_found, "Code block with known token count was not found"
    
    # Verify code blocks from the main extraction
    for block in result["code_blocks"]:
        if "python" in block["language"] and "Hello World" in block["content"]:
            assert 5 <= block["token_count"] <= 9


def test_complex_nested_structure():
    """Test processing of complex nested document structure."""
    if not MARKDOWN_IT_AVAILABLE:
        pytest.skip("markdown-it not available")
    
    # Complex markdown with deeply nested structure
    complex_markdown = """# Root
## Level 2-A
### Level 3-A
#### Level 4-A
Some content in level 4-A.
### Level 3-B
Some content in level 3-B.
## Level 2-B
Some content in level 2-B.
### Level 3-C
#### Level 4-B
##### Level 5-A
###### Level 6-A
Deep nesting content.
"""
    
    # Process markdown
    result = markdown_to_hierarchical_json(complex_markdown)
    
    # Verify hierarchy depth
    hierarchy = result["document"]["hierarchy"]
    assert "Root" in hierarchy
    assert "Level 2-A" in hierarchy["Root"]["subsections"]
    assert "Level 3-A" in hierarchy["Root"]["subsections"]["Level 2-A"]["subsections"]
    assert "Level 4-A" in hierarchy["Root"]["subsections"]["Level 2-A"]["subsections"]["Level 3-A"]["subsections"]
    
    # Verify token counts are propagated correctly
    root_total = hierarchy["Root"]["metadata"]["total_token_count_with_subsections"]
    root_own = hierarchy["Root"]["token_count"]
    
    # Root's total should be greater than its own tokens
    assert root_total > root_own
    
    # Verify the deepest nesting
    level_2b = hierarchy["Root"]["subsections"]["Level 2-B"]
    assert "Level 3-C" in level_2b["subsections"]
    level_3c = level_2b["subsections"]["Level 3-C"]
    assert "Level 4-B" in level_3c["subsections"]
    level_4b = level_3c["subsections"]["Level 4-B"]
    assert "Level 5-A" in level_4b["subsections"]
    level_5a = level_4b["subsections"]["Level 5-A"]
    assert "Level 6-A" in level_5a["subsections"]
    
    # Verify content in the deepest level
    level_6a = level_5a["subsections"]["Level 6-A"]
    assert any("Deep nesting content" in block.get("content", "") for block in level_6a["content"])
    assert level_6a["token_count"] > 0 