"""
Test markdown extraction functionality.

This module tests markdown content extraction, including:
1. Section extraction
2. Code block extraction
3. Metadata parsing
4. Block validation
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
import shutil

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
try:
    from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor import (
        extract_markdown_blocks,
        _extract_with_markdown_it,
        _extract_with_regex
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"Error importing markdown extractor: {e}")
    raise ImportError(f"Required markdown extractor not available: {e}. Fix the dependencies to run these tests.")

# Only run tests if imports are available
pytestmark = pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Required markdown extractor not available")

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

def test_extract_markdown_blocks(temp_dir, stats_dict):
    """Test Markdown block extraction."""
    # Create a test file
    test_file = temp_dir / "test.md"
    content = """# Section 1
This is the first section.

# Section 2
This is the second section.

## Subsection 2.1
This is a subsection.

# Section 3
Final section with code:
```python
def hello():
    print("Hello")
```
"""
    test_file.write_text(content)
    
    # Extract blocks
    blocks = extract_markdown_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict
    )
    
    assert blocks > 0
    assert stats_dict["doc_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check block files were created
    blocks_dir = temp_dir / "blocks" / "docs" / "markdown"
    assert blocks_dir.exists()
    assert len(list(blocks_dir.glob("*.md"))) > 0

def test_process_documentation_file(temp_dir, stats_dict):
    """Test documentation file processing."""
    # Create a test file
    test_file = temp_dir / "test.md"
    content = """# Documentation
This is a test documentation file.

## Section 1
Content of section 1.

## Section 2
Content of section 2.
"""
    test_file.write_text(content)
    
    # Process the file
    extract_markdown_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict,
        extract_blocks=True
    )
    
    assert stats_dict["documentation_files"] == 1
    assert "markdown" in stats_dict["languages"]
    assert stats_dict["doc_blocks"] > 0

def test_nested_sections(temp_dir, stats_dict):
    """Test extraction of nested Markdown sections."""
    # Create a test file
    test_file = temp_dir / "nested.md"
    content = """# Main Section
Main content.

## Subsection 1
Subsection 1 content.

### Sub-subsection 1.1
Deeper nested content.

## Subsection 2
Subsection 2 content.

### Sub-subsection 2.1
More nested content.
"""
    test_file.write_text(content)
    
    # Extract blocks
    blocks = extract_markdown_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict
    )
    
    assert blocks > 0
    assert stats_dict["doc_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check that all sections were extracted
    file_blocks = stats_dict["file_blocks"][str(test_file)]
    section_titles = {block["title"] for block in file_blocks}
    assert "Main_Section" in section_titles
    assert "Subsection_1" in section_titles
    assert "Sub_subsection_1_1" in section_titles

    # Verify block format matches specification
    main_section = next(block for block in file_blocks if block["title"] == "Main_Section")
    assert "uuid" in main_section
    assert "id" in main_section
    assert main_section["type"] == "section"
    assert main_section["language"] == "markdown"
    assert "original_title" in main_section
    assert "content" in main_section
    assert "file_path" in main_section
    assert "breadcrumb" in main_section
    assert isinstance(main_section["child_uuids"], list)
    assert isinstance(main_section["depth"], int)
    assert isinstance(main_section["header_depth"], list)
    assert isinstance(main_section["content_flags"], dict)
    assert "section_role" in main_section
    assert "toc_format" in main_section
    assert "extraction_focus" in main_section
    assert "summary_instructions" in main_section
    assert isinstance(main_section["qa_generation"], dict)
    assert "difficulty_levels" in main_section["qa_generation"]
    assert "knowledge_prerequisites" in main_section["qa_generation"]
    assert "focus_areas" in main_section["qa_generation"]
    assert "qa_examples" in main_section["qa_generation"]

    # Verify hierarchical relationships
    subsection = next(block for block in file_blocks if block["title"] == "Subsection_1")
    assert subsection["parent_uuid"] == main_section["uuid"]
    assert subsection["depth"] == main_section["depth"] + 1
    assert len(subsection["breadcrumb"]) == subsection["depth"] + 1

def test_markdown_extraction_error_handling(temp_dir, stats_dict):
    """Test error handling in Markdown extraction."""
    # Create an invalid file path
    invalid_file = temp_dir / "nonexistent.md"
    
    # Try to extract blocks
    blocks = extract_markdown_blocks(
        invalid_file,
        "some content",
        temp_dir,
        stats_dict
    )
    
    assert blocks == 0
    assert len(stats_dict["errors"]) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 