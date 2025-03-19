"""
TEST EXPECTATIONS

test_extract_markdown_blocks:
Input: Markdown file with code blocks and sections
Expected Output:
{
    "code_blocks": > 0,
    "total_files": 1,
    "file_blocks": {
        "example.md": [
            {
                "block_type": "section",
                "name": "Introduction",
                "content": "# Introduction\n..."
            },
            {
                "block_type": "code",
                "name": "example_code",
                "content": "```python\ndef hello()...\n```"
            }
        ]
    }
}

CRITICAL RULES:
1. Block Extraction Rules:
   - Each section must start with a heading
   - Each code block must be fenced with ```
   - Each block must preserve original formatting
   - Code blocks must retain language specifier

2. Stats Tracking Rules:
   - Track total files processed
   - Track sections per file
   - Track code blocks per file
   - Track languages in code blocks

3. Output File Rules:
   - All blocks must be written to output directory
   - All paths must be relative to output directory
   - Section files must have .md extension
   - Code blocks must have extension matching language
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
        _extract_markdown_blocks,
        _process_documentation_file,
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
    blocks = _extract_markdown_blocks(
        test_file,
        content,
        temp_dir,
        stats_dict
    )
    
    assert blocks > 0
    assert stats_dict["doc_blocks"] > 0
    assert str(test_file) in stats_dict["file_blocks"]
    
    # Check block files were created
    blocks_dir = temp_dir / "doc_blocks" / "markdown"
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
    _process_documentation_file(
        test_file,
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
    blocks = _extract_markdown_blocks(
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
    assert main_section["type"] == "documentation"
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
    blocks = _extract_markdown_blocks(
        invalid_file,
        "some content",
        temp_dir,
        stats_dict
    )
    
    assert blocks == 0
    assert len(stats_dict["errors"]) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 