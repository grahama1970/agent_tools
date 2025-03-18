"""
Tests for the Markdown hierarchy extraction.

This module verifies that the markdown parser correctly extracts hierarchical sections
using real-world markdown examples from actual repositories instead of synthetic examples.
This ensures the parser works on realistic documentation patterns.
"""

import os
import tempfile
from pathlib import Path
import json
import pytest
import requests
import sys

# Configure path correctly
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Flag to track if dependencies are available
HAS_DEPENDENCIES = False
try:
    from agent_tools.dualipa.markdown_hierarchy import extract_hierarchical_sections
    HAS_DEPENDENCIES = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("Skipping tests that require missing modules")

# Skip all tests in this file - the extract_hierarchical_sections function 
# is either missing or not working properly
# Remove skipif decorator to make tests fail loudly
# pytestmark = pytest.mark.skipif(
#     not HAS_DEPENDENCIES,
#     reason="Required markdown hierarchy modules not available"
# )

# Real repository markdown files to test
MARKDOWN_SOURCES = {
    'readme': 'https://raw.githubusercontent.com/pallets/flask/main/README.md',
    'contributing': 'https://raw.githubusercontent.com/pandas-dev/pandas/main/CONTRIBUTING.md',
    'changelog': 'https://raw.githubusercontent.com/tiangolo/fastapi/main/CHANGELOG.md',
    'advanced_guide': 'https://raw.githubusercontent.com/expressjs/express/master/Readme.md',
    'typescript_overview': 'https://raw.githubusercontent.com/microsoft/TypeScript/main/README.md',
    'django_readme': 'https://raw.githubusercontent.com/django/django/main/README.rst', # RST format
    'pytorch_contributing': 'https://raw.githubusercontent.com/pytorch/pytorch/main/CONTRIBUTING.md',
    'golang_readme': 'https://raw.githubusercontent.com/golang/go/master/README.md'
}

def fetch_real_markdown(source_key):
    """Fetch real markdown content from the specified source."""
    url = MARKDOWN_SOURCES.get(source_key)
    
    if not url:
        return "# Sample Markdown\n\nCould not fetch real example."
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    
    # Fallback content if fetching fails
    return """# Sample Markdown
    
## Introduction

This is a sample markdown document with several sections.

## Features

- Feature 1
- Feature 2

### Sub-feature 1

Some details about sub-feature 1.

### Sub-feature 2

Some details about sub-feature 2.

## Conclusion

This is the conclusion.
"""

def visualize_hierarchy(sections, indent=0):
    """Helper function to visualize the hierarchy for debugging."""
    result = []
    prefix = "  " * indent
    
    for section in sections:
        title = section.get("title", "No title")
        level = section.get("level", 0)
        result.append(f"{prefix}- [{level}] {title}")
        
        if "subsections" in section and section["subsections"]:
            child_result = visualize_hierarchy(section["subsections"], indent + 1)
            result.extend(child_result)
    
    return result

@pytest.fixture
def real_readme_content():
    """Fixture to provide real README content."""
    return fetch_real_markdown('readme')

@pytest.fixture
def real_contributing_content():
    """Fixture to provide real CONTRIBUTING guide content."""
    return fetch_real_markdown('contributing')

@pytest.fixture
def real_changelog_content():
    """Fixture to provide real CHANGELOG content."""
    return fetch_real_markdown('changelog')

@pytest.fixture
def real_advanced_guide_content():
    """Fixture to provide real advanced guide content."""
    return fetch_real_markdown('advanced_guide')

def check_markdown_dependencies():
    """Check if required markdown modules are available and fail if not."""
    if not HAS_DEPENDENCIES:
        pytest.fail("Required markdown hierarchy modules not available. Install markdown-it-py and other dependencies.")

def test_basic_hierarchy_extraction(real_readme_content):
    """Test extraction of hierarchical sections from real README markdown."""
    # Check dependencies first
    check_markdown_dependencies()
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_readme_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify we got something
        assert sections is not None, "Should extract sections"
        assert isinstance(sections, list), "Should return a list of sections"
        assert len(sections) > 0, "Should extract at least one section"
        
        # Print sections for debugging
        print("\nExtracted sections from README:")
        print("\n".join(visualize_hierarchy(sections)))
        
        # Verify structure
        for section in sections:
            assert "title" in section, "Section should have a title"
            assert "level" in section, "Section should have a level"
            assert "content" in section, "Section should have content"
            assert "start_line" in section, "Section should have a start line"
            assert "end_line" in section, "Section should have an end line"
            
            # If it has subsections, verify them
            if "subsections" in section and section["subsections"]:
                for subsection in section["subsections"]:
                    assert "title" in subsection, "Subsection should have a title"
                    assert "level" in subsection, "Subsection should have a level"
                    assert "content" in subsection, "Subsection should have content"
                    assert "start_line" in subsection, "Subsection should have a start line"
                    assert "end_line" in subsection, "Subsection should have an end line"
                    
                    # Verify subsection has higher level than parent
                    assert subsection["level"] > section["level"], "Subsection should have higher level than parent"

def test_complex_hierarchy_extraction(real_contributing_content):
    """Test extraction of complex hierarchical sections from CONTRIBUTING.md."""
    check_markdown_dependencies()
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_contributing_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify we got something
        assert sections is not None, "Should extract sections"
        assert isinstance(sections, list), "Should return a list of sections"
        assert len(sections) > 0, "Should extract at least one section"
        
        # Print sections for debugging
        print("\nExtracted sections from CONTRIBUTING guide:")
        print("\n".join(visualize_hierarchy(sections)))
        
        # Verify we have a multi-level hierarchy
        has_deep_nesting = False
        max_depth = 0
        
        def check_nesting(section_list, current_depth=1):
            nonlocal has_deep_nesting, max_depth
            
            for section in section_list:
                if "subsections" in section and section["subsections"]:
                    if current_depth >= 2:
                        has_deep_nesting = True
                    
                    max_depth = max(max_depth, current_depth)
                    check_nesting(section["subsections"], current_depth + 1)
        
        check_nesting(sections)
        
        print(f"\nMax hierarchy depth: {max_depth}")
        assert max_depth > 1, "Should have at least some nesting in complex document"

def test_section_content_extraction(real_readme_content):
    """Test extraction of section content from real README markdown."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_readme_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify content
        for section in sections:
            assert section["content"], "Section should have non-empty content"
            assert section["title"] in section["content"], "Section content should include the title"
            
            # Check subsections recursively
            def check_subsection_content(subsections):
                for subsect in subsections:
                    assert subsect["content"], "Subsection should have non-empty content"
                    assert subsect["title"] in subsect["content"], "Subsection content should include the title"
                    
                    if "subsections" in subsect and subsect["subsections"]:
                        check_subsection_content(subsect["subsections"])
            
            if "subsections" in section and section["subsections"]:
                check_subsection_content(section["subsections"])

def test_code_block_in_sections(real_advanced_guide_content):
    """Test handling of markdown code blocks within sections from real guide content."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_advanced_guide_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for code blocks
        found_code_blocks = False
        
        def check_for_code_blocks(section_list):
            nonlocal found_code_blocks
            for section in section_list:
                # Check if content has code blocks (```...```)
                if "```" in section["content"]:
                    found_code_blocks = True
                    # Verify code blocks are properly included in section content
                    code_block_start = section["content"].find("```")
                    code_block_end = section["content"].find("```", code_block_start + 3)
                    
                    # Skip if no closing marker
                    if code_block_end == -1:
                        continue
                    
                    code_block = section["content"][code_block_start:code_block_end + 3]
                    assert len(code_block) > 6, "Code block should have content"
                    print(f"\nFound code block in section '{section['title']}': {code_block[:50]}...")
                
                if "subsections" in section and section["subsections"]:
                    check_for_code_blocks(section["subsections"])
        
        check_for_code_blocks(sections)
        
        # If we didn't find any code blocks, try another document
        if not found_code_blocks:
            print("\nNo code blocks found in the advanced guide, trying another document...")
            # Try alternate content that likely has code blocks
            typescript_readme = fetch_real_markdown('typescript_overview')
            
            with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f2:
                f2.write(typescript_readme)
                f2.flush()
                
                sections = extract_hierarchical_sections(f2.name)
                check_for_code_blocks(sections)
        
        # If we still have no code blocks, skip the assertion
        if not found_code_blocks:
            pytest.fail("No code blocks found in any tested documents. Test data may be missing code blocks.")
        else:
            assert found_code_blocks, "Should find code blocks in real documentation"

def test_list_handling_in_sections(real_contributing_content):
    """Test handling of markdown lists within sections from real guide content."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_contributing_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for lists (lines starting with - or *)
        found_lists = False
        
        def check_for_lists(section_list):
            nonlocal found_lists
            for section in section_list:
                content_lines = section["content"].split("\n")
                for line in content_lines:
                    stripped = line.strip()
                    if stripped.startswith("- ") or stripped.startswith("* "):
                        found_lists = True
                        print(f"\nFound list item in section '{section['title']}': {stripped[:50]}...")
                        break
                
                if found_lists:
                    break
                    
                if "subsections" in section and section["subsections"]:
                    check_for_lists(section["subsections"])
                    if found_lists:
                        break
        
        check_for_lists(sections)
        
        # If we found no lists, try another document
        if not found_lists:
            print("\nNo lists found in the contributing guide, trying another document...")
            # Try alternate content that likely has lists
            typescript_readme = fetch_real_markdown('typescript_overview')
            
            with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f2:
                f2.write(typescript_readme)
                f2.flush()
                
                sections = extract_hierarchical_sections(f2.name)
                check_for_lists(sections)
        
        # If we still have no lists, skip the assertion
        if not found_lists:
            pytest.fail("No lists found in any tested documents. Test data may be missing lists.")
        else:
            assert found_lists, "Should find lists in real documentation"

def test_section_line_numbers(real_readme_content):
    """Test that section line numbers are valid in real README markdown."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_readme_content)
        f.flush()
        
        # Count lines in the file
        with open(f.name, 'r') as f_read:
            file_lines = len(f_read.readlines())
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify line numbers
        def check_section_lines(section_list):
            for section in section_list:
                assert section["start_line"] >= 1, "Start line should be at least 1"
                assert section["end_line"] <= file_lines, f"End line should not exceed file length ({file_lines})"
                assert section["start_line"] <= section["end_line"], "Start line should not exceed end line"
                
                if "subsections" in section and section["subsections"]:
                    check_section_lines(section["subsections"])
        
        check_section_lines(sections)

def test_multiple_files_extraction():
    """Test extraction from multiple real markdown files to ensure consistency."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple files with different markdown content
        file_paths = {}
        results = {}
        
        for source_key in ['readme', 'contributing', 'changelog']:
            content = fetch_real_markdown(source_key)
            file_path = os.path.join(temp_dir, f"{source_key}.md")
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            file_paths[source_key] = file_path
            
            # Extract sections
            sections = extract_hierarchical_sections(file_path)
            results[source_key] = len(sections)
            
            # Verify we got results
            assert len(sections) > 0, f"Should extract sections from {source_key}"
        
        # Print debug info
        print("\nExtracted section counts:")
        for source, count in results.items():
            print(f"  {source}: {count} sections")

def test_non_english_content():
    """Test handling of non-English markdown content."""
    # Create a sample non-English markdown
    non_english_content = """# Título en Español
    
## Introducción

Este es un ejemplo de contenido en español.

## Características

- Característica 1
- Característica 2

### Sub-característica 1

Algunos detalles sobre la sub-característica 1.

## Conclusión

Esta es la conclusión.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(non_english_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify we got something
        assert sections is not None, "Should extract sections from non-English content"
        assert len(sections) > 0, "Should extract at least one section from non-English content"
        
        # Verify titles are preserved
        titles = [section["title"] for section in sections]
        assert "Título en Español" in titles, "Should preserve non-English titles"
        
        # Print sections for debugging
        print("\nExtracted non-English sections:")
        print("\n".join(visualize_hierarchy(sections)))

def test_empty_sections_handling():
    """Test handling of empty sections in markdown."""
    # Create markdown with empty sections
    empty_sections_content = """# Main Title

## Section 1

Content for section 1.

## Section 2

## Section 3

Content for section 3.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(empty_sections_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify sections
        assert len(sections) > 0, "Should extract sections with empty sections"
        
        # Check if Section 2 is found
        section2_found = False
        
        def find_section2(section_list):
            for section in section_list:
                if section["title"] == "Section 2":
                    # Verify it has minimal content (at least its own header)
                    assert "Section 2" in section["content"], "Empty section should at least contain its header"
                    return True
                
                # Check subsections recursively
                if 'subsections' in section and section['subsections']:
                    if find_section2(section['subsections']):
                        return True
            return False
        
        section2_found = find_section2(sections)
        assert section2_found, "Should extract empty sections"

def test_malformed_markdown_handling():
    """Test handling of malformed markdown."""
    # Create malformed markdown content
    malformed_content = """# Title without closing #

## Unclosed section

# Random title levels

### Third level header
# First level again

Content without a section header
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(malformed_content)
        f.flush()
        
        # Extract sections - should not crash
        sections = extract_hierarchical_sections(f.name)
        
        # Verify we got something
        assert sections is not None, "Should extract something from malformed markdown"
        
        # Print sections for debugging
        print("\nExtracted sections from malformed markdown:")
        print("\n".join(visualize_hierarchy(sections)))

def test_section_hierarchy_relationships(real_changelog_content):
    """Test parent-child relationships in section hierarchy from real changelog."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_changelog_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify hierarchy relationships
        def verify_hierarchy(section_list, expected_min_level=1):
            for section in section_list:
                assert section["level"] >= expected_min_level, f"Section level should be at least {expected_min_level}"
                
                if "subsections" in section and section["subsections"]:
                    # Subsections should have higher level than parent
                    for subsection in section["subsections"]:
                        assert subsection["level"] > section["level"], "Subsection should have higher level than parent"
                    
                    # Recursive check
                    verify_hierarchy(section["subsections"], section["level"] + 1)
        
        verify_hierarchy(sections)

def test_variety_of_markdown_syntax(real_contributing_content):
    """Test handling of various markdown syntax elements."""
    # Create markdown with various formatting elements
    markdown_with_elements = """# Document with Formatting

## Bold Section

This section has **bold text** and __also bold__.

## Italic Section

This section has *italic text* and _also italic_.

## Links Section

This section has [links](https://example.com) and [more links](https://example.org).

## Code Section

This section has `inline code` and also:

```python
def hello():
    print("Hello world")
```

## Blockquote Section

> This is a blockquote.
> Multiple lines.

## Mixed Section

**Bold**, *italic*, `code`, [link](https://example.com), and > quote.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(markdown_with_elements)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for various markdown elements
        found_elements = {
            "bold": False,
            "italic": False,
            "link": False,
            "codespan": False,
            "blockquote": False
        }
        
        def check_for_markdown_elements(section_list):
            for section in section_list:
                content = section["content"]
                
                if "**" in content or "__" in content:
                    found_elements["bold"] = True
                
                if "*" in content and "*" != content.find("*", content.find("*") + 1):
                    found_elements["italic"] = True
                
                if "[" in content and "](" in content:
                    found_elements["link"] = True
                
                if "`" in content:
                    found_elements["codespan"] = True
                
                lines = content.split("\n")
                for line in lines:
                    if line.strip().startswith(">"):
                        found_elements["blockquote"] = True
                
                if "subsections" in section and section["subsections"]:
                    check_for_markdown_elements(section["subsections"])
        
        check_for_markdown_elements(sections)
        
        # Print found elements
        print("\nMarkdown elements found:")
        for element, found in found_elements.items():
            print(f"  {element}: {found}")
        
        # Verify at least some elements were found
        assert any(found_elements.values()), "Should find some markdown syntax elements"

def test_section_metadata_extraction():
    """Test that section metadata is extracted properly."""
    # Create markdown with metadata
    metadata_content = """# Title with Metadata

> Status: Draft
> Author: Test User
> Date: 2023-03-15

## Section 1

Content for section 1.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(metadata_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Verify metadata is included in content
        assert len(sections) > 0, "Should extract sections"
        main_section = sections[0]
        
        assert "Status: Draft" in main_section["content"], "Should include metadata in content"
        assert "Author: Test" in main_section["content"], "Should include metadata in content"

def test_large_document_handling():
    """Test handling of large markdown documents."""
    # Combine multiple real markdown files to create a large document
    with tempfile.TemporaryDirectory() as temp_dir:
        combined_content = ""
        for source_key in ['readme', 'contributing', 'changelog', 'advanced_guide']:
            content = fetch_real_markdown(source_key)
            combined_content += f"\n\n# Document: {source_key}\n\n{content}"
        
        large_file_path = os.path.join(temp_dir, "large_document.md")
        with open(large_file_path, 'w') as f:
            f.write(combined_content)
        
        # Extract sections - ensure it doesn't crash or timeout
        try:
            sections = extract_hierarchical_sections(large_file_path)
            assert sections is not None, "Should extract sections from large document"
            assert len(sections) > 0, "Should extract at least one section from large document"
            
            print(f"\nSuccessfully processed large document with {len(sections)} top-level sections")
        except Exception as e:
            pytest.fail(f"Failed to process large document: {str(e)}")

def test_table_in_sections():
    """Test handling of markdown tables within sections."""
    # Create markdown with tables
    table_content = """# Document with Tables

## Table Section

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

## Another Section

Content without table.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(table_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for the table
        table_found = False
        
        def check_sections_for_table(section_list):
            nonlocal table_found
            for section in section_list:
                if "Table Section" in section["title"]:
                    if "|" in section["content"] and "---" in section["content"]:
                        return True
                
                # Check subsections recursively
                if 'subsections' in section and section['subsections']:
                    if check_sections_for_table(section['subsections']):
                        return True
            return False
        
        table_found = check_sections_for_table(sections)
        
        assert table_found, "Should preserve tables in section content"

def test_links_in_sections(real_readme_content):
    """Test handling of markdown links within sections from real README."""
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(real_readme_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for links
        link_found = False
        
        def check_for_links(section_list):
            nonlocal link_found
            for section in section_list:
                # Check for markdown links [text](url)
                if "[" in section["content"] and "](" in section["content"]:
                    link_found = True
                    link_start = section["content"].find("[")
                    link_middle = section["content"].find("](", link_start)
                    link_end = section["content"].find(")", link_middle)
                    
                    if link_end > link_middle > link_start:
                        link_text = section["content"][link_start+1:link_middle]
                        link_url = section["content"][link_middle+2:link_end]
                        print(f"\nFound link in section '{section['title']}': {link_text} -> {link_url}")
                
                if "subsections" in section and section["subsections"]:
                    check_for_links(section["subsections"])
        
        check_for_links(sections)
        
        # If we didn't find any links, try another document
        if not link_found:
            print("\nNo links found in the README, trying another document...")
            typescript_readme = fetch_real_markdown('typescript_overview')
            
            with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f2:
                f2.write(typescript_readme)
                f2.flush()
                
                sections = extract_hierarchical_sections(f2.name)
                check_for_links(sections)
        
        assert link_found, "Should find links in real documentation"

def test_image_references_in_sections():
    """Test handling of markdown image references within sections."""
    # Create markdown with image references
    image_content = """# Document with Images

## Image Section

Here's an image:

![Alt text](https://example.com/image.png "Image Title")

## Another Image

Another image with just alt text:

![Just alt text](path/to/image.jpg)
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(image_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for image references
        image_found = False
        
        def check_sections_for_images(section_list):
            for section in section_list:
                if "![" in section["content"] and "](" in section["content"]:
                    return True
                
                # Check subsections recursively
                if 'subsections' in section and section['subsections']:
                    if check_sections_for_images(section['subsections']):
                        return True
            return False
        
        image_found = check_sections_for_images(sections)
        
        assert image_found, "Should preserve image references in section content"

def test_section_with_admonitions():
    """Test handling of admonitions/callouts in markdown."""
    # Create markdown with admonitions
    admonition_content = """# Document with Admonitions

## Note Section

> **Note**
> This is a note.

## Warning Section

> **Warning**
> This is a warning.

## Regular Section

Just regular content.
"""
    
    with tempfile.NamedTemporaryFile(suffix='.md', mode='w+') as f:
        f.write(admonition_content)
        f.flush()
        
        # Extract sections
        sections = extract_hierarchical_sections(f.name)
        
        # Look for admonitions
        note_found = False
        warning_found = False
        
        def check_sections_for_admonitions(section_list):
            nonlocal note_found, warning_found
            for section in section_list:
                if "Note Section" in section["title"]:
                    if "**Note**" in section["content"]:
                        note_found = True
                elif "Warning Section" in section["title"]:
                    if "**Warning**" in section["content"]:
                        warning_found = True
                
                # Check subsections recursively
                if 'subsections' in section and section['subsections']:
                    check_sections_for_admonitions(section['subsections'])
        
        check_sections_for_admonitions(sections)
        
        assert note_found, "Should preserve note admonition in section content"
        assert warning_found, "Should preserve warning admonition in section content" 