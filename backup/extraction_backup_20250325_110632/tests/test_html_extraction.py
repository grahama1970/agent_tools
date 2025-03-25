#!/usr/bin/env python3
"""
Test HTML Extraction Module for DuaLipa.

This module tests the HTML extraction capabilities of the DuaLipa extraction module.
It validates that HTML documentation can be properly extracted with the correct
hierarchical structure and parent-child relationships.

Tests cover:
1. HTML cleaning and preprocessing
2. Section extraction from HTML
3. Special element detection (code blocks, tables, images)
4. Conversion to DuaLipa block format
5. Hierarchical structure validation
"""

import os
import json
import pytest
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional

from agent_tools.dualipa.fetch_docs_integration import (
    convert_to_dualipa_format,
    detect_special_elements
)


class TestHTMLExtraction(unittest.TestCase):
    """Test HTML extraction capabilities."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a sample HTML file
        self.html_file = self.test_dir / "test.html"
        with open(self.html_file, "w", encoding="utf-8") as f:
            f.write(self.get_sample_html())
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def get_sample_html(self) -> str:
        """Return a sample HTML document for testing."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Test Documentation</title>
</head>
<body>
    <div class="content">
        <h1>Documentation Title</h1>
        <p>This is a test documentation page.</p>
        
        <h2>Section 1</h2>
        <p>Content of section 1.</p>
        <pre><code class="language-python">
def hello():
    print("Hello, World!")
        </code></pre>
        
        <h3>Subsection 1.1</h3>
        <p>Content of subsection 1.1.</p>
        <table>
            <tr><th>Header 1</th><th>Header 2</th></tr>
            <tr><td>Value 1</td><td>Value 2</td></tr>
            <tr><td>Value 3</td><td>Value 4</td></tr>
        </table>
        
        <h2>Section 2</h2>
        <p>Content of section 2.</p>
        <img src="example.png" alt="Example Image">
    </div>
</body>
</html>"""
    
    def test_special_element_detection(self):
        """Test detection of special elements in HTML."""
        html_content = self.get_sample_html()
        elements = detect_special_elements(html_content)
        
        # Check that code blocks were detected
        self.assertIn("code_blocks", elements)
        self.assertGreater(len(elements["code_blocks"]), 0)
        self.assertEqual(elements["code_blocks"][0]["language"], "python")
        
        # Check that tables were detected
        self.assertIn("tables", elements)
        self.assertGreater(len(elements["tables"]), 0)
        self.assertEqual(len(elements["tables"][0]["headers"]), 2)
        self.assertEqual(len(elements["tables"][0]["rows"]), 2)
        
        # Check that images were detected
        self.assertIn("images", elements)
        self.assertGreater(len(elements["images"]), 0)
        self.assertEqual(elements["images"][0]["alt"], "Example Image")
    
    def test_conversion_to_dualipa_format(self):
        """Test conversion of HTML to DuaLipa format."""
        # Create a minimal processed docs structure
        processed_docs = {
            "https://example.com": [
                {
                    "file": str(self.html_file),
                    "relative_path": "test.html",
                    "sections": [
                        {
                            "header": "Documentation Title",
                            "content": "<h1>Documentation Title</h1><p>This is a test documentation page.</p>",
                            "level": 1,
                            "token_count": 20
                        },
                        {
                            "header": "Section 1",
                            "content": (
                                "<h2>Section 1</h2><p>Content of section 1.</p>"
                                '<pre><code class="language-python">\ndef hello():\n    print("Hello, World!")\n        </code></pre>'
                            ),
                            "level": 2,
                            "token_count": 30
                        },
                        {
                            "header": "Subsection 1.1",
                            "content": (
                                "<h3>Subsection 1.1</h3><p>Content of subsection 1.1.</p>"
                                "<table><tr><th>Header 1</th><th>Header 2</th></tr>"
                                "<tr><td>Value 1</td><td>Value 2</td></tr>"
                                "<tr><td>Value 3</td><td>Value 4</td></tr></table>"
                            ),
                            "level": 3,
                            "token_count": 40
                        },
                        {
                            "header": "Section 2",
                            "content": (
                                "<h2>Section 2</h2><p>Content of section 2.</p>"
                                '<img src="example.png" alt="Example Image">'
                            ),
                            "level": 2,
                            "token_count": 25
                        }
                    ],
                    "doc_type": "test"
                }
            ]
        }
        
        # Convert to DuaLipa format
        blocks = convert_to_dualipa_format(processed_docs, self.test_dir)
        
        # Verify blocks were created
        self.assertGreater(len(blocks), 0)
        
        # Check block types and hierarchy
        block_types = [block["type"] for block in blocks]
        self.assertIn("documentation", block_types)
        self.assertIn("doc_page", block_types)
        self.assertIn("doc_section", block_types)
        
        # Find documentation block
        doc_block = next((b for b in blocks if b["type"] == "documentation"), None)
        self.assertIsNotNone(doc_block)
        self.assertIn("child_uuids", doc_block)
        self.assertGreater(len(doc_block["child_uuids"]), 0)
        
        # Find page block
        page_block = next((b for b in blocks if b["type"] == "doc_page"), None)
        self.assertIsNotNone(page_block)
        self.assertEqual(page_block["parent_uuid"], doc_block["uuid"])
        self.assertIn("child_uuids", page_block)
        self.assertGreater(len(page_block["child_uuids"]), 0)
        
        # Find section blocks
        section_blocks = [b for b in blocks if b["type"] == "doc_section"]
        self.assertGreater(len(section_blocks), 0)
        
        # Check section hierarchy
        level1_section = next((s for s in section_blocks if s["metadata"]["header_level"] == 1), None)
        self.assertIsNotNone(level1_section)
        self.assertEqual(level1_section["parent_uuid"], page_block["uuid"])
        
        level2_sections = [s for s in section_blocks if s["metadata"]["header_level"] == 2]
        self.assertGreater(len(level2_sections), 0)
        
        # At least one level 2 section should have the level 1 section as parent
        level2_with_level1_parent = any(s["parent_uuid"] == level1_section["uuid"] for s in level2_sections)
        self.assertTrue(level2_with_level1_parent)
        
        # Check special elements
        code_blocks = [b for b in blocks if b["type"] == "code_block"]
        self.assertGreater(len(code_blocks), 0)
        self.assertEqual(code_blocks[0]["language"], "python")
        
        tables = [b for b in blocks if b["type"] == "table"]
        self.assertGreater(len(tables), 0)
        
        images = [b for b in blocks if b["type"] == "image"]
        self.assertGreater(len(images), 0)
    
    def test_bidirectional_references(self):
        """Test that parent-child relationships are bidirectional."""
        # Create a minimal processed docs structure (same as previous test)
        processed_docs = {
            "https://example.com": [
                {
                    "file": str(self.html_file),
                    "relative_path": "test.html",
                    "sections": [
                        {
                            "header": "Documentation Title",
                            "content": "<h1>Documentation Title</h1><p>This is a test documentation page.</p>",
                            "level": 1,
                            "token_count": 20
                        },
                        {
                            "header": "Section 1",
                            "content": "<h2>Section 1</h2><p>Content of section 1.</p>",
                            "level": 2,
                            "token_count": 30
                        }
                    ],
                    "doc_type": "test"
                }
            ]
        }
        
        # Convert to DuaLipa format
        blocks = convert_to_dualipa_format(processed_docs, self.test_dir)
        
        # Check bidirectional references
        for block in blocks:
            # Skip blocks without children
            if "child_uuids" not in block or not block["child_uuids"]:
                continue
                
            # Check each child references this block as parent
            for child_uuid in block["child_uuids"]:
                child_block = next((b for b in blocks if b["uuid"] == child_uuid), None)
                self.assertIsNotNone(child_block, f"Child block {child_uuid} not found")
                self.assertEqual(child_block["parent_uuid"], block["uuid"], 
                               f"Child {child_uuid} has wrong parent_uuid")
        
        # Check each block with a parent is in the parent's child_uuids
        for block in blocks:
            # Skip blocks without parents
            if "parent_uuid" not in block:
                continue
                
            parent_block = next((b for b in blocks if b["uuid"] == block["parent_uuid"]), None)
            self.assertIsNotNone(parent_block, f"Parent block {block['parent_uuid']} not found")
            self.assertIn(block["uuid"], parent_block["child_uuids"], 
                       f"Block {block['uuid']} not in parent's child_uuids")


def test_section_hierarchy_extraction():
    """Test extraction of section hierarchy from HTML."""
    # Create a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        html_file = test_dir / "test.html"
        
        # Create a complex HTML file with nested sections
        with open(html_file, "w", encoding="utf-8") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Nested Sections Test</title>
</head>
<body>
    <div class="content">
        <h1>Main Title</h1>
        <p>Main content.</p>
        
        <h2>Section 1</h2>
        <p>Section 1 content.</p>
        
        <h3>Section 1.1</h3>
        <p>Section 1.1 content.</p>
        
        <h4>Section 1.1.1</h4>
        <p>Section 1.1.1 content.</p>
        
        <h3>Section 1.2</h3>
        <p>Section 1.2 content.</p>
        
        <h2>Section 2</h2>
        <p>Section 2 content.</p>
        
        <h3>Section 2.1</h3>
        <p>Section 2.1 content.</p>
    </div>
</body>
</html>""")
        
        # Create processed docs manually (normally this would come from extract_sections_from_html)
        processed_docs = {
            "https://example.com": [
                {
                    "file": str(html_file),
                    "relative_path": "test.html",
                    "sections": [
                        {"header": "Main Title", "content": "...", "level": 1, "token_count": 10},
                        {"header": "Section 1", "content": "...", "level": 2, "token_count": 10},
                        {"header": "Section 1.1", "content": "...", "level": 3, "token_count": 10},
                        {"header": "Section 1.1.1", "content": "...", "level": 4, "token_count": 10},
                        {"header": "Section 1.2", "content": "...", "level": 3, "token_count": 10},
                        {"header": "Section 2", "content": "...", "level": 2, "token_count": 10},
                        {"header": "Section 2.1", "content": "...", "level": 3, "token_count": 10},
                    ],
                    "doc_type": "test"
                }
            ]
        }
        
        # Convert to DuaLipa format
        from agent_tools.dualipa.fetch_docs_integration import convert_to_dualipa_format
        blocks = convert_to_dualipa_format(processed_docs, test_dir)
        
        # Check that section hierarchy is correctly reflected in the blocks
        section_blocks = [b for b in blocks if b["type"] == "doc_section"]
        
        # Find blocks by header
        main_title = next((b for b in section_blocks if b["name"] == "Main Title"), None)
        section1 = next((b for b in section_blocks if b["name"] == "Section 1"), None)
        section1_1 = next((b for b in section_blocks if b["name"] == "Section 1.1"), None)
        section1_1_1 = next((b for b in section_blocks if b["name"] == "Section 1.1.1"), None)
        section1_2 = next((b for b in section_blocks if b["name"] == "Section 1.2"), None)
        section2 = next((b for b in section_blocks if b["name"] == "Section 2"), None)
        section2_1 = next((b for b in section_blocks if b["name"] == "Section 2.1"), None)
        
        # Verify all sections were found
        assert main_title is not None
        assert section1 is not None
        assert section1_1 is not None
        assert section1_1_1 is not None
        assert section1_2 is not None
        assert section2 is not None
        assert section2_1 is not None
        
        # Check parent-child relationships
        # Main Title → Section 1, Section 2
        assert section1["parent_uuid"] == main_title["uuid"]
        assert section2["parent_uuid"] == main_title["uuid"]
        
        # Section 1 → Section 1.1, Section 1.2
        assert section1_1["parent_uuid"] == section1["uuid"]
        assert section1_2["parent_uuid"] == section1["uuid"]
        
        # Section 1.1 → Section 1.1.1
        assert section1_1_1["parent_uuid"] == section1_1["uuid"]
        
        # Section 2 → Section 2.1
        assert section2_1["parent_uuid"] == section2["uuid"]
        
        # Check section hierarchy in metadata
        assert section1["metadata"]["section_hierarchy"] == ["Main Title", "Section 1"]
        assert section1_1["metadata"]["section_hierarchy"] == ["Main Title", "Section 1", "Section 1.1"]
        assert section1_1_1["metadata"]["section_hierarchy"] == ["Main Title", "Section 1", "Section 1.1", "Section 1.1.1"]
        assert section2_1["metadata"]["section_hierarchy"] == ["Main Title", "Section 2", "Section 2.1"]


if __name__ == "__main__":
    unittest.main()