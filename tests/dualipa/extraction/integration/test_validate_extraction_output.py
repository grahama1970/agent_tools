#!/usr/bin/env python3
"""
Test the validate_extraction_output function specifically.
"""

import unittest
import json
from typing import Dict, List, Any

# Import the function we want to test
try:
    from agent_tools.dualipa.extraction.examples.end_to_end.validate_extraction_format import validate_extraction_output
    FUNCTION_AVAILABLE = True
except ImportError:
    FUNCTION_AVAILABLE = False


class TestValidateExtractionOutput(unittest.TestCase):
    """Test the validate_extraction_output function."""

    def setUp(self):
        """Set up test data."""
        # Skip all tests if function isn't available
        if not FUNCTION_AVAILABLE:
            self.skipTest("validate_extraction_output function not available")
            
        # Create sample blocks for testing
        self.blocks = self.create_sample_blocks_for_validation()
        
        # Create sample expected format
        self.expected_format = {
            "expected_structure": {
                "required_block_types": [
                    "documentation",
                    "doc_page", 
                    "doc_section"
                ],
                "expected_type_counts": {
                    "documentation": {"min": 1},
                    "doc_page": {"min": 1},
                    "doc_section": {"min": 1}
                },
                "required_relationships": [
                    {"parent_type": "documentation", "child_type": "doc_page"},
                    {"parent_type": "doc_page", "child_type": "doc_section"}
                ]
            }
        }
    
    def create_sample_blocks_for_validation(self) -> List[Dict[str, Any]]:
        """Create sample blocks for validation testing."""
        # Create UUIDs
        doc_uuid = "doc-uuid"
        page_uuid = "page-uuid"
        section1_uuid = "section1-uuid"
        section2_uuid = "section2-uuid"
        code_uuid = "code-uuid"
        table_uuid = "table-uuid"
        image_uuid = "image-uuid"
        
        return [
            # Documentation block
            {
                "uuid": doc_uuid,
                "id": "docs_readthedocs",
                "name": "Documentation: readthedocs",
                "type": "documentation",
                "language": "html",
                "content": "Documentation site: https://python.readthedocs.io/en/latest/",
                "file_path": "/test/dir",
                "source_url": "https://python.readthedocs.io/en/latest/",
                "child_uuids": [page_uuid],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "doc_type": "readthedocs"
                }
            },
            # Page block
            {
                "uuid": page_uuid,
                "id": "docs_readthedocs_index",
                "name": "index.html",
                "type": "doc_page",
                "language": "html",
                "content": "Documentation page",
                "file_path": "/test/dir/index.html",
                "parent_uuid": doc_uuid,
                "child_uuids": [section1_uuid, section2_uuid],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "relative_path": "index.html",
                    "doc_type": "readthedocs"
                }
            },
            # Section block 1
            {
                "uuid": section1_uuid,
                "id": "docs_readthedocs_index_section_0",
                "name": "Python Documentation",
                "type": "doc_section",
                "language": "html",
                "content": "<h1>Python Documentation</h1><p>This is the Python documentation.</p>",
                "file_path": "/test/dir/index.html",
                "parent_uuid": page_uuid,
                "child_uuids": [code_uuid, table_uuid],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "position": 0,
                    "doc_type": "readthedocs",
                    "header_level": 1,
                    "token_count": 10,
                    "section_hierarchy": ["Python Documentation"],
                    "has_code": True,
                    "has_tables": True,
                    "has_images": False
                }
            },
            # Section block 2
            {
                "uuid": section2_uuid,
                "id": "docs_readthedocs_index_section_1",
                "name": "Getting Started",
                "type": "doc_section",
                "language": "html",
                "content": "<h2>Getting Started</h2><p>Get started with Python.</p>",
                "file_path": "/test/dir/index.html",
                "parent_uuid": page_uuid,
                "child_uuids": [image_uuid],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "position": 1,
                    "doc_type": "readthedocs",
                    "header_level": 2,
                    "token_count": 8,
                    "section_hierarchy": ["Python Documentation", "Getting Started"],
                    "has_code": False,
                    "has_tables": False,
                    "has_images": True
                }
            },
            # Code block
            {
                "uuid": code_uuid,
                "id": "docs_readthedocs_index_section_0_code_0",
                "name": "Hello World Example",
                "type": "code_block",
                "language": "python",
                "content": "print('Hello, World!')",
                "file_path": "/test/dir/index.html",
                "parent_uuid": section1_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "python",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "position": 0,
                    "doc_type": "readthedocs",
                    "element_type": "code_block",
                    "is_embedded": True,
                    "section_hierarchy": ["Python Documentation"]
                }
            },
            # Table block
            {
                "uuid": table_uuid,
                "id": "docs_readthedocs_index_section_0_table_0",
                "name": "Python Versions",
                "type": "table",
                "language": "html",
                "content": [
                    ["Version", "Status"],
                    ["3.9", "Stable"]
                ],
                "file_path": "/test/dir/index.html",
                "parent_uuid": section1_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "position": 1,
                    "doc_type": "readthedocs",
                    "element_type": "table",
                    "is_embedded": True,
                    "section_hierarchy": ["Python Documentation"],
                    "headers": ["Version", "Status"],
                    "rows": [["3.9", "Stable"]]
                }
            },
            # Image block
            {
                "uuid": image_uuid,
                "id": "docs_readthedocs_index_section_1_image_0",
                "name": "Python Logo",
                "type": "image",
                "language": "html",
                "content": "![Python Logo](https://www.python.org/static/img/python-logo.png)",
                "file_path": "/test/dir/index.html",
                "parent_uuid": section2_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "html",
                    "source_url": "https://python.readthedocs.io/en/latest/",
                    "position": 0,
                    "doc_type": "readthedocs",
                    "element_type": "image",
                    "is_embedded": True,
                    "section_hierarchy": ["Python Documentation", "Getting Started"],
                    "image_url": "https://www.python.org/static/img/python-logo.png",
                    "alt_text": "Python Logo"
                }
            }
        ]
        
    def test_basic_validation(self):
        """Test basic validation functionality."""
        # Call the function
        results = validate_extraction_output(self.blocks, self.expected_format)
        
        # Check results
        self.assertTrue(results["valid"], f"Validation failed: {results.get('errors', [])}")
        self.assertIn("stats", results, "Results should include stats")
        self.assertEqual(results["stats"]["total_blocks"], len(self.blocks), 
                        "Total blocks count incorrect")
        
    def test_missing_required_type(self):
        """Test validation when required block type is missing."""
        # Create a modified blocks list with missing type
        blocks_without_documentation = [b for b in self.blocks if b["type"] != "documentation"]
        
        # Call the function
        results = validate_extraction_output(blocks_without_documentation, self.expected_format)
        
        # Check results
        self.assertFalse(results["valid"], "Validation should fail with missing required block type")
        self.assertIn("errors", results, "Results should include errors")
        self.assertTrue(any("Missing required block type" in err for err in results["errors"]), 
                       "Error should mention missing block type")
        
    def test_invalid_relationship(self):
        """Test validation when required relationship is broken."""
        # Create a modified blocks list with broken relationship
        blocks_broken_relationship = self.blocks.copy()
        
        # Break the relationship by removing child_uuids from the documentation block
        for i, block in enumerate(blocks_broken_relationship):
            if block["type"] == "documentation":
                blocks_broken_relationship[i] = {**block, "child_uuids": []}
        
        # Call the function
        results = validate_extraction_output(blocks_broken_relationship, self.expected_format)
        
        # Check results
        self.assertFalse(results["valid"], "Validation should fail with broken relationship")
        
    def test_invalid_block(self):
        """Test validation with an invalid block."""
        # Create a modified blocks list with an invalid block
        blocks_with_invalid = self.blocks.copy()
        
        # Add an invalid block missing required fields
        blocks_with_invalid.append({
            "type": "doc_section",
            # Missing uuid, content
        })
        
        # Call the function
        results = validate_extraction_output(blocks_with_invalid, self.expected_format)
        
        # Check results
        self.assertFalse(results["valid"], "Validation should fail with invalid block")


if __name__ == "__main__":
    unittest.main()