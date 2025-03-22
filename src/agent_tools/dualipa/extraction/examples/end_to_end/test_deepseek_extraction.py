#!/usr/bin/env python3
"""
Unit tests for the extraction of deepseek.md.

This script tests the ability of the markdown extraction module to correctly
extract sections, tables, code blocks, and images from the deepseek.md file
and verify that the output matches the expected format.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_deepseek_extraction")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions
from extraction_blocks import extract_all_blocks
from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output


class TestDeepseekExtraction(unittest.TestCase):
    """Test the extraction of deepseek.md file."""

    def setUp(self):
        """Set up the test case."""
        # Path to test repository
        self.repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/sglang")
        
        # Path to deepseek.md
        self.deepseek_path = self.repo_path / "docs" / "references" / "deepseek.md"
        
        # Path to expected output
        self.expected_output_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/samples/deepseek_markdown_extraction_example.json")
        
        # Verify file exists
        self.assertTrue(self.deepseek_path.exists(), f"Test file not found: {self.deepseek_path}")
        self.assertTrue(self.expected_output_path.exists(), f"Expected output file not found: {self.expected_output_path}")
        
        # Load expected output
        with open(self.expected_output_path, 'r', encoding='utf-8') as f:
            self.expected_output = json.load(f)

    def test_file_extraction(self):
        """Test that the deepseek.md file is properly extracted."""
        logger.info("Testing extraction of deepseek.md file")
        
        # Extract all blocks from the repository
        blocks = extract_all_blocks(self.repo_path)
        self.assertIsNotNone(blocks, "Failed to extract blocks from repository")
        
        # Find the deepseek.md file block
        deepseek_blocks = [b for b in blocks if b.get("type") == "file" and 
                          "deepseek.md" in b.get("file_path", "")]
        
        self.assertEqual(len(deepseek_blocks), 1, "Should find exactly one deepseek.md file")
        deepseek_block = deepseek_blocks[0]
        
        # Convert blocks to QA-compatible format
        qa_blocks = create_qa_compatible_blocks(blocks)
        
        # Create output
        output = create_qa_compatible_output(qa_blocks)
        
        # Check if output is a list (specific to deepseek format)
        self.assertIsInstance(output, list, "Output should be a list for deepseek.md format")
        
        # Save actual output for debugging
        output_file = Path(current_dir) / "test_output_deepseek.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved test output to {output_file}")
    
    def test_section_hierarchy(self):
        """Test that the section hierarchy is correctly extracted."""
        logger.info("Testing section hierarchy extraction")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Get all unique hierarchy depths
        hierarchies = [section.get("section_hierarchy_depth", []) for section in output]
        
        # Check that main "DeepSeek Usage" section exists
        self.assertTrue(
            any(h == ["DeepSeek Usage"] for h in hierarchies),
            "Main 'DeepSeek Usage' section not found"
        )
        
        # Check that "Launch DeepSeek V3 with SGLang" subsection exists
        self.assertTrue(
            any(h == ["DeepSeek Usage", "Launch DeepSeek V3 with SGLang"] for h in hierarchies),
            "Subsection 'Launch DeepSeek V3 with SGLang' not found"
        )
        
        # Check for deeper nesting
        self.assertTrue(
            any(len(h) >= 3 for h in hierarchies),
            "No deeply nested sections found"
        )
    
    def test_tables_extraction(self):
        """Test that tables are correctly extracted."""
        logger.info("Testing table extraction")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Count tables across all sections
        table_count = sum(len(section.get("tables", [])) for section in output)
        
        # The expected_output contains at least one table
        expected_table_count = sum(len(section.get("tables", [])) for section in self.expected_output)
        
        logger.info(f"Found {table_count} tables, expected at least {expected_table_count}")
        self.assertGreaterEqual(
            table_count, 
            expected_table_count,
            f"Expected at least {expected_table_count} tables, found {table_count}"
        )
        
        # Check table format
        has_valid_tables = False
        for section in output:
            for table in section.get("tables", []):
                if "content" in table and "headers" in table["content"] and "rows" in table["content"]:
                    has_valid_tables = True
        
        self.assertTrue(has_valid_tables, "No properly formatted tables found")
    
    def test_code_blocks_extraction(self):
        """Test that code blocks are correctly extracted."""
        logger.info("Testing code block extraction")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Count code blocks across all sections
        code_count = sum(len(section.get("code", [])) for section in output)
        
        # The expected_output contains at least one code block
        expected_code_count = sum(len(section.get("code", [])) for section in self.expected_output)
        
        logger.info(f"Found {code_count} code blocks, expected at least {expected_code_count}")
        self.assertGreaterEqual(
            code_count, 
            expected_code_count,
            f"Expected at least {expected_code_count} code blocks, found {code_count}"
        )
        
        # Check code block format
        has_valid_code_blocks = False
        for section in output:
            for code in section.get("code", []):
                if "language" in code and "content" in code:
                    has_valid_code_blocks = True
        
        self.assertTrue(has_valid_code_blocks, "No properly formatted code blocks found")
    
    def test_images_extraction(self):
        """Test that images are correctly extracted."""
        logger.info("Testing image extraction")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Count images across all sections
        image_count = sum(len(section.get("images", [])) for section in output)
        
        # The expected_output contains some images
        expected_image_count = sum(len(section.get("images", [])) for section in self.expected_output)
        
        logger.info(f"Found {image_count} images, expected at least {expected_image_count}")
        self.assertGreaterEqual(
            image_count, 
            expected_image_count,
            f"Expected at least {expected_image_count} images, found {image_count}"
        )
        
        # Check image format
        has_valid_images = False
        for section in output:
            for image in section.get("images", []):
                if "src" in image and "alt" in image:
                    has_valid_images = True
        
        self.assertTrue(has_valid_images or expected_image_count == 0, 
                      "No properly formatted images found")
    
    def test_content_completeness(self):
        """Test that the content is complete and correctly formatted."""
        logger.info("Testing content completeness")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Check that all sections have title and content
        for section in output:
            self.assertIn("title", section, "Section missing title")
            self.assertIn("content", section, "Section missing content")
            self.assertIn("section_hierarchy_depth", section, "Section missing hierarchy path")
    
    def test_format_matches_expected(self):
        """Test that the output format matches the expected example."""
        logger.info("Testing output format against expected")
        
        # Extract blocks and convert to output format
        blocks = extract_all_blocks(self.repo_path)
        qa_blocks = create_qa_compatible_blocks(blocks)
        output = create_qa_compatible_output(qa_blocks)
        
        # Check that the first section has the same structure as expected
        self.assertGreater(len(output), 0, "No sections found in output")
        first_section = output[0]
        
        # Check expected keys
        expected_keys = set(self.expected_output[0].keys())
        actual_keys = set(first_section.keys())
        
        self.assertEqual(
            expected_keys, 
            actual_keys,
            f"Output format does not match expected. \nExpected keys: {expected_keys}\nActual keys: {actual_keys}"
        )
        
        # Check table format if any section has tables
        for section in output:
            if section.get("tables"):
                table = section["tables"][0]
                self.assertIn("uuid", table, "Table missing UUID")
                self.assertIn("content", table, "Table missing content")
                
                table_content = table["content"]
                self.assertIn("headers", table_content, "Table missing headers")
                self.assertIn("rows", table_content, "Table missing rows")
                self.assertIsInstance(table_content["headers"], list, "Table headers should be a list")
                self.assertIsInstance(table_content["rows"], list, "Table rows should be a list")
        
        # Check code block format if any section has code blocks
        for section in output:
            if section.get("code"):
                code = section["code"][0]
                self.assertIn("uuid", code, "Code block missing UUID")
                self.assertIn("language", code, "Code block missing language")
                self.assertIn("content", code, "Code block missing content")


if __name__ == "__main__":
    unittest.main()