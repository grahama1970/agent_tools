#!/usr/bin/env python3
"""
Test script for markdown extraction functionality.

This script tests the extraction of sections and elements from markdown files.
"""

import os
import sys
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("markdown_extraction_test")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions
from extraction_blocks import extract_all_blocks, extract_markdown_sections

class MarkdownExtractionTest:
    """Test the extraction of sections and elements from markdown files."""
    
    def __init__(self):
        """Initialize the test."""
        self.test_file = os.path.join(current_dir, "MARKDOWN_EXTRACTION.md")
        
    def test_section_extraction(self):
        """Test the extraction of sections from a markdown file."""
        logger.info(f"Testing section extraction from {self.test_file}")
        
        # Read the markdown file
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a dummy parent UUID
        parent_uuid = "test-parent-uuid"
        
        # Extract sections
        sections = extract_markdown_sections(content, self.test_file, parent_uuid)
        
        # Count sections by level
        section_levels = {}
        for section in sections:
            if section.get("type") == "section":
                level = section.get("metadata", {}).get("level", 0)
                section_levels[level] = section_levels.get(level, 0) + 1
        
        logger.info(f"Found sections by level: {section_levels}")
        
        # Count special elements
        element_types = {}
        for element in sections:
            element_type = element.get("type")
            if element_type not in ["file", "section"]:
                element_types[element_type] = element_types.get(element_type, 0) + 1
        
        logger.info(f"Found element types: {element_types}")
        
        # Simple validation
        success = True
        
        # We should have at least 5 sections (h1, h2, h3)
        total_sections = sum(section_levels.values())
        if total_sections < 5:
            logger.error(f"Expected at least 5 sections, found {total_sections}")
            success = False
        
        # We should have h1, h2, and h3 level headings
        if 1 not in section_levels or 2 not in section_levels or 3 not in section_levels:
            logger.error("Missing expected heading levels (should have h1, h2, and h3)")
            success = False
        
        # We should have code blocks, tables, and text blocks
        if "code_block" not in element_types:
            logger.error("No code blocks found")
            success = False
            
        if "table" not in element_types:
            logger.error("No tables found")
            success = False
            
        if "text" not in element_types:
            logger.error("No text blocks found")
            success = False
        
        if success:
            logger.info("✅ Section extraction test passed!")
        else:
            logger.error("❌ Section extraction test failed")
        
        return success
    
    def test_full_extraction(self):
        """Test the full extraction process on a markdown file."""
        logger.info(f"Testing full extraction from {self.test_file}")
        
        # Get the directory containing the test file
        test_dir = os.path.dirname(self.test_file)
        
        # Extract all blocks from the directory
        all_blocks = extract_all_blocks(Path(test_dir))
        
        # Filter blocks related to our test file
        test_file_blocks = [b for b in all_blocks if b.get("file_path") == self.test_file]
        
        # Count block types
        block_types = {}
        for block in test_file_blocks:
            block_type = block.get("type")
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        logger.info(f"Found block types: {block_types}")
        
        # Find the file block
        file_blocks = [b for b in test_file_blocks if b.get("type") == "file"]
        if not file_blocks:
            logger.error("No file block found")
            return False
            
        file_block = file_blocks[0]
        
        # Traverse the hierarchy to verify parent-child relationships
        child_uuids = file_block.get("child_uuids", [])
        if not child_uuids:
            logger.error("File block has no children")
            return False
            
        logger.info(f"File block has {len(child_uuids)} children")
        
        # Count children by type
        child_types = {}
        for child_uuid in child_uuids:
            child_blocks = [b for b in test_file_blocks if b.get("uuid") == child_uuid]
            if child_blocks:
                child_type = child_blocks[0].get("type")
                child_types[child_type] = child_types.get(child_type, 0) + 1
        
        logger.info(f"Child types: {child_types}")
        
        # Simple validation
        success = True
        
        # We should have a file block
        if "file" not in block_types or block_types["file"] != 1:
            logger.error("Expected 1 file block")
            success = False
        
        # We should have section blocks
        if "section" not in block_types or block_types["section"] < 5:
            logger.error("Expected at least 5 section blocks")
            success = False
        
        # We should have code blocks, tables, and text blocks
        if "code_block" not in block_types:
            logger.error("No code blocks found")
            success = False
            
        if "table" not in block_types:
            logger.error("No tables found")
            success = False
            
        if "text" not in block_types:
            logger.error("No text blocks found")
            success = False
        
        if success:
            logger.info("✅ Full extraction test passed!")
        else:
            logger.error("❌ Full extraction test failed")
        
        return success
        
    def run_test(self):
        """Run all markdown extraction tests."""
        success = True
        
        # Test section extraction
        success &= self.test_section_extraction()
        
        # Test full extraction
        success &= self.test_full_extraction()
        
        return success


def run_test():
    """Run the markdown extraction test."""
    test = MarkdownExtractionTest()
    if test.run_test():
        logger.info("✅ All markdown extraction tests passed!")
        return 0
    else:
        logger.error("❌ Some markdown extraction tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_test())