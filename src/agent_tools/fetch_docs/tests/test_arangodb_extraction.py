#!/usr/bin/env python3
"""
test_arangodb_extraction.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- json: https://docs.python.org/3/library/json.html

This script performs a blind test of ArangoDB documentation extraction.
It tests the fetch_docs extraction pipeline against actual ArangoDB documentation,
verifying that the output has the expected structure and content.

The test uses the ArangoDB AQL page as a reference point, as this page has
a complex structure with code blocks, tables, and sections.

Input: None (downloads and processes actual ArangoDB documentation)
Output: Test results showing whether extraction produces the expected structure

Example usage:
    python test_arangodb_extraction.py
"""

import os
import sys
import json
from pathlib import Path
import tempfile
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_arangodb_extraction')

# Test constants
ARANGODB_AQL_URL = "https://docs.arangodb.com/stable/aql/"
EXPECTED_SECTIONS_MIN = 5  # Minimum number of sections on the AQL page
EXPECTED_CODE_BLOCKS_MIN = 3  # Minimum number of code blocks
EXPECTED_TABLES_MIN = 1  # Minimum number of tables

def test_arangodb_extraction():
    """
    Test extraction of ArangoDB AQL documentation.
    
    This function:
    1. Downloads the ArangoDB AQL documentation page
    2. Processes it with fetch_docs
    3. Verifies the structure and content of the extracted blocks
    4. Checks for expected sections, code blocks, and tables
    
    A successful test validates that the extraction pipeline correctly handles
    real-world documentation with complex structure.
    """
    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Import required modules from fetch_docs
        try:
            from agent_tools.fetch_docs.processor import process_documentation
            from agent_tools.fetch_docs.download_site import download_site
        except ImportError as e:
            logger.error(f"Error importing fetch_docs modules: {e}")
            return False
        
        # Download ArangoDB AQL documentation
        logger.info(f"Downloading ArangoDB AQL documentation from {ARANGODB_AQL_URL}")
        download_dir = temp_dir / "arangodb_aql"
        try:
            download_site(ARANGODB_AQL_URL, str(download_dir), recursive=False)
            logger.info("Download completed")
        except Exception as e:
            logger.error(f"Error downloading documentation: {e}")
            return False
        
        # Process the documentation
        logger.info("Processing documentation")
        processed_docs = process_documentation([ARANGODB_AQL_URL], temp_dir)
        
        # Verify the structure of processed docs
        logger.info("Verifying processed documentation structure")
        if not processed_docs or ARANGODB_AQL_URL not in processed_docs:
            logger.error("ArangoDB AQL URL not found in processed docs")
            return False
        
        # Get the page data for the AQL URL
        site_data = processed_docs[ARANGODB_AQL_URL]
        if not site_data:
            logger.error("No site data found for ArangoDB AQL URL")
            return False
        
        # Get the first page data (main page)
        page_data = site_data[0]
        
        # Verify page data structure
        sections = page_data.get("sections", [])
        if len(sections) < EXPECTED_SECTIONS_MIN:
            logger.error(f"Too few sections: expected at least {EXPECTED_SECTIONS_MIN}, got {len(sections)}")
            return False
        
        # Check section content
        logger.info(f"Found {len(sections)} sections")
        has_intro_section = False
        for section in sections:
            if section.get("header", "").lower() in ["introduction", "aql", "about aql"]:
                has_intro_section = True
                logger.info(f"Found introduction section: {section.get('header')}")
                break
        
        if not has_intro_section:
            logger.warning("Could not find introduction section")
        
        # Convert to DuaLipa-compatible blocks
        try:
            from agent_tools.dualipa.extraction.docs_integration import convert_page_to_blocks
        except ImportError:
            # If dualipa is not available, we need to define the function here
            logger.warning("DuaLipa integration not available, defining conversion function locally")
            
            def convert_page_to_blocks(page_data, url, repo_path):
                """Convert a documentation page to DuaLipa-compatible blocks."""
                import uuid
                blocks = []
                
                # Create a parent block for the page
                page_uuid = str(uuid.uuid4())
                page_title = page_data.get("title", "Untitled Page")
                
                # Sanitize page title for ID
                safe_title = "".join(c if c.isalnum() or c == "_" else "_" for c in page_title.lower())
                
                # Create page block
                page_block = {
                    "uuid": page_uuid,
                    "id": f"doc_{safe_title}",
                    "type": "documentation",
                    "name": page_title,
                    "content": page_data.get("summary", f"Documentation from {url}"),
                    "file_path": str(repo_path),
                    "line_start": 1,
                    "line_end": 1,
                    "metadata": {
                        "language": "html",
                        "file": page_data.get("file", ""),
                        "url": url,
                        "doc_type": page_data.get("doc_type", "documentation"),
                    },
                    "child_uuids": []
                }
                
                blocks.append(page_block)
                
                # Process sections
                section_hierarchy = {}  # level -> last section at that level
                
                for section in page_data.get("sections", []):
                    section_uuid = str(uuid.uuid4())
                    section_title = section.get("header", "Untitled Section")
                    section_level = section.get("level", 1)
                    section_content = section.get("content", "")
                    
                    # Sanitize section title for ID
                    safe_section_title = "".join(c if c.isalnum() or c == "_" else "_" for c in section_title.lower())
                    
                    # Find parent section
                    parent_uuid = page_uuid
                    for level in range(section_level - 1, 0, -1):
                        if level in section_hierarchy:
                            parent_uuid = section_hierarchy[level]
                            break
                    
                    # Create section block
                    section_block = {
                        "uuid": section_uuid,
                        "id": f"doc_{safe_title}_{safe_section_title}",
                        "type": "doc_section",
                        "name": section_title,
                        "content": section_content,
                        "file_path": page_data.get("file", ""),
                        "line_start": 1,
                        "line_end": 1,
                        "metadata": {
                            "language": "html",
                            "file": page_data.get("file", ""),
                            "url": url,
                            "doc_type": page_data.get("doc_type", "documentation"),
                            "heading_level": section_level,
                            "token_count": section.get("token_count", len(section_content.split())),
                        },
                        "child_uuids": []
                    }
                    
                    blocks.append(section_block)
                    
                    # Add to parent's child UUIDs
                    for block in blocks:
                        if block["uuid"] == parent_uuid:
                            block["child_uuids"].append(section_uuid)
                            break
                    
                    # Update hierarchy
                    section_hierarchy[section_level] = section_uuid
                    
                    # Clear any higher levels since they're no longer relevant
                    higher_levels = [l for l in section_hierarchy if l > section_level]
                    for l in higher_levels:
                        if l in section_hierarchy:
                            del section_hierarchy[l]
                
                return blocks
        
        # Convert the page data to blocks
        logger.info("Converting to DuaLipa-compatible blocks")
        blocks = convert_page_to_blocks(page_data, ARANGODB_AQL_URL, temp_dir)
        
        # Verify blocks structure
        if not blocks:
            logger.error("No blocks created")
            return False
        
        # Check if we have a parent documentation block
        doc_blocks = [b for b in blocks if b["type"] == "documentation"]
        if not doc_blocks:
            logger.error("No documentation blocks found")
            return False
        
        # Check if we have section blocks
        section_blocks = [b for b in blocks if b["type"] == "doc_section"]
        if len(section_blocks) < EXPECTED_SECTIONS_MIN:
            logger.error(f"Too few section blocks: expected at least {EXPECTED_SECTIONS_MIN}, got {len(section_blocks)}")
            return False
        
        # Check if blocks have required fields
        missing_fields = False
        required_fields = ["uuid", "id", "type", "name", "content", "metadata"]
        for block in blocks:
            for field in required_fields:
                if field not in block:
                    logger.error(f"Block missing required field: {field}")
                    missing_fields = True
        
        if missing_fields:
            return False
        
        # Check parent-child relationships
        for block in blocks:
            for child_uuid in block.get("child_uuids", []):
                child_block = next((b for b in blocks if b["uuid"] == child_uuid), None)
                if not child_block:
                    logger.error(f"Child block with UUID {child_uuid} not found")
                    return False
        
        # Save blocks to a JSON file for inspection
        output_file = temp_dir / "arangodb_aql_blocks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)
        logger.info(f"Saved blocks to {output_file}")
        
        # Check for code blocks
        code_blocks = [b for b in blocks if b.get("type") == "code_block"]
        if len(code_blocks) < EXPECTED_CODE_BLOCKS_MIN:
            logger.error(f"Too few code blocks: expected at least {EXPECTED_CODE_BLOCKS_MIN}, got {len(code_blocks)}")
            return False
        
        logger.info(f"Found {len(code_blocks)} code blocks")
        
        # Look for table blocks
        table_blocks = [b for b in blocks if b.get("type") == "table"]
        if len(table_blocks) < EXPECTED_TABLES_MIN:
            logger.error(f"Too few table blocks: expected at least {EXPECTED_TABLES_MIN}, got {len(table_blocks)}")
            return False
        
        logger.info(f"Found {len(table_blocks)} table blocks")
        
        # Test passed
        logger.info("All tests passed")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    finally:
        # Clean up
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("Running ArangoDB extraction test...")
    success = test_arangodb_extraction()
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)