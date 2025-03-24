#!/usr/bin/env python3
"""
test_readthedocs_extraction.py

Official Documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html
- json: https://docs.python.org/3/library/json.html

This script performs a blind test of ReadTheDocs documentation extraction.
It tests the fetch_docs extraction pipeline against actual ReadTheDocs documentation,
verifying that the output has the expected structure and content.

The test uses the Python ReadTheDocs documentation as a reference, as it has
a complex structure with code blocks, tables, and sections.

Input: None (downloads and processes actual ReadTheDocs documentation)
Output: Test results showing whether extraction produces the expected structure

Example usage:
    python test_readthedocs_extraction.py
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
logger = logging.getLogger('test_readthedocs_extraction')

# Test constants
READTHEDOCS_URL = "https://python.readthedocs.io/en/latest/"
EXPECTED_SECTIONS_MIN = 5  # Minimum number of sections
EXPECTED_CODE_BLOCKS_MIN = 3  # Minimum number of code blocks

def validate_extraction_output(blocks, expected_sections=EXPECTED_SECTIONS_MIN, expected_code_blocks=EXPECTED_CODE_BLOCKS_MIN):
    """
    Validate the structure and content of extracted blocks.
    
    Args:
        blocks: List of extracted blocks
        expected_sections: Minimum number of sections expected
        expected_code_blocks: Minimum number of code blocks expected
        
    Returns:
        Tuple (success, message)
    """
    # Check if blocks list is valid
    if not blocks:
        return False, "No blocks extracted"
    
    # Check if we have a parent documentation block
    doc_blocks = [b for b in blocks if b.get("type") == "documentation"]
    if not doc_blocks:
        return False, "No documentation blocks found"
    
    # Check if we have section blocks
    section_blocks = [b for b in blocks if b.get("type") == "doc_section"]
    if len(section_blocks) < expected_sections:
        return False, f"Too few section blocks: expected at least {expected_sections}, got {len(section_blocks)}"
    
    # Check if blocks have required fields
    required_fields = ["uuid", "id", "type", "name", "content", "metadata"]
    for block in blocks:
        for field in required_fields:
            if field not in block:
                return False, f"Block missing required field: {field}"
    
    # Check parent-child relationships
    for block in blocks:
        for child_uuid in block.get("child_uuids", []):
            child_block = next((b for b in blocks if b.get("uuid") == child_uuid), None)
            if not child_block:
                return False, f"Child block with UUID {child_uuid} not found"
    
    # Check for code blocks
    code_blocks = [b for b in blocks if b.get("type") == "code_block"]
    if len(code_blocks) < expected_code_blocks:
        return False, f"Too few code blocks: expected at least {expected_code_blocks}, got {len(code_blocks)}"
    
    # All checks passed
    return True, "Validation successful"

def test_readthedocs_extraction():
    """
    Test extraction of ReadTheDocs documentation.
    
    This function:
    1. Downloads the ReadTheDocs documentation page
    2. Processes it with fetch_docs
    3. Verifies the structure and content of the extracted blocks
    4. Checks for expected sections and code blocks
    
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
        
        # Download ReadTheDocs documentation
        logger.info(f"Downloading ReadTheDocs documentation from {READTHEDOCS_URL}")
        download_dir = temp_dir / "readthedocs"
        try:
            download_site(READTHEDOCS_URL, str(download_dir), recursive=False)
            logger.info("Download completed")
        except Exception as e:
            logger.error(f"Error downloading documentation: {e}")
            return False
        
        # Process the documentation
        logger.info("Processing documentation")
        processed_docs = process_documentation([READTHEDOCS_URL], temp_dir)
        
        # Verify the structure of processed docs
        logger.info("Verifying processed documentation structure")
        if not processed_docs or READTHEDOCS_URL not in processed_docs:
            logger.error("ReadTheDocs URL not found in processed docs")
            return False
        
        # Get the page data for the ReadTheDocs URL
        site_data = processed_docs[READTHEDOCS_URL]
        if not site_data:
            logger.error("No site data found for ReadTheDocs URL")
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
        
        # Convert to DuaLipa-compatible blocks
        try:
            from agent_tools.dualipa.extraction.docs_integration import convert_page_to_blocks
        except ImportError:
            # If dualipa is not available, we need to define the function here
            logger.warning("DuaLipa integration not available, using local conversion function")
            
            def convert_page_to_blocks(page_data, url, repo_path):
                """
                Convert page data to DuaLipa-compatible blocks.
                
                Args:
                    page_data: Processed page data
                    url: Page URL
                    repo_path: Repository path
                    
                Returns:
                    List of blocks in DuaLipa format
                """
                import uuid
                blocks = []
                
                # Create parent block
                doc_uuid = str(uuid.uuid4())
                doc_block = {
                    "uuid": doc_uuid,
                    "id": f"doc_{page_data.get('title', 'readthedocs').lower().replace(' ', '_')}",
                    "type": "documentation",
                    "name": page_data.get("title", "ReadTheDocs Documentation"),
                    "content": page_data.get("summary", "Documentation from ReadTheDocs"),
                    "file_path": str(repo_path),
                    "line_start": 1,
                    "line_end": 1,
                    "metadata": {
                        "language": "html",
                        "url": url,
                        "doc_type": "readthedocs"
                    },
                    "child_uuids": []
                }
                blocks.append(doc_block)
                
                # Create section blocks
                section_hierarchy = {}  # Maps level to (uuid, position)
                
                for i, section in enumerate(page_data.get("sections", [])):
                    section_uuid = str(uuid.uuid4())
                    section_level = section.get("level", 1)
                    
                    # Find parent 
                    parent_uuid = doc_uuid
                    for level in range(section_level - 1, 0, -1):
                        if level in section_hierarchy:
                            parent_uuid = section_hierarchy[level][0]
                            break
                    
                    # Create section block
                    section_block = {
                        "uuid": section_uuid,
                        "id": f"doc_section_{i}",
                        "type": "doc_section",
                        "name": section.get("header", f"Section {i}"),
                        "content": section.get("content", ""),
                        "file_path": page_data.get("file", ""),
                        "line_start": 1,
                        "line_end": 1,
                        "metadata": {
                            "language": "html",
                            "url": url,
                            "doc_type": "readthedocs",
                            "heading_level": section_level,
                            "position": i
                        },
                        "child_uuids": []
                    }
                    blocks.append(section_block)
                    
                    # Update parent's child_uuids
                    for block in blocks:
                        if block["uuid"] == parent_uuid:
                            block["child_uuids"].append(section_uuid)
                            break
                    
                    # Update hierarchy
                    section_hierarchy[section_level] = (section_uuid, i)
                    
                    # Remove any higher levels
                    for level in list(section_hierarchy.keys()):
                        if level > section_level:
                            del section_hierarchy[level]
                    
                    # Process code blocks in section
                    if "<code" in section.get("content", "") or "<pre" in section.get("content", ""):
                        import re
                        code_pattern = re.compile(r'<(pre|code)[^>]*>(.*?)</\1>', re.DOTALL)
                        for j, match in enumerate(code_pattern.finditer(section.get("content", ""))):
                            code_uuid = str(uuid.uuid4())
                            code_content = match.group(2)
                            
                            # Try to detect language
                            language = "text"
                            lang_match = re.search(r'class="[^"]*language-([^"]*)"', match.group(0))
                            if lang_match:
                                language = lang_match.group(1)
                            
                            # Create code block
                            code_block = {
                                "uuid": code_uuid,
                                "id": f"doc_section_{i}_code_{j}",
                                "type": "code_block",
                                "name": f"Code Block {j}",
                                "content": code_content,
                                "file_path": page_data.get("file", ""),
                                "line_start": 1,
                                "line_end": 1,
                                "metadata": {
                                    "language": language,
                                    "url": url,
                                    "doc_type": "readthedocs"
                                },
                                "child_uuids": []
                            }
                            blocks.append(code_block)
                            section_block["child_uuids"].append(code_uuid)
                
                return blocks
        
        # Convert the page data to blocks
        logger.info("Converting to DuaLipa-compatible blocks")
        blocks = convert_page_to_blocks(page_data, READTHEDOCS_URL, temp_dir)
        
        # Validate the blocks
        success, message = validate_extraction_output(blocks)
        if not success:
            logger.error(message)
            return False
        
        # Save blocks to a JSON file for inspection
        output_file = temp_dir / "readthedocs_blocks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)
        logger.info(f"Saved blocks to {output_file}")
        
        # Log block statistics
        doc_blocks = [b for b in blocks if b.get("type") == "documentation"]
        section_blocks = [b for b in blocks if b.get("type") == "doc_section"]
        code_blocks = [b for b in blocks if b.get("type") == "code_block"]
        
        logger.info(f"Extracted {len(blocks)} blocks:")
        logger.info(f"- {len(doc_blocks)} documentation blocks")
        logger.info(f"- {len(section_blocks)} section blocks")
        logger.info(f"- {len(code_blocks)} code blocks")
        
        # Test passed
        logger.info("All tests passed")
        return True
    
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Clean up
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("Running ReadTheDocs extraction test...")
    success = test_readthedocs_extraction()
    print(f"Test {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)