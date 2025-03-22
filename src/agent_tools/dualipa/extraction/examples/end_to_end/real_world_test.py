#!/usr/bin/env python3
"""
Real-world End-to-End Extraction Test for DuaLipa.

This script tests the extraction pipeline against a real-world repository
with multiple file types. It extracts code blocks, builds hierarchy relationships,
produces QA-compatible output, and verifies that significant content is extracted.

Unlike the mocked tests, this test processes actual repository files and generates
real extraction output that can be inspected and verified.

Dependencies:
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- json: For JSON serialization (https://docs.python.org/3/library/json.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)
- tempfile: For temporary file creation (https://docs.python.org/3/library/tempfile.html)

Usage:
    python real_world_test.py [output_path]

Example:
    python real_world_test.py ./extraction_output.json
"""

import sys
import json
import tempfile
import logging
import os
from pathlib import Path

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extraction.real_world_test")

# Import extraction functions
from extraction_blocks import extract_all_blocks, find_source_files
from hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
from validation import validate_qa_output


def run_real_world_test(repo_path: Path, output_path: Path = None) -> bool:
    """
    Run a real-world test of the extraction pipeline on a repository.
    
    Args:
        repo_path: Path to the repository to process
        output_path: Optional path to save the extraction output
        
    Returns:
        True if the test passes, False otherwise
    """
    logger.info(f"Starting real-world extraction test on {repo_path}")
    
    # Create temporary output file if not specified
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()
        logger.info(f"Output will be saved to {output_path}")
    
    try:
        # 1. Extract blocks - modify to include markdown files
        logger.info("Extracting blocks...")
        # Find Python and other code files
        code_files = find_source_files(repo_path)
        logger.info(f"Found {len(code_files)} code files")
        
        # Find markdown files
        md_files = list(repo_path.glob("**/*.md"))
        logger.info(f"Found {len(md_files)} markdown files")
        
        # Check for specific test files of interest
        target_files = {
            "deepseek.md": False,
            "README.md": False
        }
        
        for md_file in md_files:
            file_name = md_file.name
            if file_name in target_files:
                logger.info(f"Found target file: {md_file}")
                target_files[file_name] = True
        
        # Report on target files
        for file_name, found in target_files.items():
            if found:
                logger.info(f"Target file {file_name} found in repository")
            else:
                logger.warning(f"Target file {file_name} NOT found in repository")
        
        # Extract all blocks from the repository
        blocks = extract_all_blocks(repo_path)
        if not blocks:
            logger.error("No blocks extracted")
            return False
        
        logger.info(f"Extracted {len(blocks)} blocks")
        
        # Count blocks by type
        block_types = {}
        for block in blocks:
            block_type = block.get("type", "unknown")
            block_types[block_type] = block_types.get(block_type, 0) + 1
        
        logger.info(f"Block types: {block_types}")
        
        # Verify specific file extraction
        logger.info("Verifying extraction of target files...")
        extracted_files = {path.name: False for path in md_files if path.name in target_files}
        
        # Find file blocks that correspond to target files
        target_file_blocks = {}
        for block in blocks:
            if block.get("type") == "file":
                file_path = block.get("file_path", "")
                file_name = Path(file_path).name
                if file_name in target_files:
                    extracted_files[file_name] = True
                    target_file_blocks[file_name] = block.get("uuid")
                    logger.info(f"Found file block for {file_name}")
        
        # Verify if target files were extracted correctly
        for file_name, extracted in extracted_files.items():
            if extracted:
                logger.info(f"Target file {file_name} was extracted")
            else:
                logger.warning(f"Target file {file_name} was NOT extracted")
        
        # Track elements within specific files
        if "deepseek.md" in target_file_blocks:
            deepseek_uuid = target_file_blocks["deepseek.md"]
            logger.info(f"Checking elements within deepseek.md (UUID: {deepseek_uuid})")
            
            # Count elements inside deepseek.md
            element_counts = {"section": 0, "table": 0, "image": 0, "code_block": 0, "text": 0}
            section_hierarchy = {}
            
            # First pass: collect all sections
            for block in blocks:
                if block.get("parent_uuid") == deepseek_uuid:
                    block_type = block.get("type")
                    if block_type in element_counts:
                        element_counts[block_type] += 1
                    
                    # Track sections and their elements
                    if block_type == "section":
                        section_uuid = block.get("uuid")
                        section_hierarchy[section_uuid] = {
                            "uuid": section_uuid,
                            "name": block.get("name"),
                            "elements": {"table": 0, "image": 0, "code_block": 0, "text": 0}
                        }
                        
                    # Also count direct children of file that aren't sections
                    elif block_type in ["table", "image", "code_block", "text"]:
                        logger.info(f"Found direct {block_type} child of deepseek.md: {block.get('name', 'unnamed')}")
            
            # Collect all blocks directly in the file (not in a section)
            direct_file_elements = []
            for block in blocks:
                if block.get("parent_uuid") == deepseek_uuid and block.get("type") != "section":
                    direct_file_elements.append({
                        "type": block.get("type"),
                        "name": block.get("name", "unnamed"),
                        "content_preview": block.get("content", "")[:50] if block.get("content") else "",
                        "position": block.get("metadata", {}).get("position", float('inf'))
                    })
            
            # Sort direct file elements by position
            direct_file_elements.sort(key=lambda e: e["position"])
            logger.info(f"Direct children of deepseek.md in order:")
            for i, element in enumerate(direct_file_elements):
                logger.info(f"  {i+1}. {element['type']}: {element['name']} - {element['content_preview']}")
            
            # Second pass: check all blocks for parentage to sections
            all_section_children = []
            for block in blocks:
                parent_uuid = block.get("parent_uuid")
                if parent_uuid in section_hierarchy:
                    block_type = block.get("type")
                    if block_type in section_hierarchy[parent_uuid]["elements"]:
                        section_hierarchy[parent_uuid]["elements"][block_type] += 1
                        all_section_children.append({
                            "section": section_hierarchy[parent_uuid]["name"],
                            "type": block_type,
                            "name": block.get("name", "unnamed"),
                            "content_preview": block.get("content", "")[:50] if block.get("content") else "",
                            "position": block.get("metadata", {}).get("position", float('inf'))
                        })
            
            # Sort section children by position
            all_section_children.sort(key=lambda e: e["position"])
            logger.info(f"Children of sections in deepseek.md in order:")
            for i, element in enumerate(all_section_children[:5]):  # Limit to first 5 for brevity
                logger.info(f"  {i+1}. {element['section']} > {element['type']}: {element['name']} - {element['content_preview']}")
            
            # Log element counts
            logger.info(f"Elements in deepseek.md: {element_counts}")
            
            # Log section hierarchy
            logger.info("Section hierarchy in deepseek.md:")
            for section_uuid, data in section_hierarchy.items():
                logger.info(f"  Section '{data['name']}': {data['elements']}")
                
            # Perform deeper inspection of the file block
            logger.info("Inspecting deepseek.md file block content:")
            for block in blocks:
                if block.get("uuid") == deepseek_uuid:
                    file_content = block.get("content", "")
                    logger.info(f"  File has {len(file_content)} characters")
                    logger.info(f"  Contains table markers: {'|' in file_content}")
                    logger.info(f"  Contains image markers: {'![' in file_content}")
                    logger.info(f"  Contains code block markers: {'```' in file_content}")
                    
                    # Count occurrences using regex
                    import re
                    table_count = len(re.findall(r'\|[^\n]+\|\n\|[\s\-:]+\|', file_content))
                    image_count = len(re.findall(r'!\[.*?\]\(.*?\)', file_content))
                    code_block_count = len(re.findall(r'```(\w*)\n.*?\n```', file_content, re.DOTALL))
                    
                    logger.info(f"  Regex found {table_count} tables")
                    logger.info(f"  Regex found {image_count} images")
                    logger.info(f"  Regex found {code_block_count} code blocks")
                    
                    # Log the first 200 characters for verification
                    logger.info(f"  Content preview: {file_content[:200]}")
                    break
        
        # Attempt to validate element ordering
        logger.info("Validating element ordering...")
        ordered_elements = []
        for block in blocks:
            if block.get("type") in ["table", "image", "code_block", "text"]:
                ordered_elements.append({
                    "type": block.get("type"),
                    "position": block.get("metadata", {}).get("position", float('inf')),
                    "parent": block.get("parent_uuid"),
                    "content_preview": block.get("content", "")[:50] if block.get("content") else ""
                })
        
        # Sort elements by position
        ordered_elements.sort(key=lambda e: e["position"])
        
        # Log the first few ordered elements
        logger.info("First 5 ordered elements:")
        for i, element in enumerate(ordered_elements[:5]):
            logger.info(f"  Element {i+1}: {element['type']} at position {element['position']}")
            logger.info(f"    Content preview: {element['content_preview']}")
        
        # 2. Analyze hierarchies
        logger.info("Analyzing hierarchies...")
        hierarchies = analyze_hierarchies(blocks)
        logger.info(f"Analyzed {len(hierarchies)} file hierarchies")
        
        # 3. Enrich blocks with hierarchy
        logger.info("Enriching blocks...")
        enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
        logger.info(f"Enriched {len(enriched_blocks)} blocks")
        
        # 4. Create QA-compatible blocks
        logger.info("Creating QA-compatible blocks...")
        qa_blocks = create_qa_compatible_blocks(enriched_blocks)
        logger.info(f"Created {len(qa_blocks)} QA-compatible blocks")
        
        # 5. Create QA-compatible output
        logger.info("Creating QA-compatible output...")
        output = create_qa_compatible_output(qa_blocks)
        
        # 6. Validate output
        logger.info("Validating output...")
        if not validate_qa_output(output):
            logger.error("Output validation failed")
            return False
        
        # 7. Write output to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Output written to {output_path}")
        
        # 8. Verify significant content was extracted
        stats = output.get("extraction_metadata", {}).get("statistics", {})
        total_blocks = stats.get("total_blocks", 0)
        total_files = stats.get("total_files", 0)
        
        if total_blocks < 10:
            logger.error(f"Too few blocks extracted: {total_blocks}")
            return False
        
        if total_files < 2:
            logger.error(f"Too few files processed: {total_files}")
            return False
        
        logger.info("Real-world extraction test successful")
        logger.info(f"Extracted {total_blocks} blocks from {total_files} files")
        
        # Print some sample blocks for inspection
        logger.info("Sample blocks:")
        for i, block in enumerate(qa_blocks[:3]):
            logger.info(f"Block {i+1}:")
            logger.info(f"  Type: {block.get('type')}")
            logger.info(f"  Name: {block.get('name')}")
            logger.info(f"  Language: {block.get('language')}")
            logger.info(f"  Parent UUID: {block.get('parent_uuid')}")
            logger.info(f"  Child UUIDs: {len(block.get('child_uuids', []))}")
        
        # 9. Final verification of target file extraction
        extraction_success = all(extracted_files.values())
        if extraction_success:
            logger.info("All target files were successfully extracted")
        else:
            logger.warning("Some target files were not extracted")
            for file_name, extracted in extracted_files.items():
                if not extracted:
                    logger.warning(f"Failed to extract {file_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        return False


def main():
    """Main function to run the real-world extraction test."""
    # Determine output path
    output_path = None
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    
    # Get repository path (using sglang which has markdown with tables, images, and code blocks)
    repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/sglang")
    
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path not found: {repo_path}")
        sys.exit(1)
    
    # Run the test
    success = run_real_world_test(repo_path, output_path)
    
    if success:
        logger.info("Test passed!")
        sys.exit(0)
    else:
        logger.error("Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()