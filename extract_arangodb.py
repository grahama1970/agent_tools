#!/usr/bin/env python3
"""
ArangoDB Extraction Script

This script performs a comprehensive extraction of ArangoDB documentation and code,
creating a JSON format suitable for Question-Answering tasks. It integrates state management
to ensure reliable extraction, even with context limitations.

Key features:
1. Processes ArangoDB documentation using fetch_docs module
2. Extracts code from the ArangoDB repository
3. Maintains hierarchical relationships between sections
4. Generates QA-compatible JSON output
5. Validates extraction completeness

Usage:
    python extract_arangodb.py [--output OUTPUT]
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import datetime
import requests
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Initialize state manager for tracking extraction progress
try:
    from src.agent_tools.dualipa.extraction.test_state_manager import get_state_manager, TestStateManager
    from src.agent_tools.dualipa.extraction.memory import remember, think, remind_me
    state_manager = get_state_manager("extraction_state.db")
    use_state_manager = True
except ImportError:
    print("State manager not available, proceeding without state tracking")
    use_state_manager = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("arangodb_extraction")

# ArangoDB resources to extract
GITHUB_REPO = "arangodb/arangodb"
GITHUB_BRANCH = "bba7f899831ee71373e3f673e30148154cb9f761"  # Use a specific commit for consistency
KEY_FILES = [
    "utils/gantt.py",
    "scripts/buildUnittestBashCompletion.bash",
    "js/apps/system/_admin/aardvark/APP/react/src/views/query/ArangoQuery.types.ts",
    "js/apps/system/_admin/aardvark/APP/react/src/views/views/ViewsList.tsx",
    "scripts/toolbox/modules/Pipeline.py"
]

DOC_URLS = [
    "https://docs.arangodb.com/stable/aql/data-queries/",
    "https://docs.arangodb.com/stable/aql/functions/arangosearch/",
    "https://docs.arangodb.com/stable/aql/operators/"
]


def set_extraction_checkpoint(name: str, description: str = "") -> None:
    """Set a checkpoint for the extraction process."""
    if use_state_manager:
        state_manager.set_checkpoint(name, description)
        logger.info(f"Checkpoint set: {name} - {description}")
    else:
        logger.info(f"Checkpoint: {name} - {description}")


def fetch_github_file(repo: str, branch: str, file_path: str, target_dir: Path) -> Optional[Path]:
    """
    Download a specific file from GitHub.
    
    Args:
        repo: GitHub repository (owner/repo)
        branch: Git branch or commit hash
        file_path: Path to the file within the repository
        target_dir: Directory to save the file
        
    Returns:
        Path to downloaded file or None if download failed
    """
    if use_state_manager:
        remember(
            f"Fetching GitHub file: {file_path}",
            "Download source code for extraction",
            "Preparing request to GitHub raw content",
            "Download file and save to target directory"
        )
    
    try:
        # Construct raw URL
        raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
        logger.info(f"Fetching file from {raw_url}")
        
        # Make request
        response = requests.get(raw_url)
        response.raise_for_status()
        
        # Create filename
        file_name = os.path.basename(file_path)
        local_path = target_dir / file_name
        
        # Save file
        with open(local_path, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Saved file to {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error fetching GitHub file {file_path}: {e}")
        if use_state_manager:
            state_manager.log_verification("fetch_github_file", 
                                          {"repo": repo, "branch": branch, "file_path": file_path},
                                          {"error": str(e)},
                                          False)
        return None


def download_docs_page(url: str, target_dir: Path) -> Optional[Path]:
    """
    Download an ArangoDB documentation page.
    
    Args:
        url: Documentation URL to download
        target_dir: Directory to save the downloaded file
        
    Returns:
        Path to downloaded file or None if download failed
    """
    if use_state_manager:
        remember(
            f"Downloading docs page: {url}",
            "Download documentation for extraction",
            "Preparing request to ArangoDB docs site",
            "Download page and save to target directory"
        )
    
    try:
        # Import fetch_docs utility if available
        try:
            from src.agent_tools.fetch_docs.download_site import download_site
            logger.info("Using fetch_docs download utility")
            download_available = True
        except ImportError:
            logger.warning("fetch_docs not available, using fallback download")
            download_available = False
        
        # Create output directory
        output_dir = target_dir / "docs"
        output_dir.mkdir(exist_ok=True)
        
        # Create a unique subdirectory for this URL
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        doc_dir = output_dir / url_hash
        doc_dir.mkdir(exist_ok=True)
        
        if download_available:
            # Use fetch_docs utility to download
            download_site(url, str(doc_dir), recursive=False)
        else:
            # Fallback to direct download
            response = requests.get(url)
            response.raise_for_status()
            
            # Save file
            file_name = "index.html"
            local_path = doc_dir / file_name
            with open(local_path, 'wb') as f:
                f.write(response.content)
                
        # Look for the downloaded HTML file
        html_files = list(doc_dir.glob("**/*.html"))
        if not html_files:
            logger.error(f"No HTML files found after downloading {url}")
            return None
            
        # Return the path to the first HTML file
        return html_files[0]
        
    except Exception as e:
        logger.error(f"Error downloading documentation from {url}: {e}")
        if use_state_manager:
            state_manager.log_verification("download_docs_page", 
                                          {"url": url},
                                          {"error": str(e)},
                                          False)
        return None


def extract_code_blocks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract code blocks from a file.
    
    Args:
        file_path: Path to the file to extract from
        
    Returns:
        List of extracted code blocks
    """
    if use_state_manager:
        remember(
            f"Extracting code blocks from: {file_path}",
            "Process source code to extract structured blocks",
            "Loading file and detecting language",
            "Extract functions, classes, methods and their relationships"
        )
    
    try:
        # Import the necessary extraction functions
        try:
            from src.agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import extract_all_blocks
            logger.info("Using extraction_blocks module")
        except ImportError:
            try:
                from src.agent_tools.dualipa.extraction.extractors.code.hierarchy import extract_code_hierarchy
                logger.info("Using code hierarchy module")
                
                def extract_all_blocks(source_dir):
                    """Wrapper function for extract_code_hierarchy"""
                    if isinstance(source_dir, Path):
                        file_path = str(source_dir)
                    else:
                        file_path = source_dir
                        
                    return extract_code_hierarchy(file_path)
            except ImportError:
                logger.error("Could not import extraction modules")
                return []
                
        # Extract code blocks
        logger.info(f"Extracting code blocks from {file_path}")
        
        # Create a Path object from the file path
        path_obj = Path(file_path)
        
        # Extract blocks
        blocks = extract_all_blocks(path_obj.parent)
        
        # Filter blocks for this specific file
        file_blocks = [block for block in blocks if block.get("file_path") == str(file_path)]
        
        logger.info(f"Extracted {len(file_blocks)} code blocks from {file_path}")
        return file_blocks
        
    except Exception as e:
        logger.error(f"Error extracting code blocks: {e}")
        if use_state_manager:
            state_manager.log_verification("extract_code_blocks", 
                                          {"file_path": file_path},
                                          {"error": str(e)},
                                          False)
        return []


def extract_documentation(html_file: Path) -> List[Dict[str, Any]]:
    """
    Extract documentation blocks from an HTML file.
    
    Args:
        html_file: Path to the HTML file
        
    Returns:
        List of extracted documentation blocks
    """
    if use_state_manager:
        remember(
            f"Extracting documentation from: {html_file}",
            "Process HTML docs to extract structured blocks",
            "Loading HTML and cleaning content",
            "Extract sections, code examples, tables, etc."
        )
    
    try:
        # Try to import required functions
        try:
            from src.agent_tools.fetch_docs.clean_html import clean_html
            from src.agent_tools.fetch_docs.extract_sections import extract_sections_from_html
            from src.agent_tools.dualipa.fetch_docs_integration import convert_to_dualipa_format
            logger.info("Successfully imported documentation extraction functions")
        except ImportError:
            logger.warning("Could not import documentation extraction functions, using fallback")
            
            # Define fallback functions
            def clean_html(html_content):
                """Simple HTML cleaning function."""
                import re
                # Remove scripts and styles
                html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
                html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
                return html_content
                
            def extract_sections_from_html(html_content, file_path=None):
                """Simple section extraction function."""
                import re
                # Extract headings and content
                sections = []
                
                # Try to extract title
                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.DOTALL)
                title = title_match.group(1) if title_match else "Unknown Title"
                
                # Create a basic section
                sections.append({
                    "header": title,
                    "content": html_content,
                    "level": 1,
                    "token_count": len(html_content.split())
                })
                
                return sections
                
            def convert_to_dualipa_format(processed_docs, repo_path):
                """Simple conversion function."""
                import uuid
                blocks = []
                
                for url, site_data in processed_docs.items():
                    # Create site block
                    site_uuid = str(uuid.uuid4())
                    blocks.append({
                        "uuid": site_uuid,
                        "id": f"docs_{url.split('//')[-1].split('/')[0].replace('.', '_')}",
                        "name": f"Documentation: {url}",
                        "type": "documentation",
                        "language": "html",
                        "content": f"Documentation site: {url}",
                        "source_url": url,
                        "child_uuids": [],
                        "metadata": {
                            "language": "html",
                            "source_url": url,
                            "doc_type": "arangodb" if "arangodb" in url else "generic"
                        }
                    })
                    
                    # Add pages and sections
                    for file_data in site_data:
                        # Add page block
                        page_uuid = str(uuid.uuid4())
                        page_name = "Documentation Page"
                        blocks.append({
                            "uuid": page_uuid,
                            "id": f"docs_page_{len(blocks)}",
                            "name": page_name,
                            "type": "doc_page",
                            "language": "html",
                            "content": f"Documentation page from {url}",
                            "file_path": file_data.get("file", ""),
                            "parent_uuid": site_uuid,
                            "child_uuids": [],
                            "metadata": {
                                "language": "html",
                                "source_url": url,
                                "doc_type": "arangodb" if "arangodb" in url else "generic"
                            }
                        })
                        blocks[0]["child_uuids"].append(page_uuid)
                        
                        # Add section blocks
                        for i, section in enumerate(file_data.get("sections", [])):
                            section_uuid = str(uuid.uuid4())
                            section_title = section.get("header", f"Section {i+1}")
                            blocks.append({
                                "uuid": section_uuid,
                                "id": f"docs_section_{len(blocks)}",
                                "name": section_title,
                                "type": "doc_section",
                                "language": "html",
                                "content": section.get("content", ""),
                                "file_path": file_data.get("file", ""),
                                "parent_uuid": page_uuid,
                                "child_uuids": [],
                                "metadata": {
                                    "language": "html",
                                    "source_url": url,
                                    "doc_type": "arangodb" if "arangodb" in url else "generic",
                                    "header_level": section.get("level", 1)
                                }
                            })
                            blocks[1]["child_uuids"].append(section_uuid)
                
                return blocks
        
        # Read HTML file
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
            
        # Clean HTML
        cleaned_content = clean_html(html_content)
        
        # Extract sections
        sections = extract_sections_from_html(cleaned_content, html_file)
        
        # Create file data structure for conversion
        url = f"https://docs.arangodb.com/{html_file.parent.name}/"
        site_data = [{
            "file": str(html_file),
            "relative_path": html_file.name,
            "sections": sections,
            "doc_type": "arangodb"
        }]
        
        # Convert to DuaLipa format
        processed_docs = {url: site_data}
        blocks = convert_to_dualipa_format(processed_docs, html_file.parent)
        
        logger.info(f"Extracted {len(blocks)} documentation blocks from {html_file}")
        return blocks
        
    except Exception as e:
        logger.error(f"Error extracting documentation: {e}")
        if use_state_manager:
            state_manager.log_verification("extract_documentation", 
                                          {"html_file": str(html_file)},
                                          {"error": str(e)},
                                          False)
        
        # Create minimal fallback blocks
        import uuid
        site_uuid = str(uuid.uuid4())
        page_uuid = str(uuid.uuid4())
        section_uuid = str(uuid.uuid4())
        
        blocks = [
            {
                "uuid": site_uuid,
                "id": "docs_arangodb",
                "name": "Documentation: ArangoDB",
                "type": "documentation",
                "language": "html",
                "content": f"Documentation site: {html_file}",
                "child_uuids": [page_uuid],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb"
                }
            },
            {
                "uuid": page_uuid,
                "id": "docs_page",
                "name": html_file.name,
                "type": "doc_page",
                "language": "html",
                "content": f"Documentation page from {html_file}",
                "parent_uuid": site_uuid,
                "child_uuids": [section_uuid],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb"
                }
            },
            {
                "uuid": section_uuid,
                "id": "docs_section",
                "name": "Content",
                "type": "doc_section",
                "language": "html",
                "content": "Content section",
                "parent_uuid": page_uuid,
                "child_uuids": [],
                "metadata": {
                    "language": "html",
                    "doc_type": "arangodb",
                    "header_level": 1
                }
            }
        ]
        
        logger.info(f"Created fallback documentation blocks for {html_file}")
        return blocks


def validate_and_fix_parent_references(blocks: List[Dict[str, Any]]) -> None:
    """
    Ensure all parent-child relationships are valid.
    
    Args:
        blocks: List of blocks to validate and fix
    """
    if use_state_manager:
        remember(
            "Validating parent-child references",
            "Ensure hierarchical relationships are correct",
            "Checking for invalid or missing references",
            "Fix any issues to maintain proper hierarchy"
        )
    
    # Create a set of all UUIDs for quick lookup
    all_uuids = {block["uuid"] for block in blocks}
    
    # Create a map from UUID to block for easier updates
    uuid_to_block = {block["uuid"]: block for block in blocks}
    
    # Track invalid and fixed references
    invalid_refs = 0
    fixed_refs = 0
    
    # Check each block's parent reference
    for block in blocks:
        if "parent_uuid" in block and block["parent_uuid"]:
            parent_uuid = block["parent_uuid"]
            
            if parent_uuid not in all_uuids:
                invalid_refs += 1
                logger.warning(f"Block {block.get('name')} references non-existent parent UUID: {parent_uuid}")
                
                # Try to find another parent based on file path
                fixed = False
                file_path = block.get("file_path", "")
                
                if file_path:
                    for potential_parent in blocks:
                        if (potential_parent.get("type") == "file" and 
                                potential_parent.get("file_path") == file_path):
                            block["parent_uuid"] = potential_parent["uuid"]
                            
                            # Add this block to the parent's children if needed
                            if "child_uuids" not in potential_parent:
                                potential_parent["child_uuids"] = []
                            if block["uuid"] not in potential_parent["child_uuids"]:
                                potential_parent["child_uuids"].append(block["uuid"])
                            
                            fixed = True
                            fixed_refs += 1
                            break
                            
                if not fixed:
                    # If still not fixed, remove parent reference
                    block["parent_uuid"] = None
    
    # Check child references
    for block in blocks:
        if "child_uuids" in block:
            valid_children = []
            for child_uuid in block["child_uuids"]:
                if child_uuid in all_uuids:
                    valid_children.append(child_uuid)
                    
                    # Make sure the child's parent_uuid is set correctly
                    child_block = uuid_to_block.get(child_uuid)
                    if child_block and child_block.get("parent_uuid") != block["uuid"]:
                        child_block["parent_uuid"] = block["uuid"]
                        fixed_refs += 1
            
            # Update with only valid children
            block["child_uuids"] = valid_children
            
    logger.info(f"Fixed {fixed_refs} invalid parent references out of {invalid_refs} invalid references")


def convert_to_qa_format(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert blocks to QA-compatible format.
    
    Args:
        blocks: List of blocks to convert
        
    Returns:
        QA-compatible output format
    """
    if use_state_manager:
        remember(
            "Converting to QA format",
            "Create format suitable for Question-Answering",
            "Organizing blocks and fixing relationships",
            "Generate final JSON output"
        )
    
    try:
        # Try to import QA formatter
        try:
            from src.agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
            logger.info("Successfully imported QA formatter")
        except ImportError:
            logger.warning("Could not import QA formatter, using fallback conversion")
            
            # Define fallback functions
            def create_qa_compatible_blocks(blocks):
                """Basic function to make blocks QA-compatible."""
                import uuid
                qa_blocks = []
                
                for block in blocks:
                    # Create a copy to avoid modifying the original
                    enhanced_block = block.copy()
                    
                    # Ensure all blocks have required fields
                    if "uuid" not in enhanced_block:
                        enhanced_block["uuid"] = str(uuid.uuid4())
                    if "extraction_focus" not in enhanced_block:
                        if enhanced_block.get("type") in ["section", "text", "doc_section"]:
                            enhanced_block["extraction_focus"] = ["documentation"]
                        else:
                            enhanced_block["extraction_focus"] = ["code"]
                    if "summary_instructions" not in enhanced_block:
                        enhanced_block["summary_instructions"] = "Extract key points from content"
                    if "breadcrumb" not in enhanced_block:
                        # Create breadcrumb based on file path and name
                        file_path = enhanced_block.get("file_path", "")
                        name = enhanced_block.get("name", "")
                        if file_path:
                            file_name = file_path.split("/")[-1] if "/" in file_path else file_path
                            enhanced_block["breadcrumb"] = [file_name]
                            if name:
                                enhanced_block["breadcrumb"].append(name)
                        else:
                            enhanced_block["breadcrumb"] = [name or "Unnamed Block"]
                    if "parent_uuid" not in enhanced_block:
                        enhanced_block["parent_uuid"] = None
                    if "child_uuids" not in enhanced_block:
                        enhanced_block["child_uuids"] = []
                        
                    qa_blocks.append(enhanced_block)
                    
                return qa_blocks
                
            def create_qa_compatible_output(blocks):
                """Basic function to create output format."""
                # Build statistics
                block_types = {}
                languages = {}
                file_count = 0
                
                for block in blocks:
                    block_type = block.get("type", "unknown")
                    block_types[block_type] = block_types.get(block_type, 0) + 1
                    
                    if block_type == "file":
                        file_count += 1
                    
                    language = block.get("language", "unknown")
                    languages[language] = languages.get(language, 0) + 1
                
                # Generate metadata
                metadata = {
                    "statistics": {
                        "total_blocks": len(blocks),
                        "total_files": file_count,
                        "block_types": block_types,
                        "languages": languages
                    },
                    "model_used": "dualipa-extraction",
                    "timestamp": str(datetime.datetime.now().isoformat()),
                    "version": "1.0"
                }
                
                # Build section relationships
                section_relationships = {
                    "parent_child": {},
                    "imports": {},
                    "inheritance": {}
                }
                
                # Create relationship structures
                for block in blocks:
                    block_uuid = block.get("uuid")
                    parent_uuid = block.get("parent_uuid")
                    child_uuids = block.get("child_uuids", [])
                    
                    # Add to parent-child relationships
                    if block_uuid:
                        section_relationships["parent_child"][block_uuid] = {
                            "parent": parent_uuid,
                            "children": child_uuids
                        }
                
                # Return standard format with all blocks as sections and metadata
                return {
                    "sections": blocks,
                    "section_relationships": section_relationships,
                    "extraction_metadata": metadata
                }
        
        # Make blocks QA-compatible
        qa_blocks = create_qa_compatible_blocks(blocks)
        
        # Create final output
        output = create_qa_compatible_output(qa_blocks)
        
        logger.info(f"Created QA-compatible output with {len(qa_blocks)} blocks")
        
        # If using state manager, store statistics
        if use_state_manager:
            block_types = output.get("extraction_metadata", {}).get("statistics", {}).get("block_types", {})
            for block_type, count in block_types.items():
                state_manager.set_metadata(f"block_type_{block_type}", count)
            
        return output
    
    except Exception as e:
        logger.error(f"Error converting to QA format: {e}")
        if use_state_manager:
            state_manager.log_verification("convert_to_qa_format", 
                                         {"blocks_count": len(blocks)},
                                         {"error": str(e)},
                                         False)
        
        # Return a minimal format as fallback
        return {
            "sections": blocks,
            "section_relationships": {},
            "extraction_metadata": {
                "timestamp": str(datetime.datetime.now().isoformat()),
                "statistics": {"total_blocks": len(blocks)}
            }
        }


def extract_arangodb(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to extract ArangoDB documentation and code.
    
    Args:
        output_path: Optional path to save output JSON
        
    Returns:
        Dictionary containing extraction results and stats
    """
    start_time = time.time()
    
    # Create temporary directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="arangodb_extraction_"))
    
    if use_state_manager:
        set_extraction_checkpoint("start", "Starting ArangoDB extraction")
        state_manager.set_metadata("extraction_start_time", datetime.datetime.now().isoformat())
        remember(
            "Starting ArangoDB extraction",
            "Create comprehensive QA-compatible extraction",
            "Setting up extraction environment",
            "Download files, extract content, generate QA format"
        )
    
    # Initialize results
    results = {
        "success": False,
        "timestamp": datetime.datetime.now().isoformat(),
        "files_processed": 0,
        "docs_processed": 0,
        "extraction_stats": {}
    }
    
    try:
        # Step 1: Fetch GitHub files
        logger.info("Step 1: Fetching GitHub files")
        set_extraction_checkpoint("fetch_github", "Fetching GitHub files")
        
        code_files = []
        for file_path in KEY_FILES:
            local_path = fetch_github_file(GITHUB_REPO, GITHUB_BRANCH, file_path, temp_dir)
            if local_path:
                code_files.append(str(local_path))
                results["files_processed"] += 1
        
        if use_state_manager:
            state_manager.verify("github_files_fetched", len(KEY_FILES), results["files_processed"])
        
        logger.info(f"Successfully fetched {len(code_files)} of {len(KEY_FILES)} code files")
        
        # Step 2: Download documentation
        logger.info("Step 2: Downloading documentation")
        set_extraction_checkpoint("fetch_docs", "Downloading documentation")
        
        doc_files = []
        for url in DOC_URLS:
            local_path = download_docs_page(url, temp_dir)
            if local_path:
                doc_files.append(str(local_path))
                results["docs_processed"] += 1
        
        if use_state_manager:
            state_manager.verify("docs_downloaded", len(DOC_URLS), results["docs_processed"])
        
        logger.info(f"Successfully downloaded {len(doc_files)} of {len(DOC_URLS)} documentation pages")
        
        # Step 3: Extract code from files
        logger.info("Step 3: Extracting code from files")
        set_extraction_checkpoint("extract_code", "Extracting code blocks")
        
        code_blocks_list = []
        for file_path in code_files:
            blocks = extract_code_blocks(file_path)
            if blocks:
                code_blocks_list.append(blocks)
        
        # Flatten code blocks
        code_blocks = [block for blocks in code_blocks_list for block in blocks]
        logger.info(f"Extracted {len(code_blocks)} code blocks total")
        
        # Track code extraction stats
        if use_state_manager:
            state_manager.set_metadata("code_blocks_count", len(code_blocks))
            
            # Track block types
            code_block_types = {}
            for block in code_blocks:
                block_type = block.get("type", "unknown")
                code_block_types[block_type] = code_block_types.get(block_type, 0) + 1
            state_manager.set_metadata("code_block_types", code_block_types)
        
        # Step 4: Extract documentation
        logger.info("Step 4: Extracting documentation")
        set_extraction_checkpoint("extract_docs", "Extracting documentation blocks")
        
        doc_blocks_list = []
        for file_path in doc_files:
            blocks = extract_documentation(Path(file_path))
            if blocks:
                doc_blocks_list.append(blocks)
        
        # Flatten doc blocks
        doc_blocks = [block for blocks in doc_blocks_list for block in blocks]
        logger.info(f"Extracted {len(doc_blocks)} documentation blocks total")
        
        # Track documentation extraction stats
        if use_state_manager:
            state_manager.set_metadata("doc_blocks_count", len(doc_blocks))
            
            # Track block types
            doc_block_types = {}
            for block in doc_blocks:
                block_type = block.get("type", "unknown")
                doc_block_types[block_type] = doc_block_types.get(block_type, 0) + 1
            state_manager.set_metadata("doc_block_types", doc_block_types)
        
        # Step 5: Combine and validate
        logger.info("Step 5: Combining and validating extraction results")
        set_extraction_checkpoint("combine_validate", "Combining and validating blocks")
        
        all_blocks = code_blocks + doc_blocks
        
        # Fix parent-child references
        validate_and_fix_parent_references(all_blocks)
        
        # Step 6: Convert to QA format
        logger.info("Step 6: Converting to QA-compatible format")
        set_extraction_checkpoint("qa_format", "Converting to QA format")
        
        qa_output = convert_to_qa_format(all_blocks)
        
        # If output path provided, save output
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(qa_output, f, indent=2)
                
            logger.info(f"Saved QA-compatible output to {output_file}")
            results["output_path"] = str(output_file)
        
        # Set results
        results["success"] = True
        results["extraction_stats"] = qa_output.get("extraction_metadata", {}).get("statistics", {})
        results["blocks_count"] = len(qa_output.get("sections", []))
        
        # Set extraction complete checkpoint
        set_extraction_checkpoint("complete", "Extraction completed successfully")
        if use_state_manager:
            state_manager.set_metadata("extraction_end_time", datetime.datetime.now().isoformat())
            state_manager.set_metadata("extraction_duration", time.time() - start_time)
            state_manager.set_metadata("extraction_success", True)
        
        logger.info("✅ Extraction completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        results["error"] = str(e)
        
        if use_state_manager:
            set_extraction_checkpoint("error", f"Extraction failed: {e}")
            state_manager.set_metadata("extraction_error", str(e))
            state_manager.set_metadata("extraction_end_time", datetime.datetime.now().isoformat())
            state_manager.set_metadata("extraction_duration", time.time() - start_time)
            state_manager.set_metadata("extraction_success", False)
            
        return results
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Extract ArangoDB documentation and code for QA")
    parser.add_argument("--output", "-o", type=str, default="arangodb_qa_compatible.json",
                       help="Path to save QA-compatible JSON output")
    args = parser.parse_args()
    
    # Run extraction
    results = extract_arangodb(args.output)
    
    # Print results
    if results["success"]:
        print("\n✅ Extraction completed successfully!")
    else:
        print("\n❌ Extraction failed.")
        if "error" in results:
            print(f"Error: {results['error']}")
    
    # Print statistics
    print("\nExtraction Statistics:")
    for key, value in results.get("extraction_stats", {}).items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Print block types if available
    block_types = results.get("extraction_stats", {}).get("block_types", {})
    if block_types:
        print("\nBlock Types:")
        for block_type, count in block_types.items():
            print(f"  {block_type}: {count}")
    
    print(f"\nOutput saved to: {results.get('output_path', args.output)}")
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())