#!/usr/bin/env python3
"""
Main Module for End-to-End Extraction Example.

This module provides the entry point for the end-to-end extraction example,
orchestrating the complete pipeline from code files to QA-compatible output.

Key Functions:
- main: Main entry point for the end-to-end extraction example
- extract_repository: Extract from a code repository
- extract_file: Extract from a single file
- extract_markdown: Extract from Markdown content
- extract_html: Extract from HTML content
- analyze_hierarchy: Analyze hierarchy relationships

Dependencies:
- sys: For command line arguments (https://docs.python.org/3/library/sys.html)
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- json: For JSON serialization (https://docs.python.org/3/library/json.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)

Usage:
    python -m agent_tools.dualipa.extraction.examples.end_to_end.main <source_dir> <output_file>
    
Example:
    python -m agent_tools.dualipa.extraction.examples.end_to_end.main ./test_repos/python-sample ./output.json
"""

import sys
import json
import os
from pathlib import Path
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union

# Import extraction modules
try:
    # Try relative import first
    from .extraction_blocks import extract_all_blocks, find_source_files
    from .hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from .qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from .validation import validate_qa_output, validate_extraction
except ImportError:
    # Fall back to direct import
    from extraction_blocks import extract_all_blocks, find_source_files
    from hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from validation import validate_qa_output, validate_extraction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extraction.main")


def extract_repository(repo_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Extract code blocks from a repository.
    
    Args:
        repo_path: Path to the repository
        output_dir: Optional output directory for extracted blocks
        
    Returns:
        Dictionary containing extracted blocks and metadata
    """
    repo_path = Path(repo_path) if not isinstance(repo_path, Path) else repo_path
    
    if not repo_path.exists() or not repo_path.is_dir():
        logger.error(f"Repository path not found: {repo_path}")
        return {"error": f"Repository path not found: {repo_path}"}
    
    # Use temporary directory if not specified
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="extraction_"))
    else:
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Extracting from repository: {repo_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Extract blocks
    blocks = extract_all_blocks(repo_path)
    if not blocks:
        logger.warning("No blocks extracted from repository")
        return {"blocks": [], "metadata": {"source": str(repo_path), "output_dir": str(output_dir)}}
    
    # Analyze hierarchies
    hierarchies = analyze_hierarchies(blocks)
    
    # Enrich blocks with hierarchy
    enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
    
    # Create QA-compatible blocks
    qa_blocks = create_qa_compatible_blocks(enriched_blocks)
    
    # Create output
    output = {
        "blocks": qa_blocks,
        "metadata": {
            "source": str(repo_path),
            "output_dir": str(output_dir),
            "num_blocks": len(qa_blocks),
            "num_files": len(hierarchies)
        }
    }
    
    # Save output to file
    output_file = output_dir / "extraction_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Extraction complete. Output written to {output_file}")
    logger.info(f"Extracted {len(qa_blocks)} blocks from {len(hierarchies)} files")
    
    return output


def extract_file(file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Extract code blocks from a single file.
    
    Args:
        file_path: Path to the file
        output_dir: Optional output directory for extracted blocks
        
    Returns:
        Dictionary containing extracted blocks and metadata
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if not file_path.exists() or not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    # Use temporary directory if not specified
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="extraction_"))
    else:
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Extracting from file: {file_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create a file list with a single file
    files = [file_path]
    
    # Extract blocks
    blocks = []
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract blocks from file content
            file_blocks = extract_all_blocks(file, content=content)
            blocks.extend(file_blocks)
        except Exception as e:
            logger.error(f"Error extracting from {file}: {e}")
    
    if not blocks:
        logger.warning("No blocks extracted from file")
        return {"blocks": [], "metadata": {"source": str(file_path), "output_dir": str(output_dir)}}
    
    # Analyze hierarchies
    hierarchies = analyze_hierarchies(blocks)
    
    # Enrich blocks with hierarchy
    enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
    
    # Create QA-compatible blocks
    qa_blocks = create_qa_compatible_blocks(enriched_blocks)
    
    # Create output
    output = {
        "blocks": qa_blocks,
        "metadata": {
            "source": str(file_path),
            "output_dir": str(output_dir),
            "num_blocks": len(qa_blocks)
        }
    }
    
    # Save output to file
    output_file = output_dir / "extraction_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Extraction complete. Output written to {output_file}")
    logger.info(f"Extracted {len(qa_blocks)} blocks")
    
    return output


def extract_markdown(markdown_content: str, file_path: Optional[Union[str, Path]] = None, 
                    output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Extract code blocks from Markdown content.
    
    Args:
        markdown_content: Markdown content as string
        file_path: Optional file path for reference
        output_dir: Optional output directory for extracted blocks
        
    Returns:
        Dictionary containing extracted blocks and metadata
    """
    # Use temporary directory if not specified
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="extraction_"))
    else:
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    source_name = str(file_path) if file_path else "markdown_content"
    logger.info(f"Extracting from Markdown: {source_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save content to temporary file if no file path provided
    if file_path is None:
        temp_file = output_dir / "temp_markdown.md"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        file_path = temp_file
    else:
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    # Extract blocks using the markdown-specific extractor
    try:
        # Import markdown-specific extractor here to avoid circular imports
        from agent_tools.dualipa.extraction.extractors.markdown.markdown_extractor import extract_markdown_blocks
        blocks = extract_markdown_blocks(markdown_content, str(file_path))
    except ImportError:
        logger.error("Markdown extractor not available")
        return {"error": "Markdown extractor not available"}
    
    if not blocks:
        logger.warning("No blocks extracted from Markdown")
        return {"blocks": [], "metadata": {"source": source_name, "output_dir": str(output_dir)}}
    
    # Process blocks (no hierarchy analysis needed for flat markdown)
    qa_blocks = create_qa_compatible_blocks(blocks)
    
    # Create output
    output = {
        "blocks": qa_blocks,
        "metadata": {
            "source": source_name,
            "output_dir": str(output_dir),
            "num_blocks": len(qa_blocks)
        }
    }
    
    # Save output to file
    output_file = output_dir / "markdown_extraction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Extraction complete. Output written to {output_file}")
    logger.info(f"Extracted {len(qa_blocks)} blocks from Markdown")
    
    return output


def extract_html(html_content: str, url: Optional[str] = None, 
                output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Extract content from HTML.
    
    Args:
        html_content: HTML content as string
        url: Optional URL for reference
        output_dir: Optional output directory for extracted content
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    # Use temporary directory if not specified
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="extraction_"))
    else:
        output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    source_name = url if url else "html_content"
    logger.info(f"Extracting from HTML: {source_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save content to temporary file
    temp_file = output_dir / "temp_html.html"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Extract content using the HTML-specific extractor
    try:
        # Import HTML-specific extractor here to avoid circular imports
        from agent_tools.dualipa.extraction.extractors.docs.docs_extractor import extract_html_content
        blocks = extract_html_content(html_content, url)
    except ImportError:
        logger.error("HTML extractor not available")
        return {"error": "HTML extractor not available"}
    
    if not blocks:
        logger.warning("No content extracted from HTML")
        return {"blocks": [], "metadata": {"source": source_name, "output_dir": str(output_dir)}}
    
    # Process blocks
    qa_blocks = create_qa_compatible_blocks(blocks)
    
    # Create output
    output = {
        "blocks": qa_blocks,
        "metadata": {
            "source": source_name,
            "output_dir": str(output_dir),
            "num_blocks": len(qa_blocks)
        }
    }
    
    # Save output to file
    output_file = output_dir / "html_extraction.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Extraction complete. Output written to {output_file}")
    logger.info(f"Extracted {len(qa_blocks)} blocks from HTML")
    
    return output


def analyze_hierarchy(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze hierarchical relationships in blocks.
    
    Args:
        blocks: List of blocks to analyze
        
    Returns:
        Dictionary containing hierarchy analysis
    """
    try:
        # Analyze hierarchies
        hierarchies = analyze_hierarchies(blocks)
        
        # Enrich blocks with hierarchy
        enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
        
        return {
            "blocks": enriched_blocks,
            "hierarchies": hierarchies,
            "metadata": {
                "num_blocks": len(blocks),
                "num_hierarchies": len(hierarchies)
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing hierarchy: {e}")
        return {
            "error": f"Error analyzing hierarchy: {e}",
            "blocks": blocks,
            "hierarchies": {}
        }


def main():
    """Main function to run the end-to-end extraction example.
    
    This function orchestrates the complete extraction pipeline:
    1. Extract blocks from source files
    2. Analyze hierarchical relationships
    3. Enrich blocks with hierarchy information
    4. Convert to QA-compatible format
    5. Validate output
    6. Write output to file
    
    Usage:
        python -m agent_tools.dualipa.extraction.examples.end_to_end.main <source_dir> <output_file>
        
    Example:
        python -m agent_tools.dualipa.extraction.examples.end_to_end.main ./test_repos/python-sample ./output.json
    """
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_file>")
        sys.exit(1)
    
    source_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Extract blocks
    blocks = extract_all_blocks(source_dir)
    if not blocks:
        logger.error("No blocks extracted")
        sys.exit(1)
    
    # Analyze hierarchies
    hierarchies = analyze_hierarchies(blocks)
    
    # Enrich blocks with hierarchy
    enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
    
    # Create QA-compatible blocks
    qa_blocks = create_qa_compatible_blocks(enriched_blocks)
    
    # Create QA-compatible output
    output = create_qa_compatible_output(qa_blocks)
    
    # Validate output
    if not validate_qa_output(output):
        logger.error("Output validation failed")
        sys.exit(1)
    
    # Write output to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Extraction complete. Output written to {output_file}")
    logger.info(f"Extracted {len(qa_blocks)} blocks from {len(hierarchies)} files")


if __name__ == "__main__":
    main()