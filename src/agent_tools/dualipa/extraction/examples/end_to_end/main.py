#!/usr/bin/env python3
"""
Main Module for End-to-End Extraction Example.

This module provides the entry point for the end-to-end extraction example,
orchestrating the complete pipeline from code files to QA-compatible output.

Key Functions:
- main: Main entry point for the end-to-end extraction example

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
from pathlib import Path
import logging

# Import extraction modules
try:
    # Try relative import first
    from .extraction_blocks import extract_all_blocks
    from .hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from .qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from .validation import validate_qa_output
except ImportError:
    # Fall back to direct import
    from extraction_blocks import extract_all_blocks
    from hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from validation import validate_qa_output

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extraction.main")


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