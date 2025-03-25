#!/usr/bin/env python3
"""
End-to-End Extraction Example for DuaLipa.

This script demonstrates the complete extraction pipeline from code files
to QA-compatible output format. It shows how to extract code blocks, build
hierarchy relationships, and produce output that meets QA module requirements.

This file is being maintained for backwards compatibility. The actual 
implementation has been refactored into smaller modules in the 
./end_to_end directory.

Usage:
    python end_to_end_extraction.py <source_dir> <output_file>

Example:
    python end_to_end_extraction.py ./test_repos/python-sample ./output.json
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("extraction")

# Import refactored modules
try:
    from .end_to_end.main import main
    from .end_to_end.extraction_blocks import find_source_files, extract_all_blocks
    from .end_to_end.hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
    from .end_to_end.qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
    from .end_to_end.validation import validate_qa_output
except ImportError as e:
    logger.error(f"Failed to import refactored modules: {e}")
    logger.error("Falling back to direct imports...")
    
    # Add path for direct imports
    sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))
    
    try:
        from agent_tools.dualipa.extraction.examples.end_to_end.main import main
        from agent_tools.dualipa.extraction.examples.end_to_end.extraction_blocks import find_source_files, extract_all_blocks
        from agent_tools.dualipa.extraction.examples.end_to_end.hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
        from agent_tools.dualipa.extraction.examples.end_to_end.qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
        from agent_tools.dualipa.extraction.examples.end_to_end.validation import validate_qa_output
    except ImportError as e2:
        logger.error(f"Failed to import refactored modules with absolute paths: {e2}")
        logger.error("Make sure you run this script from the project root or add the project to your PYTHONPATH")
        sys.exit(1)

# Re-export symbols for backwards compatibility
__all__ = [
    'find_source_files',
    'extract_all_blocks',
    'analyze_hierarchies',
    'enrich_blocks_with_hierarchy',
    'create_qa_compatible_blocks',
    'create_qa_compatible_output',
    'validate_qa_output',
    'main'
]

if __name__ == "__main__":
    main()