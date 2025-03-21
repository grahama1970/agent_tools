"""
Code extraction module for DuaLipa.

This module contains extractors for various programming languages including Python,
JavaScript/TypeScript, and other code formats. It provides functionality for extracting 
code blocks from source files and analyzing their hierarchical structure.

Key Features:
1. AST-based Python extraction
2. Tree-sitter based JS/TS extraction
3. Pattern-based generic extraction
4. Block metadata and statistics
5. Code hierarchy analysis
6. Block relationship tracking

Dependencies:
- ast: For Python parsing (https://docs.python.org/3/library/ast.html)
- tree-sitter: For JS/TS parsing (https://tree-sitter.github.io/tree-sitter/)
- loguru: For logging (https://github.com/Delgan/loguru)

Related Files:
- python_extractor.py: AST-based Python extraction
- js_ts_extractor.py: Tree-sitter based JS/TS extraction
- generic_extractor.py: Pattern-based extraction
- hierarchy/: Code hierarchy analysis (refactored from hierarchy.py)

Architecture Notes:
- The module has been refactored to comply with the 500-line limit standard
- Hierarchy analysis has been moved to dedicated modules in hierarchy/
- See /hierarchy/TECHNICAL_DEBT.md for known limitations and future work
- For integration with QA module, see /docs/extraction_learnings.md
"""

import os
import textwrap
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info
from agent_tools.dualipa.extraction.extractors.utils.validation_utils import validate_block_format
from agent_tools.dualipa.extraction.extractors.utils.verification_utils import verify_code_block
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats

from .code_extractor import (
    extract_code_blocks,
    extract_python_blocks,
    extract_js_ts_blocks,
    extract_generic_blocks,
    validate_block,
    verify_block
)

from .hierarchy import (
    analyze_code_hierarchy,
    analyze_python_hierarchy,
    analyze_js_ts_hierarchy,
    analyze_generic_hierarchy
)

__all__ = [
    'extract_code_blocks',
    'extract_python_blocks',
    'extract_js_ts_blocks',
    'extract_generic_blocks',
    'validate_block',
    'verify_block',
    'analyze_code_hierarchy',
    'analyze_python_hierarchy',
    'analyze_js_ts_hierarchy',
    'analyze_generic_hierarchy'
]

def extract_code_blocks(
    file_path: str,
    language: Optional[str] = None,
    include_metadata: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract code blocks from a file using the appropriate extractor.
    
    Args:
        file_path: Path to source file
        language: Optional language override
        include_metadata: Whether to include block metadata
        
    Returns:
        Tuple of (extracted blocks, statistics)
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        # Detect language if not provided
        if language is None:
            language = detect_language(file_path)
            
        # Get language info
        info = get_language_info(language)
        if not info:
            logger.warning(f"Unsupported language: {language}")
            return [], stats
            
        # Select appropriate extractor
        if language == "python":
            blocks, block_stats = extract_python_blocks(file_path)
        elif language in ("javascript", "typescript"):
            blocks, block_stats = extract_js_ts_blocks(file_path)
        else:
            blocks, block_stats = extract_generic_blocks(file_path)
            
        # Update stats
        update_stats(stats, block_stats)
        
        # Validate blocks
        for block in blocks:
            if include_metadata:
                # Add file metadata
                block["metadata"].update({
                    "source_file": str(file_path),
                    "language": language
                })
                
            # Validate format
            errors = validate_block_format(block)
            if errors:
                logger.warning(f"Block validation errors: {errors}")
                stats["validation_errors"] += 1
                continue
                
            # Verify code
            is_valid, error = verify_code_block(block)
            if not is_valid:
                logger.warning(f"Block verification failed: {error}")
                stats["verification_errors"] += 1
                continue
                
        return blocks, stats
        
    except Exception as e:
        logger.error(f"Error extracting code blocks: {e}")
        return [], stats

def usage_example() -> None:
    """Example usage of code extraction."""
    # Example Python file
    python_content = """
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
        
    class Calculator:
        def add(self, x, y):
            return x + y
    """
    
    # Save to temp file
    with open("temp.py", "w") as f:
        f.write(python_content)
        
    # Extract blocks
    blocks, stats = extract_code_blocks("temp.py")
    
    print(f"Found {stats['total_blocks']} blocks:")
    for block in blocks:
        print(f"\nType: {block['type']}")
        print(f"Language: {block['metadata']['language']}")
        print("Content:")
        print(textwrap.indent(block['content'], "    "))
        
    # Cleanup
    os.remove("temp.py")
