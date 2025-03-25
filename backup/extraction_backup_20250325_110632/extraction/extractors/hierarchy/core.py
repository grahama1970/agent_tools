"""
Core hierarchy analysis functionality for DuaLipa.

This module provides the central functionality for code hierarchy analysis,
dispatching to language-specific analyzers and managing the process.

Key Features:
1. Language-specific analyzer dispatch
2. Statistics tracking
3. Hierarchy construction
4. Common utility functions

Dependencies:
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- typing: For type hints (https://docs.python.org/3/library/typing.html)
- loguru: For logging (https://github.com/Delgan/loguru)

Documentation Links:
- Pathlib: https://docs.python.org/3/library/pathlib.html
- Typing: https://docs.python.org/3/library/typing.html
- Loguru: https://loguru.readthedocs.io/

Input/Output Specifications:

analyze_code_hierarchy(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Input:
        - file_path: Path to source file
    Output:
        - Tuple containing:
            1. Hierarchy dictionary
            2. Statistics dictionary

build_code_hierarchy(blocks: List[Dict[str, Any]], source: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    Input:
        - blocks: List of code blocks
        - source: Optional source directory
        - output_dir: Optional output directory
    Output:
        - Statistics dictionary

Related Files:
- python/parser.py: Python-specific hierarchy analysis
- js_ts/parser.py: JavaScript/TypeScript-specific hierarchy analysis
- generic/parser.py: Generic language hierarchy analysis
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language, get_language_info
from agent_tools.dualipa.extraction.extractors.utils.stats_utils import init_stats, update_stats
from agent_tools.dualipa.extraction.extractors.utils.block_metadata import initialize_stats_dict

# Import language-specific analysis modules
from .python.parser import analyze_python_hierarchy
from .js_ts.parser import analyze_js_ts_hierarchy
from .generic.parser import analyze_generic_hierarchy

# Re-export initialize_stats_dict for backward compatibility
__all__ = ['analyze_code_hierarchy', 'build_code_hierarchy', 'initialize_stats_dict']


def analyze_code_hierarchy(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze code hierarchy in a file.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Tuple of (hierarchy info, statistics)
    """
    try:
        # Initialize stats
        stats = init_stats()
        
        # Detect language
        language = detect_language(file_path)
        if language == "unknown":
            stats["errors"].append(f"Unknown language for file: {file_path}")
            return {}, stats
            
        # Get language info
        info = get_language_info(language)
        if not info:
            stats["errors"].append(f"Unsupported language: {language}")
            return {}, stats
            
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Analyze based on language
        if language == "python":
            return analyze_python_hierarchy(content, file_path, stats)
        elif language in ("javascript", "typescript"):
            return analyze_js_ts_hierarchy(content, file_path, language, stats)
        else:
            return analyze_generic_hierarchy(content, file_path, language, stats)
            
    except Exception as e:
        logger.error(f"Error analyzing code hierarchy: {e}")
        return {}, stats


def build_code_hierarchy(blocks: List[Dict[str, Any]], source: Optional[Path] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Build a hierarchy of code blocks.
    
    This function processes code blocks and organizes them into a hierarchical
    structure based on their relationships.
    
    Args:
        blocks: List of code blocks
        source: Optional source directory
        output_dir: Optional output directory
        
    Returns:
        Statistics dictionary with hierarchy information
    """
    stats = initialize_stats_dict(source=source, output_dir=output_dir)
    
    try:
        # Skip empty blocks list
        if not blocks:
            stats["warnings"].append("No blocks provided for hierarchy building")
            return stats
            
        # Group blocks by file
        files = {}
        for block in blocks:
            file_path = block.get("file_path", "")
            if file_path:
                if file_path not in files:
                    files[file_path] = []
                files[file_path].append(block)
                
        # Build relationships
        for file_path, file_blocks in files.items():
            # Sort blocks by line number/position
            file_blocks.sort(key=lambda b: b.get("line_start", 0))
            
            # Identify parent-child relationships
            _build_relationships(file_blocks)
            
        # Update stats
        stats["files_processed"] = len(files)
        stats["blocks_processed"] = len(blocks)
        stats["hierarchies_built"] = len(files)
        
        return stats
    except Exception as e:
        logger.error(f"Error building code hierarchy: {e}")
        stats["errors"].append(f"Hierarchy building failed: {str(e)}")
        return stats


def _build_relationships(blocks: List[Dict[str, Any]]) -> None:
    """
    Build parent-child relationships between blocks.
    
    Args:
        blocks: List of blocks from the same file
    """
    if not blocks:
        return
        
    # Track parent stack
    stack = []
    
    for block in blocks:
        # Get block information
        block_type = block.get("type", "")
        depth = block.get("depth", 0)
        
        # Pop stack until we find the parent
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
            
        # Set parent-child relationship
        if stack:
            parent = stack[-1]
            block["parent_uuid"] = parent.get("uuid")
            
            # Add child to parent's child_uuids
            if "child_uuids" not in parent:
                parent["child_uuids"] = []
            if block.get("uuid") not in parent["child_uuids"]:
                parent["child_uuids"].append(block.get("uuid"))
                
        # Push current block onto stack
        stack.append(block)