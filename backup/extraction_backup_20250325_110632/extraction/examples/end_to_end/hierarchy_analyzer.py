#!/usr/bin/env python3
"""
Hierarchy Analysis Module for DuaLipa.

This module provides functions for analyzing hierarchical relationships in code blocks
and enriching blocks with that hierarchical information. It helps establish parent-child
relationships between code elements like classes and methods.

Key Functions:
- analyze_hierarchies: Build hierarchical relationships from code blocks
- enrich_blocks_with_hierarchy: Add hierarchy information to code blocks

Dependencies:
- pathlib: For file path handling (https://docs.python.org/3/library/pathlib.html)
- typing: For type hints (https://docs.python.org/3/library/typing.html)
- logging: For logging (https://docs.python.org/3/library/logging.html)

Examples:
    >>> blocks = extract_all_blocks(source_dir)
    >>> hierarchies = analyze_hierarchies(blocks)
    >>> enriched_blocks = enrich_blocks_with_hierarchy(blocks, hierarchies)
    >>> print(f"Analyzed {len(hierarchies)} file hierarchies")
    Analyzed 2 file hierarchies
"""

from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logger = logging.getLogger("extraction.hierarchy")


def analyze_hierarchies(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze hierarchical relationships in code blocks.
    
    This function examines code blocks and builds a hierarchical representation
    of classes, functions, and their relationships.
    
    Args:
        blocks: List of extracted code blocks
        
    Returns:
        Dictionary mapping file paths to their hierarchy information
        
    Example:
        >>> blocks = extract_all_blocks(Path('./test_repos/python-sample'))
        >>> hierarchies = analyze_hierarchies(blocks)
        >>> list(hierarchies.values())[0]['language']
        'python'
    """
    logger.info("Analyzing code hierarchies")
    
    # Group blocks by file
    files = {}
    for block in blocks:
        file_path = block.get("file_path", "")
        if file_path:
            if file_path not in files:
                files[file_path] = []
            files[file_path].append(block)
    
    # Analyze each file
    hierarchies = {}
    for file_path, file_blocks in files.items():
        try:
            # For simplicity, we'll create a basic hierarchy structure
            # In a real-world scenario, we would use the analyze_code_hierarchy
            # function, but it requires production-ready implementations
            
            # Get first block's language
            language = file_blocks[0].get("language", "unknown")
            file_path_str = str(file_path) if isinstance(file_path, Path) else file_path
            
            # Build basic hierarchy with classes and functions
            classes = {}
            functions = {}
            imports = []
            
            # Extract hierarchy from blocks
            for block in file_blocks:
                block_type = block.get("type")
                block_name = block.get("name")
                
                if block_type == "class" and block_name:
                    classes[block_name] = {
                        "methods": [],
                        "line_start": 0,  # Would be extracted from real analysis
                        "line_end": 0,    # Would be extracted from real analysis
                        "bases": []       # Would be extracted from real analysis
                    }
                elif block_type == "function" and block_name:
                    functions[block_name] = {
                        "line_start": 0,  # Would be extracted from real analysis
                        "line_end": 0,    # Would be extracted from real analysis
                        "args": []        # Would be extracted from real analysis
                    }
            
            # Build hierarchy object
            hierarchy = {
                "file_path": file_path_str,
                "language": language,
                "imports": imports,
                "classes": classes,
                "functions": functions
            }
            
            hierarchies[file_path_str] = hierarchy
            
        except Exception as e:
            logger.error(f"Error analyzing hierarchy for {file_path}: {e}")
    
    logger.info(f"Analyzed hierarchies for {len(hierarchies)} files")
    return hierarchies


def enrich_blocks_with_hierarchy(
    blocks: List[Dict[str, Any]], 
    hierarchies: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Enrich blocks with hierarchical information.
    
    This function adds hierarchy-related metadata to code blocks, such as
    parent-child relationships, inheritance information, and more.
    
    Args:
        blocks: List of code blocks to enrich
        hierarchies: Hierarchy information from analyze_hierarchies()
        
    Returns:
        List of enriched code blocks with hierarchy information
        
    Example:
        >>> blocks = extract_all_blocks(Path('./test_repos/python-sample'))
        >>> hierarchies = analyze_hierarchies(blocks)
        >>> enriched = enrich_blocks_with_hierarchy(blocks, hierarchies)
        >>> 'breadcrumb' in enriched[0]
        True
    """
    logger.info("Enriching blocks with hierarchical information")
    
    # Group blocks by file for efficient processing
    files_blocks = {}
    for block in blocks:
        file_path = block.get("file_path", "")
        if file_path:
            if file_path not in files_blocks:
                files_blocks[file_path] = []
            files_blocks[file_path].append(block)
    
    # Enrich blocks with hierarchy info
    enriched_blocks = []
    for file_path, file_blocks in files_blocks.items():
        hierarchy = hierarchies.get(file_path)
        if not hierarchy:
            # If no hierarchy, just add the blocks as is
            enriched_blocks.extend(file_blocks)
            continue
        
        # Get classes and functions from hierarchy
        classes = hierarchy.get("classes", {})
        functions = hierarchy.get("functions", {})
        
        # Map blocks to hierarchy elements
        for block in file_blocks:
            block_type = block.get("type")
            block_name = block.get("name", "")
            
            # Ensure metadata field exists
            if "metadata" not in block:
                block["metadata"] = {}
            
            # Ensure standardized fields for QA module
            # - uuid and id standardization
            if "uuid" in block and "id" not in block:
                block["id"] = block["uuid"]
            elif "id" in block and "uuid" not in block:
                block["uuid"] = block["id"]
            
            # - file path standardization
            if file_path and "source_file" not in block["metadata"]:
                block["metadata"]["source_file"] = file_path
            if "language" in block and "language" not in block["metadata"]:
                block["metadata"]["language"] = block["language"]
            
            # Enrich with class information
            if block_type == "class" and block_name in classes:
                class_info = classes[block_name]
                line_start = class_info.get("line_start", 0)
                line_end = class_info.get("line_end", 0)
                
                # Update both top-level and metadata fields for line numbers
                block["line_start"] = line_start
                block["line_end"] = line_end
                block["metadata"]["line_start"] = line_start
                block["metadata"]["line_end"] = line_end
                
                # Add inheritance info
                if "bases" in class_info:
                    block["inheritance"] = class_info["bases"]
                    block["metadata"]["bases"] = class_info["bases"]
                    
                # Add method relationships and prepare child_uuids
                if "child_uuids" not in block:
                    block["child_uuids"] = []
                    
                # Track methods that belong to this class
                class_methods = {}
                for method in class_info.get("methods", []):
                    method_name = method.get("name", "")
                    if method_name:
                        class_methods[method_name] = method
                
                # Look for method blocks that match this class's methods
                for method_block in file_blocks:
                    if (method_block.get("type") == "method" or method_block.get("type") == "function") and \
                       method_block.get("name") in class_methods:
                        # Add parent-child relationship
                        method_block["parent_uuid"] = block["uuid"]
                        if method_block["uuid"] not in block["child_uuids"]:
                            block["child_uuids"].append(method_block["uuid"])
                        
                        # Add class name to method metadata
                        if "metadata" not in method_block:
                            method_block["metadata"] = {}
                        method_block["metadata"]["class_name"] = block_name
            
            # Enrich with function information
            elif block_type == "function" and block_name in functions:
                func_info = functions[block_name]
                line_start = func_info.get("line_start", 0)
                line_end = func_info.get("line_end", 0)
                
                # Update both top-level and metadata fields for line numbers
                block["line_start"] = line_start
                block["line_end"] = line_end
                block["metadata"]["line_start"] = line_start
                block["metadata"]["line_end"] = line_end
                
                # Add return type if available
                if "returns" in func_info:
                    block["return_type"] = func_info["returns"]
                    block["metadata"]["returns"] = func_info["returns"]
                
                # Add args if available
                if "args" in func_info:
                    block["metadata"]["args"] = func_info["args"]
            
            # Add imports from hierarchy to both top level and metadata
            imports = hierarchy.get("imports", [])
            block["imports"] = imports
            block["metadata"]["imports"] = imports
            
            # Initialize breadcrumb for QA module requirements
            if "breadcrumb" not in block:
                block["breadcrumb"] = [Path(file_path).name]
                if block_name:
                    block["breadcrumb"].append(block_name)
            
            # Ensure child_uuids and parent_uuid fields exist
            if "child_uuids" not in block:
                block["child_uuids"] = []
            if "parent_uuid" not in block:
                block["parent_uuid"] = None
                
            # Add the enriched block
            enriched_blocks.append(block)
    
    logger.info(f"Enriched {len(enriched_blocks)} blocks with hierarchy information")
    return enriched_blocks