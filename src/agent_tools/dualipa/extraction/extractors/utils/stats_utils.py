"""
Statistics tracking utilities for DuaLipa.

This module handles tracking and aggregating statistics for code
and content extraction, including file counts, block types,
language distribution, and errors.

Key Features:
1. Statistics initialization
2. Block counting and aggregation
3. Language distribution tracking
4. Error collection and reporting

Dependencies:
- loguru: For logging

Related Files:
- code_extractor.py: Uses stats for code blocks
- markdown_extractor.py: Uses stats for markdown blocks
- repo_utils.py: Uses stats for repository analysis
"""

from typing import Dict, List, Any
from loguru import logger

def init_stats() -> Dict[str, Any]:
    """
    Initialize statistics dictionary.
    
    Returns:
        Empty statistics dictionary
    """
    return {
        "total_files": 0,
        "total_blocks": 0,
        "languages": {},
        "block_types": {},
        "errors": [],
        "classes": 0,
        "functions": 0,
        "imports": 0,
        "file_blocks": {}
    }

def update_stats(
    stats: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    language: str
) -> None:
    """
    Update statistics with extracted blocks.
    
    Args:
        stats: Statistics dictionary
        blocks: List of extracted blocks
        language: Source language
    """
    try:
        # Update file count
        stats["total_files"] = stats.get("total_files", 0) + 1
        
        # Update language stats
        if language not in stats["languages"]:
            stats["languages"][language] = {
                "files": 0,
                "blocks": 0,
                "block_types": {}
            }
        stats["languages"][language]["files"] += 1
        
        # Process blocks
        for block in blocks:
            # Update total blocks
            stats["total_blocks"] = stats.get("total_blocks", 0) + 1
            
            # Update language block count
            stats["languages"][language]["blocks"] += 1
            
            # Update block type stats
            block_type = block.get("type", "unknown")
            
            # Global block types
            if block_type not in stats["block_types"]:
                stats["block_types"][block_type] = 0
            stats["block_types"][block_type] += 1
            
            # Language-specific block types
            lang_stats = stats["languages"][language]
            if block_type not in lang_stats["block_types"]:
                lang_stats["block_types"][block_type] = 0
            lang_stats["block_types"][block_type] += 1
            
    except Exception as e:
        logger.error(f"Error updating statistics: {e}")
        stats["errors"].append(str(e))

def merge_stats(stats1: Dict[str, Any], stats2: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Merge two statistics dictionaries.
    
    Args:
        stats1: First statistics dictionary
        stats2: Second statistics dictionary (optional)
        
    Returns:
        Merged statistics
    """
    try:
        # Initialize merged stats
        merged = init_stats()
        
        # If stats2 is None, just return a copy of stats1
        if stats2 is None:
            return stats1.copy()
            
        # List of stats to merge
        stats_list = [stats1, stats2]
        
        # Merge each stats dict
        for stats in stats_list:
            # Update totals
            merged["total_files"] += stats.get("total_files", 0)
            merged["total_blocks"] += stats.get("total_blocks", 0)
            merged["classes"] += stats.get("classes", 0)
            merged["functions"] += stats.get("functions", 0)
            merged["imports"] += stats.get("imports", 0)
            
            # Merge file_blocks
            for file_path, blocks in stats.get("file_blocks", {}).items():
                if file_path in merged["file_blocks"]:
                    merged["file_blocks"][file_path].extend(blocks)
                else:
                    merged["file_blocks"][file_path] = blocks.copy()
            
            # Merge languages
            for lang, lang_stats in stats.get("languages", {}).items():
                if lang not in merged["languages"]:
                    merged["languages"][lang] = {
                        "files": 0,
                        "blocks": 0,
                        "block_types": {}
                    }
                    
                # Update language stats
                lang_merged = merged["languages"][lang]
                lang_merged["files"] += lang_stats.get("files", 0)
                lang_merged["blocks"] += lang_stats.get("blocks", 0)
                
                # Merge block types
                for block_type, count in lang_stats.get("block_types", {}).items():
                    lang_merged["block_types"][block_type] = (
                        lang_merged["block_types"].get(block_type, 0) + count
                    )
                    
            # Merge block types
            for block_type, count in stats.get("block_types", {}).items():
                merged["block_types"][block_type] = (
                    merged["block_types"].get(block_type, 0) + count
                )
                
            # Merge errors
            merged["errors"].extend(stats.get("errors", []))
            
            # Copy any additional fields
            for key, value in stats.items():
                if key not in merged and key not in ["total_files", "total_blocks", "languages", 
                                                  "block_types", "errors", "classes", 
                                                  "functions", "imports", "file_blocks"]:
                    merged[key] = value
            
        return merged
        
    except Exception as e:
        logger.error(f"Error merging statistics: {e}")
        return init_stats()

def format_stats(stats: Dict[str, Any]) -> str:
    """
    Format statistics as human-readable string.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        Formatted statistics string
    """
    try:
        lines = []
        
        # Add summary
        lines.append("Extraction Statistics:")
        lines.append(f"Total Files: {stats['total_files']}")
        lines.append(f"Total Blocks: {stats['total_blocks']}")
        
        # Add language stats
        lines.append("\nLanguage Distribution:")
        for lang, lang_stats in sorted(stats["languages"].items()):
            lines.append(f"\n{lang}:")
            lines.append(f"  Files: {lang_stats['files']}")
            lines.append(f"  Blocks: {lang_stats['blocks']}")
            
            # Add block types
            if lang_stats["block_types"]:
                lines.append("  Block Types:")
                for block_type, count in sorted(lang_stats["block_types"].items()):
                    lines.append(f"    {block_type}: {count}")
                    
        # Add errors if any
        if stats["errors"]:
            lines.append("\nErrors:")
            for error in stats["errors"]:
                lines.append(f"  {error}")
                
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Error formatting statistics: {e}")
        return "Error formatting statistics"

def usage_example() -> None:
    """Example usage of statistics utilities."""
    # Initialize stats
    stats = init_stats()
    
    # Example blocks
    python_blocks = [
        {"type": "function", "name": "example_func"},
        {"type": "class", "name": "ExampleClass"},
        {"type": "method", "name": "example_method"}
    ]
    
    typescript_blocks = [
        {"type": "interface", "name": "ExampleInterface"},
        {"type": "class", "name": "ExampleComponent"},
        {"type": "method", "name": "render"}
    ]
    
    # Update stats
    update_stats(stats, python_blocks, "python")
    update_stats(stats, typescript_blocks, "typescript")
    
    # Create second stats dict
    stats2 = init_stats()
    markdown_blocks = [
        {"type": "section", "title": "Introduction"},
        {"type": "code", "language": "python"}
    ]
    update_stats(stats2, markdown_blocks, "markdown")
    
    # Merge stats
    merged = merge_stats(stats, stats2)
    
    # Print formatted stats
    print(format_stats(merged)) 