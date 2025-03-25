#!/usr/bin/env python3
"""
Hierarchy Validator Tool for DuaLipa Documentation Extraction.

This tool provides a simple way to validate and visualize the parent-child 
relationships in documentation extraction outputs, making it easy to verify
structural consistency.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
import html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hierarchy_validator")


def load_json_file(file_path: Path) -> Any:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_html_report(output_path: Path, html_content: str) -> bool:
    """Save HTML report to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving HTML report: {e}")
        return False


def validate_parent_child_relationships(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parent-child relationships in the extraction outputs.
    
    Args:
        blocks: List of extracted blocks
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "total_blocks": len(blocks),
            "root_blocks": 0,
            "child_blocks": 0,
            "orphaned_blocks": 0,
            "bidirectional_references": 0,
            "missing_references": 0
        }
    }
    
    if not blocks:
        results["valid"] = False
        results["errors"].append("No blocks to validate")
        return results
    
    # Collect all UUIDs
    all_uuids = {block.get("uuid") for block in blocks if "uuid" in block}
    
    # Mapping from UUID to block
    uuid_to_block = {block.get("uuid"): block for block in blocks if "uuid" in block}
    
    # Track parent relationships
    parent_to_children = {}
    child_to_parent = {}
    
    # Collect declared parent-child relationships
    for block in blocks:
        uuid = block.get("uuid")
        if not uuid:
            results["warnings"].append(f"Block with no UUID: {block.get('name', 'unnamed')}")
            continue
            
        # Check parent references
        parent_uuid = block.get("parent_uuid")
        if parent_uuid:
            # Is this a valid parent?
            if parent_uuid in all_uuids:
                if parent_uuid not in parent_to_children:
                    parent_to_children[parent_uuid] = set()
                parent_to_children[parent_uuid].add(uuid)
                child_to_parent[uuid] = parent_uuid
                results["stats"]["child_blocks"] += 1
            else:
                results["errors"].append(f"Block {uuid} references non-existent parent {parent_uuid}")
                results["stats"]["missing_references"] += 1
        else:
            # This is a root block
            results["stats"]["root_blocks"] += 1
            
        # Check child references
        child_uuids = block.get("child_uuids", [])
        for child_uuid in child_uuids:
            if child_uuid in all_uuids:
                if uuid not in parent_to_children:
                    parent_to_children[uuid] = set()
                parent_to_children[uuid].add(child_uuid)
                if child_uuid in child_to_parent:
                    if child_to_parent[child_uuid] == uuid:
                        # Bidirectional reference confirmed
                        results["stats"]["bidirectional_references"] += 1
                    else:
                        # Child references a different parent!
                        results["errors"].append(
                            f"Child {child_uuid} references parent {child_to_parent[child_uuid]} "
                            f"but is claimed as child by {uuid}"
                        )
                else:
                    child_to_parent[child_uuid] = uuid
                    results["warnings"].append(
                        f"Child {child_uuid} doesn't reference parent {uuid} in its parent_uuid field"
                    )
            else:
                results["errors"].append(f"Block {uuid} references non-existent child {child_uuid}")
                results["stats"]["missing_references"] += 1
    
    # Check for orphaned blocks (blocks with no parent or children)
    orphaned_blocks = []
    for uuid in all_uuids:
        if uuid not in child_to_parent and uuid not in parent_to_children:
            orphaned_blocks.append(uuid)
            results["stats"]["orphaned_blocks"] += 1
    
    if orphaned_blocks:
        results["warnings"].append(f"Found {len(orphaned_blocks)} orphaned blocks with no parent or children")
    
    # Check for circular references
    circular_refs = find_circular_references(parent_to_children)
    if circular_refs:
        for ref_path in circular_refs:
            results["errors"].append(f"Circular reference detected: {' -> '.join(ref_path)}")
    
    # Overall validation result
    results["valid"] = len(results["errors"]) == 0
    
    return results


def find_circular_references(parent_to_children: Dict[str, Set[str]]) -> List[List[str]]:
    """Find circular references in the parent-child hierarchy."""
    circular_refs = []
    
    def dfs(node: str, path: List[str], visited: Set[str]):
        if node in path:
            # Found a circular reference
            cycle_start = path.index(node)
            circular_refs.append(path[cycle_start:] + [node])
            return
            
        if node in visited:
            return
            
        visited.add(node)
        path.append(node)
        
        for child in parent_to_children.get(node, set()):
            dfs(child, path.copy(), visited)
    
    # Start DFS from all nodes to find cycles
    visited = set()
    for node in parent_to_children:
        if node not in visited:
            dfs(node, [], visited)
    
    return circular_refs


def visualize_hierarchy(blocks: List[Dict[str, Any]]) -> str:
    """
    Create a visual HTML representation of the block hierarchy.
    
    Args:
        blocks: List of extracted blocks
        
    Returns:
        HTML string representing the hierarchy
    """
    # Mapping from UUID to block
    uuid_to_block = {block.get("uuid"): block for block in blocks if "uuid" in block}
    
    # Find root blocks (no parent_uuid)
    root_blocks = [block for block in blocks if not block.get("parent_uuid")]
    
    # Sort root blocks by name
    root_blocks.sort(key=lambda b: b.get("name", ""))
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Document Hierarchy Visualization</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 20px; }",
        "    h1 { color: #333; }",
        "    .hierarchy { margin-left: 20px; }",
        "    .block { margin: 5px 0; padding: 5px; border: 1px solid #ddd; border-radius: 4px; }",
        "    .block-header { display: flex; justify-content: space-between; }",
        "    .block-type { color: #666; }",
        "    .doc { background-color: #f8f8ff; }",
        "    .page { background-color: #fff8f8; }",
        "    .section { background-color: #f8fff8; }",
        "    .code { background-color: #fff8ff; }",
        "    .table { background-color: #fffff8; }",
        "    .children { margin-left: 20px; border-left: 1px solid #ccc; padding-left: 10px; }",
        "    .error { color: red; font-weight: bold; }",
        "    .warning { color: orange; }",
        "    .info { color: blue; }",
        "    .collapsible { cursor: pointer; }",
        "    .content { max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out; }",
        "    .active .content { max-height: 500px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Document Hierarchy Visualization</h1>",
        "  <div class='summary'>",
        f"    <p><strong>Total Blocks:</strong> {len(blocks)}</p>",
        f"    <p><strong>Root Blocks:</strong> {len(root_blocks)}</p>",
        "  </div>",
        "  <div class='hierarchy'>"
    ]
    
    # Recursively build the hierarchy
    def render_block(block: Dict[str, Any], depth: int = 0) -> List[str]:
        if not block:
            return ["<div class='error'>Missing block</div>"]
            
        uuid = block.get("uuid", "unknown")
        name = block.get("name", "Unnamed Block")
        block_type = block.get("type", "unknown")
        
        # Determine CSS class based on block type
        css_class = "block "
        if "documentation" in block_type:
            css_class += "doc"
        elif "page" in block_type:
            css_class += "page"
        elif "section" in block_type:
            css_class += "section"
        elif "code" in block_type:
            css_class += "code"
        elif "table" in block_type:
            css_class += "table"
        
        # Create block header
        parts = [
            f"<div class='{css_class}'>",
            f"  <div class='block-header'>",
            f"    <span class='collapsible'>{html.escape(name)}</span>",
            f"    <span class='block-type'>{block_type}</span>",
            f"  </div>",
            f"  <div class='content'>",
            f"    <p><strong>UUID:</strong> {uuid}</p>"
        ]
        
        # Add content preview if available
        content = block.get("content", "")
        if content:
            # Handle different content types (string, list, etc.)
            if isinstance(content, str):
                content_preview = content[:100] + "..." if len(content) > 100 else content
                parts.append(f"    <p><strong>Content:</strong> {html.escape(content_preview)}</p>")
            elif isinstance(content, list):
                parts.append(f"    <p><strong>Content:</strong> [List with {len(content)} items]</p>")
            else:
                parts.append(f"    <p><strong>Content:</strong> {type(content).__name__}</p>")
        
        # Check parent reference
        parent_uuid = block.get("parent_uuid")
        if parent_uuid:
            parent_block = uuid_to_block.get(parent_uuid)
            parent_name = parent_block.get("name", "Unknown") if parent_block else "Missing Parent"
            parent_class = "" if parent_block else "error"
            parts.append(f"    <p><strong>Parent:</strong> <span class='{parent_class}'>{html.escape(parent_name)} ({parent_uuid})</span></p>")
        
        # Check for metadata
        metadata = block.get("metadata", {})
        if metadata:
            parts.append(f"    <p><strong>Metadata:</strong> {len(metadata)} fields</p>")
        
        parts.append("  </div>")
        
        # Render children if any
        child_uuids = block.get("child_uuids", [])
        if child_uuids:
            parts.append("  <div class='children'>")
            
            # Get all children blocks for the current block
            children = []
            missing_children = []
            
            for child_uuid in child_uuids:
                child_block = uuid_to_block.get(child_uuid)
                if child_block:
                    children.append(child_block)
                else:
                    missing_children.append(child_uuid)
            
            # Sort children by name
            children.sort(key=lambda b: b.get("name", ""))
            
            # Render each child
            for child in children:
                parts.extend(render_block(child, depth + 1))
            
            # Report missing children
            for missing_uuid in missing_children:
                parts.append(f"<div class='block error'>Missing Child: {missing_uuid}</div>")
            
            parts.append("  </div>")
        
        parts.append("</div>")
        return parts
    
    # Render the hierarchy starting from root blocks
    for root_block in root_blocks:
        html_parts.extend(render_block(root_block))
    
    # Add JavaScript for collapsible sections
    html_parts.extend([
        "  </div>",
        "  <script>",
        "    document.addEventListener('DOMContentLoaded', function() {",
        "      var collapsibles = document.getElementsByClassName('collapsible');",
        "      for (var i = 0; i < collapsibles.length; i++) {",
        "        collapsibles[i].addEventListener('click', function() {",
        "          this.parentElement.parentElement.classList.toggle('active');",
        "        });",
        "      }",
        "    });",
        "  </script>",
        "</body>",
        "</html>"
    ])
    
    return "\n".join(html_parts)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate and visualize extraction hierarchy")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the extraction JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the HTML visualization (optional)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Load extraction data
    extraction_data = load_json_file(input_path)
    if not extraction_data:
        logger.error("Failed to load extraction data")
        sys.exit(1)
    
    # Ensure we have a list of blocks
    blocks = extraction_data
    if isinstance(extraction_data, dict) and "sections" in extraction_data:
        blocks = extraction_data["sections"]
    
    logger.info(f"Loaded {len(blocks)} blocks from {input_path}")
    
    # Validate parent-child relationships
    validation_results = validate_parent_child_relationships(blocks)
    
    # Print validation results
    if validation_results["valid"]:
        logger.info("✅ Hierarchy validation successful")
    else:
        logger.error("❌ Hierarchy validation failed")
        for error in validation_results["errors"]:
            logger.error(f"  - {error}")
    
    for warning in validation_results["warnings"]:
        logger.warning(f"  - {warning}")
    
    # Print statistics
    stats = validation_results["stats"]
    logger.info("Hierarchy Statistics:")
    logger.info(f"  Total Blocks: {stats['total_blocks']}")
    logger.info(f"  Root Blocks: {stats['root_blocks']}")
    logger.info(f"  Child Blocks: {stats['child_blocks']}")
    logger.info(f"  Orphaned Blocks: {stats['orphaned_blocks']}")
    logger.info(f"  Bidirectional References: {stats['bidirectional_references']}")
    logger.info(f"  Missing References: {stats['missing_references']}")
    
    # Generate HTML visualization
    html_content = visualize_hierarchy(blocks)
    
    # Save or display the HTML
    if args.output:
        output_path = Path(args.output)
        if save_html_report(output_path, html_content):
            logger.info(f"HTML visualization saved to {output_path}")
            print(f"\nOpen the following file to view the hierarchy visualization:")
            print(f"  file://{output_path.absolute()}")
        else:
            logger.error("Failed to save HTML visualization")
    else:
        # Create a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(html_content.encode('utf-8'))
        
        logger.info(f"HTML visualization saved to temporary file: {temp_path}")
        print(f"\nOpen the following file to view the hierarchy visualization:")
        print(f"  file://{temp_path.absolute()}")
    
    # Exit with appropriate code
    sys.exit(0 if validation_results["valid"] else 1)


if __name__ == "__main__":
    main()