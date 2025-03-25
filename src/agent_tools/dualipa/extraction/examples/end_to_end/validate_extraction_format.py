#!/usr/bin/env python3
"""
validate_extraction_format.py

This script validates the format of extraction blocks to ensure they follow
the expected schema for the DuaLipa extraction system.

Usage:
    python validate_extraction_format.py path/to/extraction_blocks.json

Example:
    python validate_extraction_format.py test_results/arangodb_blocks.json
"""

import sys
import json
import os
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union


def validate_block(block: Dict[str, Any], block_idx: int) -> List[str]:
    """
    Validate a single extraction block.
    
    Args:
        block: The block to validate
        block_idx: The index of the block in the array for error reporting
        
    Returns:
        A list of error messages, empty if the block is valid
    """
    errors = []
    
    # Check required fields
    required_fields = ["uuid", "type", "content"]
    for field in required_fields:
        if field not in block:
            errors.append(f"Block {block_idx} is missing required field: {field}")
    
    # Check type-specific requirements
    block_type = block.get("type")
    if block_type:
        # Parent-child relationship checks
        if block_type == "documentation":
            if "child_uuids" not in block:
                errors.append(f"Block {block_idx} (documentation) must have child_uuids")
                
        elif block_type == "doc_page":
            if "parent_uuid" not in block:
                errors.append(f"Block {block_idx} (doc_page) must have parent_uuid")
                
        elif block_type == "doc_section":
            if "parent_uuid" not in block:
                errors.append(f"Block {block_idx} (doc_section) must have parent_uuid")
                
        elif block_type in ["code_block", "table", "image"]:
            if "parent_uuid" not in block:
                errors.append(f"Block {block_idx} ({block_type}) must have parent_uuid")
                
        # Content type checks
        if block_type == "table" and not isinstance(block.get("content"), list):
            errors.append(f"Block {block_idx} (table) must have list content")
            
    # Check metadata
    if "metadata" in block and not isinstance(block["metadata"], dict):
        errors.append(f"Block {block_idx} metadata must be a dictionary")
    
    return errors


def validate_block_relationships(blocks: List[Dict[str, Any]]) -> List[str]:
    """
    Validate the relationships between blocks.
    
    Args:
        blocks: The list of blocks to validate
        
    Returns:
        A list of error messages, empty if all relationships are valid
    """
    errors = []
    
    # Build a map of UUID to block index for efficient lookup
    uuid_to_idx = {block["uuid"]: idx for idx, block in enumerate(blocks) if "uuid" in block}
    
    # Check parent-child relationships
    for idx, block in enumerate(blocks):
        # Check parent references
        if "parent_uuid" in block:
            parent_uuid = block["parent_uuid"]
            if parent_uuid not in uuid_to_idx:
                errors.append(f"Block {idx} references non-existent parent UUID: {parent_uuid}")
            else:
                # Check that parent has this child in its child_uuids list
                parent_idx = uuid_to_idx[parent_uuid]
                parent_block = blocks[parent_idx]
                if "child_uuids" in parent_block and block["uuid"] not in parent_block["child_uuids"]:
                    errors.append(f"Block {idx} parent (index {parent_idx}) does not list it as a child")
        
        # Check child references
        if "child_uuids" in block:
            child_uuids = block["child_uuids"]
            for child_uuid in child_uuids:
                if child_uuid not in uuid_to_idx:
                    errors.append(f"Block {idx} references non-existent child UUID: {child_uuid}")
                else:
                    # Check that child has this block as its parent
                    child_idx = uuid_to_idx[child_uuid]
                    child_block = blocks[child_idx]
                    if "parent_uuid" in child_block and child_block["parent_uuid"] != block["uuid"]:
                        errors.append(f"Block {idx} child (index {child_idx}) does not reference it as parent")
    
    return errors


def validate_extraction_blocks(blocks_file: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate extraction blocks from a JSON file.
    
    Args:
        blocks_file: Path to the JSON file containing blocks
        
    Returns:
        A tuple of (is_valid, error_messages, stats)
    """
    try:
        with open(blocks_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
        
        if not isinstance(blocks, list):
            return False, ["Blocks file must contain a JSON array"], {}
        
        # Validate individual blocks
        all_errors = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                all_errors.append(f"Block {idx} must be a JSON object")
                continue
                
            block_errors = validate_block(block, idx)
            all_errors.extend(block_errors)
        
        # Validate relationships between blocks
        relationship_errors = validate_block_relationships(blocks)
        all_errors.extend(relationship_errors)
        
        # Collect statistics
        stats = {
            "total_blocks": len(blocks),
            "block_types": {},
            "language_counts": {},
            "doc_types": {},
            "unique_parent_count": len(set(block["parent_uuid"] for block in blocks if "parent_uuid" in block)),
            "orphaned_blocks": len([b for b in blocks if "parent_uuid" not in b and b.get("type") != "documentation"]),
            "hierarchical_depth": calculate_hierarchical_depth(blocks),
        }
        
        # Count by block type
        for block in blocks:
            block_type = block.get("type", "unknown")
            stats["block_types"][block_type] = stats["block_types"].get(block_type, 0) + 1
            
            # Count by language
            language = block.get("language", "unknown")
            stats["language_counts"][language] = stats["language_counts"].get(language, 0) + 1
            
            # Count by doc_type
            doc_type = block.get("metadata", {}).get("doc_type", "unknown")
            stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1
        
        is_valid = len(all_errors) == 0
        return is_valid, all_errors, stats
    
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"], {}
    except Exception as e:
        return False, [f"Error validating blocks: {e}"], {}


def calculate_hierarchical_depth(blocks: List[Dict[str, Any]]) -> int:
    """
    Calculate the maximum hierarchical depth of blocks.
    
    Args:
        blocks: The list of blocks to analyze
        
    Returns:
        The maximum depth of the hierarchy
    """
    # Build a map of UUID to block
    uuid_to_block = {block["uuid"]: block for block in blocks if "uuid" in block}
    
    # Function to calculate depth of a block
    def get_depth(block_uuid, visited=None):
        if visited is None:
            visited = set()
        
        if block_uuid in visited:  # Handle cycles
            return 0
        
        visited.add(block_uuid)
        
        if block_uuid not in uuid_to_block:
            return 0
            
        block = uuid_to_block[block_uuid]
        
        if "parent_uuid" not in block:
            return 1  # Root level
            
        parent_uuid = block["parent_uuid"]
        return 1 + get_depth(parent_uuid, visited)
    
    # Calculate depth for each block
    depths = [get_depth(block["uuid"]) for block in blocks if "uuid" in block]
    
    # Return the maximum depth
    return max(depths) if depths else 0


def generate_html_report(blocks_file: Path, is_valid: bool, errors: List[str], stats: Dict[str, Any]) -> str:
    """
    Generate an HTML validation report.
    
    Args:
        blocks_file: Path to the blocks file
        is_valid: Whether the blocks are valid
        errors: List of validation errors
        stats: Block statistics
        
    Returns:
        HTML report as a string
    """
    # Load the blocks for creating examples
    try:
        with open(blocks_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
    except:
        blocks = []
    
    # Get sample blocks of different types
    sample_blocks = {}
    for block_type in ["documentation", "doc_page", "doc_section", "code_block", "table", "image"]:
        sample = next((b for b in blocks if b.get("type") == block_type), None)
        if sample:
            sample_blocks[block_type] = sample
    
    # Format block type counts
    block_type_rows = ""
    for block_type, count in sorted(stats.get("block_types", {}).items()):
        block_type_rows += f"""
        <tr>
            <td>{block_type}</td>
            <td>{count}</td>
            <td>{count / stats["total_blocks"] * 100:.1f}%</td>
        </tr>
        """
    
    # Format language counts
    language_rows = ""
    for language, count in sorted(stats.get("language_counts", {}).items()):
        language_rows += f"""
        <tr>
            <td>{language}</td>
            <td>{count}</td>
            <td>{count / stats["total_blocks"] * 100:.1f}%</td>
        </tr>
        """
    
    # Format doc type counts
    doc_type_rows = ""
    for doc_type, count in sorted(stats.get("doc_types", {}).items()):
        doc_type_rows += f"""
        <tr>
            <td>{doc_type}</td>
            <td>{count}</td>
            <td>{count / stats["total_blocks"] * 100:.1f}%</td>
        </tr>
        """
    
    # Format errors
    error_html = ""
    if errors:
        error_html = "<h3>Validation Errors</h3><ul>"
        for error in errors[:100]:  # Limit to first 100 errors
            error_html += f"<li>{error}</li>"
        if len(errors) > 100:
            error_html += f"<li>... and {len(errors) - 100} more errors</li>"
        error_html += "</ul>"
    
    # Create sample block HTML
    sample_blocks_html = ""
    for block_type, block in sample_blocks.items():
        sample_blocks_html += f"""
        <div class="sample-block">
            <h4>{block_type.replace('_', ' ').title()}</h4>
            <pre>{json.dumps(block, indent=2)}</pre>
        </div>
        """
    
    # Create the HTML report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extraction Format Validation: {blocks_file.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: {is_valid and '#e7f5e7' or '#f8e7e7'};
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid {is_valid and '#c3e6c3' or '#e6c3c3'};
        }}
        .valid-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 3px;
            color: white;
            background-color: {is_valid and '#28a745' or '#dc3545'};
            font-weight: bold;
        }}
        .section {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        .stat-card {{
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow: auto;
            font-size: 12px;
        }}
        .sample-blocks {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .sample-block {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            overflow: auto;
        }}
        .sample-block h4 {{
            margin-top: 0;
            color: #0066cc;
        }}
        h2 {{
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Extraction Format Validation</h1>
            <p>File: <strong>{blocks_file.name}</strong></p>
            <p>File size: <strong>{blocks_file.stat().st_size / 1024:.1f} KB</strong></p>
            <p>Validation time: <strong>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></p>
            <p>Status: <span class="valid-badge">{is_valid and 'VALID' or 'INVALID'}</span></p>
        </div>

        <div class="section">
            <h2>Extraction Statistics</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats.get("total_blocks", 0)}</div>
                    <div class="stat-label">Total Blocks</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{len(stats.get("block_types", {}))}</div>
                    <div class="stat-label">Block Types</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{len(stats.get("language_counts", {}))}</div>
                    <div class="stat-label">Languages</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{stats.get("hierarchical_depth", 0)}</div>
                    <div class="stat-label">Max Hierarchical Depth</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{stats.get("unique_parent_count", 0)}</div>
                    <div class="stat-label">Unique Parents</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-value">{stats.get("orphaned_blocks", 0)}</div>
                    <div class="stat-label">Orphaned Blocks</div>
                </div>
            </div>
            
            <h3>Block Types</h3>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {block_type_rows}
            </table>
            
            <h3>Languages</h3>
            <table>
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {language_rows}
            </table>
            
            <h3>Document Types</h3>
            <table>
                <tr>
                    <th>Document Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {doc_type_rows}
            </table>
        </div>
        
        {error_html}
        
        <div class="section">
            <h2>Sample Blocks</h2>
            <p>Examples of each block type found in the file:</p>
            
            <div class="sample-blocks">
                {sample_blocks_html}
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html


def create_extraction_dashboard(output_dir: Path):
    """
    Create an extraction dashboard HTML file with links to all validation reports.
    
    Args:
        output_dir: Directory containing validation reports
    """
    # Find all HTML report files
    reports = list(output_dir.glob("*.validation.html"))
    
    # Group by extraction type
    report_groups = {}
    for report in reports:
        # Try to determine the extraction type from filename
        name_parts = report.stem.split("_")
        if len(name_parts) > 0:
            extraction_type = name_parts[0]  # Use first part of filename
            if extraction_type not in report_groups:
                report_groups[extraction_type] = []
            report_groups[extraction_type].append(report)
    
    # Create links for each report
    report_sections = ""
    for group_name, group_reports in sorted(report_groups.items()):
        report_links = ""
        for report in sorted(group_reports, key=lambda p: p.stat().st_mtime, reverse=True):
            # Get file timestamp
            timestamp = datetime.datetime.fromtimestamp(report.stat().st_mtime)
            report_links += f"""
            <li>
                <a href="{report.name}">{report.name}</a>
                <span class="timestamp">{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </li>
            """
        
        report_sections += f"""
        <div class="report-section">
            <h2>{group_name.title()} Extraction Reports</h2>
            <ul class="report-list">
                {report_links}
            </ul>
        </div>
        """
    
    # Create dashboard HTML
    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DuaLipa Extraction Validation Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background-color: #4a6fa5;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        header h1 {{
            margin: 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-left: 10px;
        }}
        .report-section {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }}
        .report-list {{
            list-style-type: none;
            padding: 0;
        }}
        .report-list li {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .report-list li:last-child {{
            border-bottom: none;
        }}
        .report-list a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .report-list a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DuaLipa Extraction Validation Dashboard</h1>
            <p>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        {report_sections}
        
        <div class="report-section">
            <h2>Run Validation</h2>
            <p>To run validation on a new extraction file, use the following command:</p>
            <pre>python validate_extraction_format.py path/to/extraction_blocks.json --output-dir validation_reports</pre>
        </div>
    </div>
</body>
</html>
"""
    
    # Write dashboard HTML
    dashboard_path = output_dir / "extraction_dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    return dashboard_path


def validate_extraction_output(blocks: List[Dict[str, Any]], expected_format: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extraction output blocks against an expected format template.
    
    Args:
        blocks: A list of extraction blocks (dictionaries) to validate
        expected_format: A template/reference format to validate against
        
    Returns:
        Dictionary with validation results containing:
        - valid: Boolean indicating if validation passed
        - errors: List of error messages if validation failed
        - stats: Statistics about the blocks like counts by type, language, etc.
    """
    errors = []
    stats = {
        "total_blocks": len(blocks),
        "block_types": {},
        "language_counts": {},
        "doc_types": {},
        "unique_parent_count": len(set(block["parent_uuid"] for block in blocks if "parent_uuid" in block)),
        "orphaned_blocks": len([b for b in blocks if "parent_uuid" not in b and b.get("type") != "documentation"]),
        "hierarchical_depth": calculate_hierarchical_depth(blocks),
    }
    
    # Count by block type
    for block in blocks:
        block_type = block.get("type", "unknown")
        stats["block_types"][block_type] = stats["block_types"].get(block_type, 0) + 1
        
        # Count by language
        language = block.get("language", "unknown")
        stats["language_counts"][language] = stats["language_counts"].get(language, 0) + 1
        
        # Count by doc_type
        doc_type = block.get("metadata", {}).get("doc_type", "unknown")
        stats["doc_types"][doc_type] = stats["doc_types"].get(doc_type, 0) + 1
    
    # Check required block types
    required_types = expected_format.get("expected_structure", {}).get("required_block_types", [])
    actual_types = set(block["type"] for block in blocks)
    
    for req_type in required_types:
        if req_type not in actual_types:
            errors.append(f"Missing required block type: {req_type}")
    
    # Validate individual blocks
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            errors.append(f"Block {idx} must be a JSON object")
            continue
            
        block_errors = validate_block(block, idx)
        errors.extend(block_errors)
    
    # Validate relationships between blocks
    relationship_errors = validate_block_relationships(blocks)
    errors.extend(relationship_errors)
    
    # Check expected type counts
    expected_counts = expected_format.get("expected_structure", {}).get("expected_type_counts", {})
    for type_name, count_range in expected_counts.items():
        actual_count = stats["block_types"].get(type_name, 0)
        min_count = count_range.get("min", 0)
        max_count = count_range.get("max", float("inf"))
        
        if actual_count < min_count:
            errors.append(f"Too few {type_name} blocks: {actual_count} (expected at least {min_count})")
        elif max_count != float("inf") and actual_count > max_count:
            errors.append(f"Too many {type_name} blocks: {actual_count} (expected at most {max_count})")
    
    # Check any required parent-child relationships
    required_relationships = expected_format.get("expected_structure", {}).get("required_relationships", [])
    for relationship in required_relationships:
        parent_type = relationship.get("parent_type")
        child_type = relationship.get("child_type")
        
        # Find all parents of the specified type
        parent_blocks = [b for b in blocks if b.get("type") == parent_type]
        
        # Check each parent for at least one child of the required type
        for parent in parent_blocks:
            if "child_uuids" not in parent:
                errors.append(f"{parent_type} block {parent.get('uuid')} has no child_uuids")
                continue
                
            child_uuids = parent.get("child_uuids", [])
            child_blocks = [b for b in blocks if b.get("uuid") in child_uuids]
            
            if not any(b.get("type") == child_type for b in child_blocks):
                errors.append(f"{parent_type} block {parent.get('uuid')} has no child of type {child_type}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": stats
    }


def main():
    parser = argparse.ArgumentParser(description="Validate extraction blocks format")
    parser.add_argument("blocks_file", help="Path to the JSON file containing extraction blocks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed error messages")
    parser.add_argument("--output-dir", "-o", help="Directory to save HTML validation report")
    parser.add_argument("--dashboard", "-d", action="store_true", help="Create or update dashboard")
    parser.add_argument("--expected-format", "-e", help="Path to expected format JSON file")
    args = parser.parse_args()
    
    blocks_file = Path(args.blocks_file)
    if not blocks_file.exists():
        print(f"Error: File not found: {blocks_file}")
        sys.exit(1)
    
    # Get output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = blocks_file.parent / "validation_reports"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate blocks
    is_valid, errors, stats = validate_extraction_blocks(blocks_file)
    
    # If expected format is provided, validate against it
    if args.expected_format:
        expected_format_path = Path(args.expected_format)
        if expected_format_path.exists():
            try:
                with open(expected_format_path, 'r', encoding='utf-8') as f:
                    expected_format = json.load(f)
                
                # Load blocks for validation
                with open(blocks_file, 'r', encoding='utf-8') as f:
                    blocks = json.load(f)
                
                # Validate against expected format
                results = validate_extraction_output(blocks, expected_format)
                
                # Update validation results
                is_valid = is_valid and results["valid"]
                errors.extend(results["errors"])
                
                # Update stats with any additional information
                for key, value in results["stats"].items():
                    if key not in stats:
                        stats[key] = value
            except Exception as e:
                print(f"Error validating against expected format: {e}")
    
    # Generate HTML report
    html_report = generate_html_report(blocks_file, is_valid, errors, stats)
    
    # Save HTML report
    report_filename = f"{blocks_file.stem}.validation.html"
    report_path = output_dir / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # Create or update dashboard
    if args.dashboard:
        dashboard_path = create_extraction_dashboard(output_dir)
        print(f"📊 Updated dashboard: {dashboard_path}")
    
    # Print validation result
    if is_valid:
        print(f"✅ Validation passed: {blocks_file} contains valid extraction blocks")
        print(f"📋 Statistics: {stats.get('total_blocks', 0)} blocks, {len(stats.get('block_types', {}))} block types")
        print(f"📊 Report saved to: {report_path}")
        sys.exit(0)
    else:
        print(f"❌ Validation failed: {blocks_file} contains {len(errors)} errors")
        print(f"📊 Report saved to: {report_path}")
        if args.verbose:
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print(f"  Run with --verbose to see detailed error messages")
        sys.exit(1)


if __name__ == "__main__":
    main()