#!/usr/bin/env python3
"""
Hierarchy Validation and Reporting Script for DuaLipa Extractions.

This script demonstrates how to validate the parent-child relationships in
extraction outputs and generate a visual report for easy verification.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_hierarchy")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the hierarchy validator
from hierarchy_validator import (
    load_json_file,
    validate_parent_child_relationships,
    visualize_hierarchy,
    save_html_report
)


def main():
    """Main function demonstrating hierarchy validation."""
    parser = argparse.ArgumentParser(
        description="Validate and visualize parent-child relationships in extraction outputs"
    )
    parser.add_argument("--input", type=str,
                        default=str(Path(current_dir) / "test_results_dashboard/arangodb/arangodb_blocks.json"),
                        help="Path to the extraction JSON file")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(current_dir) / "hierarchy_validation"),
                        help="Directory to save validation results")
    parser.add_argument("--extraction-name", type=str,
                        default="arangodb",
                        help="Name of the extraction for report titles")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load extraction data
    logger.info(f"Loading extraction data from {input_path}")
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
    logger.info("Validating parent-child relationships")
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
    logger.info("Generating HTML visualization")
    html_content = visualize_hierarchy(blocks)
    
    # Save the HTML
    output_path = output_dir / f"{args.extraction_name}_hierarchy.html"
    if save_html_report(output_path, html_content):
        logger.info(f"HTML visualization saved to {output_path}")
        print(f"\nOpen the following file to view the hierarchy visualization:")
        print(f"  file://{output_path.absolute()}")
    else:
        logger.error("Failed to save HTML visualization")
    
    # Save validation results
    import json
    results_path = output_dir / f"{args.extraction_name}_validation.json"
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error saving validation results: {e}")
    
    # Create a summary HTML page
    summary_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{args.extraction_name} Validation Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ margin: 20px 0; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
        .stat-card {{ 
            padding: 15px; 
            border-radius: 5px; 
            background-color: #f5f5f5; 
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
        .stat-label {{ font-size: 14px; color: #666; }}
        .errors, .warnings {{ 
            margin: 20px 0; 
            padding: 10px; 
            border-radius: 5px;
        }}
        .errors {{ background-color: #ffeeee; }}
        .warnings {{ background-color: #ffffee; }}
        .error-title, .warning-title {{ font-weight: bold; margin-bottom: 10px; }}
        .error-title {{ color: red; }}
        .warning-title {{ color: orange; }}
        .links {{ margin: 20px 0; }}
        .links a {{ display: block; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{args.extraction_name} Hierarchy Validation Summary</h1>
    
    <div class="summary">
        <h2>Validation Result: 
            <span class="{'success' if validation_results['valid'] else 'failure'}">
                {'✅ PASSED' if validation_results['valid'] else '❌ FAILED'}
            </span>
        </h2>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{stats['total_blocks']}</div>
            <div class="stat-label">Total Blocks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['root_blocks']}</div>
            <div class="stat-label">Root Blocks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['child_blocks']}</div>
            <div class="stat-label">Child Blocks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['orphaned_blocks']}</div>
            <div class="stat-label">Orphaned Blocks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['bidirectional_references']}</div>
            <div class="stat-label">Bidirectional References</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['missing_references']}</div>
            <div class="stat-label">Missing References</div>
        </div>
    </div>
    
    {"" if not validation_results["errors"] else f'''
    <div class="errors">
        <div class="error-title">Errors ({len(validation_results["errors"])})</div>
        <ul>
            {"".join(f'<li>{error}</li>' for error in validation_results["errors"])}
        </ul>
    </div>
    '''}
    
    {"" if not validation_results["warnings"] else f'''
    <div class="warnings">
        <div class="warning-title">Warnings ({len(validation_results["warnings"])})</div>
        <ul>
            {"".join(f'<li>{warning}</li>' for warning in validation_results["warnings"])}
        </ul>
    </div>
    '''}
    
    <div class="links">
        <h3>Related Files</h3>
        <a href="{output_path.name}">View Complete Hierarchy Visualization</a>
        <a href="{results_path.name}">Download Validation Results (JSON)</a>
    </div>
</body>
</html>
"""
    
    summary_path = output_dir / f"{args.extraction_name}_summary.html"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_html)
        logger.info(f"Summary page saved to {summary_path}")
        print(f"\nOpen the following file to view the validation summary:")
        print(f"  file://{summary_path.absolute()}")
    except Exception as e:
        logger.error(f"Error saving summary page: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if validation_results["valid"] else 1)


if __name__ == "__main__":
    main()