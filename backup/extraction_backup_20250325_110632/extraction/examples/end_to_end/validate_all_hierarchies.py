#!/usr/bin/env python3
"""
Batch Hierarchy Validation Tool for All Extractions.

This script finds and validates all extraction outputs, generating visual reports
to make it easy to verify parent-child relationships across all documents.
"""

import os
import sys
import glob
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_all_hierarchies")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def find_extraction_files() -> list:
    """Find all extraction block files in the project."""
    extraction_files = []
    
    # Define patterns to search for
    patterns = [
        "**/arangodb_blocks.json",
        "**/readthedocs_blocks.json",
        "*_function_extraction.json"
    ]
    
    # Search for files matching patterns
    for pattern in patterns:
        files = glob.glob(os.path.join(current_dir, pattern), recursive=True)
        extraction_files.extend(files)
    
    return extraction_files


def get_extraction_name(file_path: str) -> str:
    """Extract a friendly name from the file path."""
    path = Path(file_path)
    
    # Extract the test result directory name
    if "test_results" in str(path):
        parent_parts = path.parent.parts
        for part in parent_parts:
            if "test_results" in part:
                parent_dir = part
                break
        else:
            parent_dir = path.parent.name
    else:
        parent_dir = "default"
    
    # Extract the base extraction type from filename
    if "arangodb" in path.name:
        extraction_type = "arangodb"
    elif "readthedocs" in path.name:
        extraction_type = "readthedocs"
    elif "length" in path.name:
        extraction_type = "length_function"
    elif "array" in path.name:
        extraction_type = "array_function"
    else:
        extraction_type = path.stem
    
    # Combine for a unique name
    return f"{extraction_type}_{parent_dir}"


def create_index_page(output_dir: Path, validations: list) -> Path:
    """Create an index page linking to all validation reports."""
    # Sort validations by extraction name
    validations.sort(key=lambda v: v["name"])
    
    # Count valid and invalid validations
    valid_count = sum(1 for v in validations if v["valid"])
    invalid_count = len(validations) - valid_count
    
    # Create table rows
    table_rows = []
    for validation in validations:
        name = validation["name"]
        valid = validation["valid"]
        
        # Get statistics
        stats = validation.get("stats", {})
        total_blocks = stats.get("total_blocks", 0)
        root_blocks = stats.get("root_blocks", 0)
        child_blocks = stats.get("child_blocks", 0)
        
        # Get errors and warnings
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])
        
        # Create status cell
        status_class = "status-valid" if valid else "status-invalid"
        status_text = "✅ VALID" if valid else "❌ INVALID"
        
        # Create links cell
        summary_link = f'<a href="{name}_summary.html">Summary</a>'
        hierarchy_link = f'<a href="{name}_hierarchy.html">Hierarchy</a>'
        json_link = f'<a href="{name}_validation.json">JSON</a>'
        links = f"{summary_link} | {hierarchy_link} | {json_link}"
        
        # Create table row
        row = f"""
            <tr>
                <td>{name}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{total_blocks}</td>
                <td>{root_blocks}</td>
                <td>{child_blocks}</td>
                <td>{len(errors)}</td>
                <td>{len(warnings)}</td>
                <td>{links}</td>
            </tr>
        """
        table_rows.append(row)
    
    # Create the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DuaLipa Extraction Hierarchy Validation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table th, .results-table td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .results-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .results-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .results-table tr:hover {{
            background-color: #f1f1f1;
        }}
        .status-valid {{
            color: green;
            font-weight: bold;
        }}
        .status-invalid {{
            color: red;
            font-weight: bold;
        }}
        .summary {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }}
        .summary-item {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <h1>DuaLipa Extraction Hierarchy Validation</h1>
    
    <div class="summary">
        <div class="summary-item"><strong>Total Validations:</strong> {len(validations)}</div>
        <div class="summary-item"><strong>Valid:</strong> <span class="status-valid">{valid_count}</span></div>
        <div class="summary-item"><strong>Invalid:</strong> <span class="status-invalid">{invalid_count}</span></div>
        <div class="summary-item"><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
    
    <table class="results-table">
        <thead>
            <tr>
                <th>Extraction</th>
                <th>Status</th>
                <th>Blocks</th>
                <th>Root Blocks</th>
                <th>Child Blocks</th>
                <th>Errors</th>
                <th>Warnings</th>
                <th>Links</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""
    
    # Save the index page
    index_path = output_dir / "index.html"
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Created index page at {index_path}")
        return index_path
    except Exception as e:
        logger.error(f"Error creating index page: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate parent-child relationships in all extraction outputs"
    )
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(current_dir) / "hierarchy_validation"),
                        help="Directory to save validation results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all extraction files
    logger.info("Finding extraction files")
    extraction_files = find_extraction_files()
    logger.info(f"Found {len(extraction_files)} extraction files")
    
    # Track unique names to avoid duplicates
    seen_names = set()
    
    # Validate each extraction
    all_validations = []
    for file_path in extraction_files:
        extraction_name = get_extraction_name(file_path)
        
        # Skip duplicate names (prefer first found)
        if extraction_name in seen_names:
            logger.info(f"Skipping duplicate: {extraction_name} from {file_path}")
            continue
            
        seen_names.add(extraction_name)
        logger.info(f"Validating {extraction_name} from {file_path}")
        
        # Run the validate_hierarchy.py script
        cmd = [
            sys.executable,
            os.path.join(current_dir, "validate_hierarchy.py"),
            "--input", file_path,
            "--output-dir", str(output_dir),
            "--extraction-name", extraction_name
        ]
        
        try:
            # Run the validation script
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load the validation results
            results_path = output_dir / f"{extraction_name}_validation.json"
            if results_path.exists():
                with open(results_path, 'r', encoding='utf-8') as f:
                    validation_results = json.load(f)
                    validation_results["name"] = extraction_name
                    all_validations.append(validation_results)
                    logger.info(f"  {'✅ Valid' if validation_results['valid'] else '❌ Invalid'} - {extraction_name}")
            else:
                logger.error(f"  No validation results found for {extraction_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"  Validation failed for {extraction_name}: {e}")
    
    # Create index page
    if all_validations:
        index_path = create_index_page(output_dir, all_validations)
        if index_path:
            print(f"\nValidation of {len(all_validations)} extractions complete.")
            print(f"Open the following file to view the validation results:")
            print(f"  file://{index_path.absolute()}")
    else:
        logger.error("No validations were completed successfully")
    
    # Return success if all validations passed
    all_valid = all(v.get("valid", False) for v in all_validations)
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())