#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite for DuaLipa Documentation Extraction.

This script orchestrates testing across multiple documentation sources and
creates a comprehensive report validating extraction quality for each source.
It integrates with the transparent testing framework for easy visual verification.
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validation_suite")

# Add the parent directory to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import validation functions
from validation import (
    validate_extraction_result, 
    load_expected_format,
    save_validation_results
)
from convert_for_validation import convert_to_validation_format
from format_adapter import adapt_extraction_to_validation_format


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: Path, indent: int = 2) -> bool:
    """Save data to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"Saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def detect_doc_type(extraction_data: List[Dict[str, Any]]) -> str:
    """
    Automatically detect the type of documentation in the extraction data.
    
    Args:
        extraction_data: The raw extraction data
        
    Returns:
        String identifying the document type: "arangodb", "readthedocs", "markdown", or "generic"
    """
    # Check if this is an ArangoDB documentation
    arangodb_indicators = ["arangodb.com", "aql", "ArangoDB", "Arango"]
    
    # Check if this is a ReadTheDocs documentation
    readthedocs_indicators = ["readthedocs.io", "readthedocs.org", "sphinx"]
    
    # Count occurrences of each indicator
    arangodb_count = 0
    readthedocs_count = 0
    markdown_count = 0
    
    for block in extraction_data:
        content = block.get("content", "")
        metadata = block.get("metadata", {})
        source_url = metadata.get("source_url", "") or block.get("source_url", "")
        
        # Check for ArangoDB indicators
        for indicator in arangodb_indicators:
            if indicator in content or indicator in source_url:
                arangodb_count += 1
        
        # Check for ReadTheDocs indicators
        for indicator in readthedocs_indicators:
            if indicator in content or indicator in source_url:
                readthedocs_count += 1
        
        # Check for Markdown indicators
        if metadata.get("language") == "markdown" or block.get("language") == "markdown":
            markdown_count += 1
    
    # Determine the document type based on counts
    if arangodb_count > 2:
        return "arangodb"
    elif readthedocs_count > 2:
        return "readthedocs"
    elif markdown_count > len(extraction_data) / 2:
        return "markdown"
    else:
        return "generic"


def get_template_path(doc_type: str) -> Optional[Path]:
    """
    Get the appropriate template path based on document type.
    
    Args:
        doc_type: The type of document (arangodb, readthedocs, markdown, generic)
        
    Returns:
        Path to the template file
    """
    current_dir = Path(__file__).resolve().parent
    
    # Map doc types to template paths
    templates = {
        "arangodb": current_dir / "arangodb_expected_format.json",
        "readthedocs": current_dir / "html_docs_expected_format.json",
        "markdown": current_dir / "markdown_docs_expected_format.json",
        "generic": current_dir / "expected_format_template.json"
    }
    
    # Special function-specific templates (overrides)
    function_templates = {
        "length": current_dir / "length_expected_format.json",
        "array_intersection": current_dir / "array_intersection_expected_format.json"
    }
    
    # Check for function-specific template first
    for func_name, template_path in function_templates.items():
        if func_name in doc_type.lower() and template_path.exists():
            return template_path
    
    # Fall back to general templates
    template_path = templates.get(doc_type, templates["generic"])
    
    if template_path.exists():
        return template_path
    else:
        # If template doesn't exist, fall back to generic template
        return templates.get("generic") if templates.get("generic").exists() else None


def run_test_for_source(
    source_name: str,
    extraction_file: Path,
    output_dir: Path,
    auto_detect: bool = True,
    expected_format_file: Optional[Path] = None,
    convert: bool = True
) -> Dict[str, Any]:
    """
    Run validation for a single documentation source.
    
    Args:
        source_name: Name of the documentation source
        extraction_file: Path to the extraction file
        output_dir: Directory to save results
        auto_detect: Whether to automatically detect document type
        expected_format_file: Path to the expected format file (optional)
        convert: Whether to convert raw extraction to validation format
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running validation for {source_name}")
    
    # Load the extraction data
    extraction_data = load_json_file(extraction_file)
    if not extraction_data:
        return {
            "source": source_name,
            "valid": False,
            "error": "Failed to load extraction data"
        }
    
    # Create source output directory
    source_dir = output_dir / source_name.lower().replace(" ", "_")
    os.makedirs(source_dir, exist_ok=True)
    
    # Save a copy of the raw extraction
    raw_extraction_path = source_dir / "raw_extraction.json"
    save_json_file(extraction_data, raw_extraction_path)
    
    # Convert to validation format if needed
    if convert:
        try:
            # First, adapt the extraction format to match expected block types
            if isinstance(extraction_data, list):
                adapted_data = adapt_extraction_to_validation_format(extraction_data)
                
                # Save adapted data
                adapted_path = source_dir / "adapted_extraction.json"
                save_json_file(adapted_data, adapted_path)
                logger.info(f"Adapted extraction format saved to {adapted_path}")
                
                # Then convert the adapted data to validation format
                converted_data = convert_to_validation_format(adapted_data)
            else:
                # Handle both raw blocks and QA-compatible format
                blocks = extraction_data.get("sections", extraction_data)
                if isinstance(blocks, list):
                    # First adapt then convert
                    adapted_data = adapt_extraction_to_validation_format(blocks)
                    
                    # Save adapted data
                    adapted_path = source_dir / "adapted_extraction.json"
                    save_json_file(adapted_data, adapted_path)
                    logger.info(f"Adapted extraction format saved to {adapted_path}")
                    
                    # Then convert
                    converted_data = convert_to_validation_format(adapted_data)
                else:
                    logger.error(f"Unrecognized extraction format for {source_name}")
                    return {
                        "source": source_name,
                        "valid": False,
                        "error": "Unrecognized extraction format"
                    }
            
            # Save converted data
            converted_path = source_dir / "converted_extraction.json"
            save_json_file(converted_data, converted_path)
            
            # Use converted data for validation
            validation_data = converted_data
        except Exception as e:
            logger.error(f"Error converting extraction data for {source_name}: {e}")
            return {
                "source": source_name,
                "valid": False,
                "error": f"Conversion error: {str(e)}"
            }
    else:
        # Use raw data for validation
        validation_data = extraction_data
    
    # Detect document type if auto_detect is enabled and no expected_format_file provided
    if not expected_format_file and auto_detect:
        doc_type = detect_doc_type(
            extraction_data if isinstance(extraction_data, list) else 
            extraction_data.get("sections", [])
        )
        logger.info(f"Detected document type for {source_name}: {doc_type}")
        
        # Get the appropriate template
        template_path = get_template_path(doc_type)
        if template_path:
            logger.info(f"Using template: {template_path}")
            expected_format_file = template_path
        else:
            logger.warning(f"No template found for document type: {doc_type}")
    
    # Load the expected format
    if expected_format_file:
        expected_format = load_expected_format(expected_format_file)
        if not expected_format:
            return {
                "source": source_name,
                "valid": False,
                "error": "Failed to load expected format"
            }
    else:
        logger.error(f"No expected format file provided for {source_name}")
        return {
            "source": source_name,
            "valid": False,
            "error": "No expected format file"
        }
    
    # Run validation
    logger.info(f"Validating {source_name} against expected format")
    validation_results = validate_extraction_result(validation_data, expected_format)
    
    # Save validation results
    results_path = source_dir / "validation_results.json"
    save_validation_results(validation_results, results_path)
    
    # Generate HTML report
    try:
        # Try to import the generate_html_report function
        from generate_validation_report import generate_html_report
        html_path = source_dir / "validation_report.html"
        
        generate_html_report(
            validation_results, 
            extraction_file, 
            expected_format_file, 
            html_path, 
            source_name
        )
        logger.info(f"Generated HTML report at {html_path}")
    except ImportError:
        # Create a simple HTML report if the generate_html_report function is not available
        logger.warning("Could not import generate_validation_report, creating simple HTML report")
        html_content = generate_simple_html_report(validation_results, source_name)
        html_path = source_dir / "validation_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    # Return a summary of the validation results
    return {
        "source": source_name,
        "valid": validation_results.get("valid", False),
        "overall_score": validation_results.get("overall_score", 0),
        "structure_valid": validation_results.get("structure_validation", {}).get("valid", False),
        "structure_score": validation_results.get("structure_validation", {}).get("score", 0),
        "content_valid": validation_results.get("content_validation", {}).get("valid", False),
        "content_score": validation_results.get("content_validation", {}).get("score", 0),
        "format_valid": validation_results.get("format_validation", {}).get("valid", False),
        "results_path": str(results_path),
        "html_path": str(html_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


def generate_simple_html_report(validation_results: Dict[str, Any], source_name: str) -> str:
    """
    Generate a simple HTML report for validation results.
    
    Args:
        validation_results: The validation results
        source_name: Name of the documentation source
        
    Returns:
        HTML string
    """
    # Start with HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report - {source_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ border: 1px solid #ddd; border-radius: 4px; padding: 20px; margin-bottom: 20px; }}
        .status {{ display: inline-block; padding: 6px 12px; border-radius: 4px; font-weight: bold; }}
        .passed {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .score {{ font-size: 24px; font-weight: bold; margin-right: 10px; }}
        .detail-section {{ margin-top: 15px; }}
        .error-list {{ color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Report - {source_name}</h1>
        
        <div class="card">
            <h2>Overall Results</h2>
            <div class="status {('passed' if validation_results.get('valid', False) else 'failed')}">
                {('PASSED' if validation_results.get('valid', False) else 'FAILED')}
            </div>
            <p>
                <span class="score">{validation_results.get('overall_score', 0)}%</span>
                Overall validation score
            </p>
        </div>
"""
    
    # Add structure validation section
    if "structure_validation" in validation_results:
        structure = validation_results["structure_validation"]
        html += f"""
        <div class="card">
            <h2>Structure Validation</h2>
            <div class="status {('passed' if structure.get('valid', False) else 'failed')}">
                {('PASSED' if structure.get('valid', False) else 'FAILED')}
            </div>
            <p>
                <span class="score">{structure.get('score', 0)}%</span>
                Structure validation score
            </p>
            <p>Checks passed: {structure.get('passed_checks', 0)}/{structure.get('total_checks', 0)}</p>
            
            {f'''<div class="detail-section">
                <h3>Structure Errors</h3>
                <div class="error-list">
                    <ul>
                        {"".join(f'<li>{error}</li>' for error in structure.get('errors', [])[:10])}
                        {f'<li>... and {len(structure.get("errors", [])) - 10} more errors</li>' if len(structure.get("errors", [])) > 10 else ''}
                    </ul>
                </div>
            </div>''' if structure.get('errors') else ''}
        </div>
"""
    
    # Add content validation section
    if "content_validation" in validation_results:
        content = validation_results["content_validation"]
        html += f"""
        <div class="card">
            <h2>Content Validation</h2>
            <div class="status {('passed' if content.get('valid', False) else 'failed')}">
                {('PASSED' if content.get('valid', False) else 'FAILED')}
            </div>
            <p>
                <span class="score">{content.get('score', 0)}%</span>
                Content validation score
            </p>
            <p>Checks passed: {content.get('passed_checks', 0)}/{content.get('total_checks', 0)}</p>
            
            {f'''<div class="detail-section">
                <h3>Content Errors</h3>
                <div class="error-list">
                    <ul>
                        {"".join(f'<li>{error}</li>' for error in content.get('errors', [])[:10])}
                        {f'<li>... and {len(content.get("errors", [])) - 10} more errors</li>' if len(content.get("errors", [])) > 10 else ''}
                    </ul>
                </div>
            </div>''' if content.get('errors') else ''}
        </div>
"""
    
    # Add structure consistency section
    if "structure_consistency" in validation_results:
        consistency = validation_results["structure_consistency"]
        html += f"""
        <div class="card">
            <h2>Structure Consistency</h2>
            <div class="status {('passed' if consistency.get('valid', False) else 'failed')}">
                {('PASSED' if consistency.get('valid', False) else 'FAILED')}
            </div>
            <p>
                <span class="score">{consistency.get('score', 0)}%</span>
                Structure consistency score
            </p>
            <p>Checks passed: {consistency.get('passed_checks', 0)}/{consistency.get('total_checks', 0)}</p>
            
            {f'''<div class="detail-section">
                <h3>Consistency Errors</h3>
                <div class="error-list">
                    <ul>
                        {"".join(f'<li>{error}</li>' for error in consistency.get('errors', [])[:10])}
                        {f'<li>... and {len(consistency.get("errors", [])) - 10} more errors</li>' if len(consistency.get("errors", [])) > 10 else ''}
                    </ul>
                </div>
            </div>''' if consistency.get('errors') else ''}
        </div>
"""
    
    # Add format validation section
    if "format_validation" in validation_results:
        format_valid = validation_results["format_validation"].get('valid', False)
        html += f"""
        <div class="card">
            <h2>Format Validation</h2>
            <div class="status {('passed' if format_valid else 'failed')}">
                {('PASSED' if format_valid else 'FAILED')}
            </div>
            <p>Basic format compatibility check for QA module</p>
        </div>
"""
    
    # Close HTML
    html += """
        <p>Generated on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
</body>
</html>
"""
    
    return html


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Generate a summary HTML report for all validation tests.
    
    Args:
        all_results: List of validation results for each source
        output_dir: Directory to save the summary report
    """
    logger.info("Generating summary report")
    
    # Sort results by validity and score
    sorted_results = sorted(
        all_results, 
        key=lambda x: (x.get("valid", False), x.get("overall_score", 0)), 
        reverse=True
    )
    
    # Calculate overall statistics
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.get("valid", False))
    average_score = sum(r.get("overall_score", 0) for r in all_results) / total_tests if total_tests > 0 else 0
    
    # Start with HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ border: 1px solid #ddd; border-radius: 4px; padding: 20px; margin-bottom: 20px; }}
        .status {{ display: inline-block; padding: 6px 12px; border-radius: 4px; font-weight: bold; margin-left: 10px; }}
        .passed {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .warning {{ background-color: #fff3cd; color: #856404; }}
        .score {{ font-size: 18px; font-weight: bold; }}
        .summary {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
        .summary-item {{ text-align: center; padding: 15px; border-radius: 4px; background-color: #f8f9fa; flex: 1; margin: 0 10px; }}
        .summary-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .progress-bar {{ height: 20px; background-color: #e9ecef; border-radius: 4px; overflow: hidden; }}
        .progress {{ height: 100%; background-color: #007bff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Summary Report</h1>
        
        <div class="summary">
            <div class="summary-item">
                <div class="summary-value">{passed_tests}/{total_tests}</div>
                <div>Tests Passed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{average_score:.1f}%</div>
                <div>Average Score</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{(passed_tests/total_tests*100):.1f}%</div>
                <div>Success Rate</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Source</th>
                        <th>Status</th>
                        <th>Overall Score</th>
                        <th>Structure</th>
                        <th>Content</th>
                        <th>Format</th>
                        <th>Report</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rows for each test
    for result in sorted_results:
        source = result.get("source", "Unknown")
        valid = result.get("valid", False)
        overall_score = result.get("overall_score", 0)
        structure_score = result.get("structure_score", 0)
        content_score = result.get("content_score", 0)
        format_valid = result.get("format_valid", False)
        html_path = result.get("html_path", "")
        
        # Create relative path to HTML report
        if html_path:
            rel_path = os.path.relpath(html_path, output_dir)
        else:
            rel_path = ""
        
        html += f"""
                    <tr>
                        <td>{source}</td>
                        <td><span class="status {('passed' if valid else 'failed')}">{('PASSED' if valid else 'FAILED')}</span></td>
                        <td>
                            <div class="score">{overall_score:.1f}%</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {overall_score}%;"></div>
                            </div>
                        </td>
                        <td>
                            <div class="score">{structure_score:.1f}%</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {structure_score}%;"></div>
                            </div>
                        </td>
                        <td>
                            <div class="score">{content_score:.1f}%</div>
                            <div class="progress-bar">
                                <div class="progress" style="width: {content_score}%;"></div>
                            </div>
                        </td>
                        <td><span class="status {('passed' if format_valid else 'failed')}">{('PASSED' if format_valid else 'FAILED')}</span></td>
                        <td><a href="{rel_path}" target="_blank">View Report</a></td>
                    </tr>
"""
    
    # Close the HTML
    html += """
                </tbody>
            </table>
        </div>
        
        <p>Generated on """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
</body>
</html>
"""
    
    # Save the HTML report
    report_path = output_dir / "summary.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Save results as JSON
    save_json_file(all_results, output_dir / "all_tests_results.json")
    
    logger.info(f"Summary report generated at {report_path}")


def main():
    """Main function for the validation test suite."""
    parser = argparse.ArgumentParser(description="Run comprehensive validation tests")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                        help="Directory to save validation results")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Automatically detect document type and select template")
    parser.add_argument("--convert", action="store_true",
                        help="Convert raw extraction to validation format")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing extraction files to validate")
    parser.add_argument("--specific-test", type=str, default=None,
                        help="Only run validation for a specific test")
    parser.add_argument("--expected-format", type=str, default=None,
                        help="Path to the expected format JSON file to use (overrides auto-detect)")
    parser.add_argument("--docker-serve", action="store_true",
                        help="Serve results with Docker")
    parser.add_argument("--serve", action="store_true",
                        help="Serve results with Python HTTP server")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to use when serving results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results list
    all_results = []
    
    # Define test cases with extraction files and expected formats
    test_cases = []
    
    # Check for specific test or input directory
    if args.specific_test:
        # Find files for the specific test
        if args.specific_test.lower() == "length":
            # LENGTH function test
            test_cases.append({
                "name": "LENGTH Function",
                "extraction": Path("length_function_extraction.json"),
                "expected_format": Path(args.expected_format) if args.expected_format else Path("length_expected_format.json")
            })
        elif args.specific_test.lower() == "array_intersection":
            # Array intersection test
            test_cases.append({
                "name": "Array Intersection",
                "extraction": Path("array_intersection_summary.json"),
                "expected_format": Path("array_intersection_expected_format.json")
            })
        elif args.specific_test.lower() == "arangodb":
            # ArangoDB test
            test_cases.append({
                "name": "ArangoDB AQL",
                "extraction": Path("arangodb_extraction_summary.json"),
                "expected_format": Path("arangodb_expected_format.json")
            })
        elif args.specific_test.lower() == "readthedocs":
            # ReadTheDocs test
            test_cases.append({
                "name": "Python ReadTheDocs",
                "extraction": Path("deepseek_markdown_extraction_example.json"),
                "expected_format": Path("html_docs_expected_format.json")
            })
        else:
            logger.error(f"Unknown specific test: {args.specific_test}")
            sys.exit(1)
    elif args.input_dir:
        # Find all JSON files in the input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        # Find all JSON files
        json_files = list(input_dir.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        for file_path in json_files:
            # Skip files with "expected" in the name
            if "expected" in file_path.name:
                continue
                
            # Create a test case for each file
            test_cases.append({
                "name": file_path.stem.replace("_", " ").title(),
                "extraction": file_path,
                "expected_format": None  # Will be auto-detected if --auto-detect is used
            })
    else:
        # Default test cases
        test_cases = [
            {
                "name": "LENGTH Function",
                "extraction": Path("length_function_extraction.json"),
                "expected_format": Path("length_expected_format.json")
            },
            {
                "name": "Array Intersection",
                "extraction": Path("array_intersection_summary.json"),
                "expected_format": Path("array_intersection_expected_format.json")
            },
            {
                "name": "ArangoDB AQL",
                "extraction": Path("arangodb_extraction_summary.json"),
                "expected_format": Path("arangodb_expected_format.json")
            },
            {
                "name": "Python ReadTheDocs",
                "extraction": Path("deepseek_markdown_extraction_example.json"),
                "expected_format": Path("html_docs_expected_format.json")
            }
        ]
    
    # Run tests for each case
    for test_case in test_cases:
        name = test_case["name"]
        extraction_file = test_case["extraction"]
        expected_format_file = test_case["expected_format"]
        
        # Check if extraction file exists
        if not extraction_file.exists():
            logger.warning(f"Extraction file not found for {name}: {extraction_file}")
            continue
        
        # Run test
        result = run_test_for_source(
            name,
            extraction_file,
            output_dir,
            auto_detect=args.auto_detect,
            expected_format_file=expected_format_file,
            convert=args.convert
        )
        
        # Add to results
        all_results.append(result)
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    # Print summary
    print("\n----- VALIDATION SUMMARY -----")
    passed = sum(1 for r in all_results if r.get("valid", False))
    total = len(all_results)
    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Average score: {sum(r.get('overall_score', 0) for r in all_results)/total:.1f}%")
    print("\nDetailed results:")
    for result in all_results:
        status = "✅ PASSED" if result.get("valid", False) else "❌ FAILED"
        print(f"  {result.get('source', 'Unknown')}: {status} ({result.get('overall_score', 0):.1f}%)")
    
    # Serve results if requested
    if args.docker_serve:
        try:
            logger.info("Serving results with Docker")
            import subprocess
            
            # Set environment variables for Docker
            os.environ["RESULTS_DIR"] = str(output_dir.absolute())
            os.environ["PORT"] = str(args.port)
            
            # Start Docker container
            subprocess.run(["docker-compose", "up", "-d"])
            print(f"\nResults available at: http://localhost:{args.port}")
            
            # Check if we're in WSL2
            is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
            if is_wsl:
                # Get Windows host IP
                try:
                    windows_host = subprocess.check_output("cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }'", shell=True).decode().strip()
                    print(f"WSL2 detected. Results also available at: http://{windows_host}:{args.port}")
                except:
                    pass
                    
            # Check for Tailscale
            try:
                tailscale_ip = subprocess.check_output("tailscale ip", shell=True).decode().strip()
                print(f"Tailscale detected. Results also available at: http://{tailscale_ip}:{args.port}")
            except:
                pass
        except Exception as e:
            logger.error(f"Error starting Docker container: {e}")
            print("Failed to start Docker container. Try using --serve instead.")
    elif args.serve:
        try:
            logger.info("Serving results with Python HTTP server")
            import http.server
            import socketserver
            import threading
            import socket
            
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            # Set up HTTP server
            handler = http.server.SimpleHTTPRequestHandler
            httpd = socketserver.TCPServer(("", args.port), handler)
            
            # Set directory to serve
            os.chdir(output_dir)
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            print(f"\nServing results at: http://localhost:{args.port}")
            print(f"Also available at: http://{local_ip}:{args.port}")
            
            # Check if we're in WSL2
            is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
            if is_wsl:
                # Get Windows host IP
                try:
                    windows_host = subprocess.check_output("cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }'", shell=True).decode().strip()
                    print(f"WSL2 detected. Results also available at: http://{windows_host}:{args.port}")
                    print("\nNOTE: For WSL2, you may need to set up port forwarding on Windows.")
                    print("Run this in PowerShell as Administrator:")
                    print(f"netsh interface portproxy add v4tov4 listenport={args.port} listenaddress=0.0.0.0 connectport={args.port} connectaddress={local_ip}")
                    print(f"New-NetFirewallRule -DisplayName \"WSL2 Port {args.port}\" -Direction Inbound -LocalPort {args.port} -Action Allow -Protocol TCP")
                except:
                    pass
            
            # Check for Tailscale
            try:
                tailscale_ip = subprocess.check_output("tailscale ip", shell=True).decode().strip()
                print(f"Tailscale detected. Results also available at: http://{tailscale_ip}:{args.port}")
            except:
                pass
            
            print("\nPress Ctrl+C to stop server")
            
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server stopped")
        except Exception as e:
            logger.error(f"Error serving results: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())