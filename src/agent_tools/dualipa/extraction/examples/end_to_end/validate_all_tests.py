#!/usr/bin/env python3
"""
Validation Tool for All Documentation Tests.

This script finds and validates all documentation extraction tests in the project.
It applies the validation framework to ensure consistent validation across tests.
"""

import os
import sys
import json
import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add the parent directory to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import validation functions
from validation import (
    validate_extraction_result, 
    load_expected_format,
    save_validation_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_all_tests")


def find_test_pairs() -> List[Tuple[Path, Path]]:
    """Find all test pairs (extraction output and expected format) in the project."""
    current_dir = Path(__file__).resolve().parent
    
    # Pattern matching for expected format files
    expected_pattern = "*_expected_format.json"
    expected_files = list(current_dir.glob(expected_pattern))
    
    # Find matching extraction outputs for each expected format
    test_pairs = []
    for expected_file in expected_files:
        # Extract the base name (e.g., "length" from "length_expected_format.json")
        base_name = expected_file.stem.replace("_expected_format", "")
        
        # Look for matching extraction output files
        extraction_pattern = f"{base_name}_*.json"
        extraction_files = list(current_dir.glob(extraction_pattern))
        
        # Filter out the expected format itself and any non-extraction files
        extraction_files = [
            f for f in extraction_files 
            if "_expected_format.json" not in f.name
            and "_summary.json" not in f.name
            and "_validation_results.json" not in f.name
        ]
        
        if extraction_files:
            # Sort by modification time (newest first) and take the first one
            extraction_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            test_pairs.append((extraction_files[0], expected_file))
    
    return test_pairs


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def validate_test_pair(extraction_path: Path, expected_path: Path, output_dir: Optional[Path] = None, auto_detect: bool = False) -> Dict[str, Any]:
    """Validate a test pair and optionally save the results.
    
    Args:
        extraction_path: Path to the extraction result JSON
        expected_path: Path to the expected format JSON
        output_dir: Optional directory to save results
        auto_detect: If True, automatically detect doc type and use appropriate template
    """
    logger.info(f"Validating {extraction_path.name} against {expected_path.name}")
    
    # Load the extraction result
    extraction_result = load_json_file(extraction_path)
    if extraction_result is None:
        return {
            "name": extraction_path.stem,
            "valid": False,
            "error": "Failed to load extraction result"
        }
    
    # Determine the expected format
    if auto_detect:
        # Import the detection module
        try:
            from detect_doc_type import detect_doc_type, get_template_path
            
            # Detect document type
            doc_type = detect_doc_type(extraction_result)
            logger.info(f"Detected document type: {doc_type}")
            
            # Get template path based on detected type
            template_path = get_template_path(doc_type)
            logger.info(f"Using template: {template_path}")
            
            # Load detected template if it exists
            if template_path.exists():
                expected_format = load_json_file(template_path)
                if expected_format:
                    logger.info(f"Using auto-detected template for {doc_type} documentation")
                else:
                    logger.warning(f"Could not load auto-detected template, falling back to provided expected format")
                    expected_format = load_json_file(expected_path)
            else:
                logger.warning(f"Auto-detected template does not exist, falling back to provided expected format")
                expected_format = load_json_file(expected_path)
        except ImportError:
            logger.warning("Could not import detect_doc_type module, using provided expected format")
            expected_format = load_json_file(expected_path)
    else:
        # Use the provided expected format
        expected_format = load_json_file(expected_path)
    
    # Check if we have a valid expected format
    if not expected_format:
        return {
            "name": extraction_path.stem,
            "valid": False,
            "error": "Failed to load expected format"
        }
    
    # Validate the extraction result
    validation_results = validate_extraction_result(extraction_result, expected_format)
    validation_results["name"] = extraction_path.stem
    validation_results["extraction_file"] = str(extraction_path)
    validation_results["expected_file"] = str(expected_path if not auto_detect else template_path)
    validation_results["doc_type"] = doc_type if auto_detect else "manual"
    
    # Save validation results if an output directory is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{extraction_path.stem}_validation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Saved validation results to {output_path}")
    
    return validation_results


def print_validation_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of all validation results."""
    print("\n===== VALIDATION SUMMARY =====")
    
    # Count passed and failed tests
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("valid", False))
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    # Collect results by document type
    doc_types = {}
    for result in results:
        doc_type = result.get("doc_type", "unknown")
        if doc_type not in doc_types:
            doc_types[doc_type] = {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        doc_types[doc_type]["total"] += 1
        if result.get("valid", False):
            doc_types[doc_type]["passed"] += 1
        else:
            doc_types[doc_type]["failed"] += 1
    
    # Print results by document type
    if len(doc_types) > 1:
        print("\nResults by Document Type:")
        for doc_type, stats in doc_types.items():
            print(f"  {doc_type}: {stats['passed']}/{stats['total']} passed")
    
    # Calculate average scores
    structure_scores = [r.get("structure_validation", {}).get("score", 0) for r in results if "structure_validation" in r]
    content_scores = [r.get("content_validation", {}).get("score", 0) for r in results if "content_validation" in r]
    consistency_scores = [r.get("structure_consistency", {}).get("score", 0) for r in results if "structure_consistency" in r]
    overall_scores = [r.get("overall_score", 0) for r in results]
    
    # Print average scores
    print("\nAverage Scores:")
    if structure_scores:
        avg_structure = sum(structure_scores) / len(structure_scores)
        print(f"  Structure: {avg_structure:.1f}%")
    if content_scores:
        avg_content = sum(content_scores) / len(content_scores)
        print(f"  Content: {avg_content:.1f}%")
    if consistency_scores:
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        print(f"  Consistency: {avg_consistency:.1f}%")
    if overall_scores:
        avg_overall = sum(overall_scores) / len(overall_scores)
        print(f"  Overall: {avg_overall:.1f}%")
    
    # Print results for each test
    print("\nDetailed Results:")
    for result in results:
        name = result.get("name", "Unknown")
        valid = result.get("valid", False)
        score = result.get("overall_score", 0)
        doc_type = result.get("doc_type", "")
        
        status = "✅ PASSED" if valid else "❌ FAILED"
        doc_type_info = f" [{doc_type}]" if doc_type else ""
        print(f"{name}{doc_type_info}: {status} ({score}%)")
        
        # Print error if there is one
        if "error" in result:
            print(f"  Error: {result['error']}")
        
        # Print which template was used
        expected_file = result.get("expected_file", "")
        if expected_file:
            template_name = Path(expected_file).name
            print(f"  Template: {template_name}")
        
        # Print individual validation scores
        if "structure_validation" in result:
            structure = result["structure_validation"]
            structure_score = structure.get("score", 0)
            structure_valid = structure.get("valid", False)
            status = "✅" if structure_valid else "❌"
            print(f"  Structure: {status} {structure_score}% " +
                  f"({structure.get('passed_checks', 0)}/{structure.get('total_checks', 0)})")
        
        if "content_validation" in result:
            content = result["content_validation"]
            content_score = content.get("score", 0)
            content_valid = content.get("valid", False)
            status = "✅" if content_valid else "❌"
            print(f"  Content: {status} {content_score}% " +
                  f"({content.get('passed_checks', 0)}/{content.get('total_checks', 0)})")
        
        if "structure_consistency" in result:
            consistency = result["structure_consistency"]
            consistency_score = consistency.get("score", 0)
            consistency_valid = consistency.get("valid", False)
            status = "✅" if consistency_valid else "❌"
            print(f"  Consistency: {status} {consistency_score}% " +
                  f"({consistency.get('passed_checks', 0)}/{consistency.get('total_checks', 0)})")
        
        # Print a few errors if validation failed
        if not valid:
            errors = []
            if "structure_validation" in result and not result["structure_validation"].get("valid", True):
                errors.extend(result["structure_validation"].get("errors", [])[:2])
            if "content_validation" in result and not result["content_validation"].get("valid", True):
                errors.extend(result["content_validation"].get("errors", [])[:2])
            if "structure_consistency" in result and not result["structure_consistency"].get("valid", True):
                errors.extend(result["structure_consistency"].get("errors", [])[:2])
            
            if errors:
                print(f"  Sample errors:")
                for i, error in enumerate(errors[:3]):  # Show up to 3 errors
                    print(f"    - {error}")
                if len(errors) > 3:
                    print(f"    - ... and {len(errors) - 3} more errors")
    
    print("\n===== END OF SUMMARY =====")


def main():
    """Main function to validate all tests."""
    parser = argparse.ArgumentParser(description="Validate all documentation extraction tests")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save validation results (optional)")
    parser.add_argument("--specific-test", type=str, default=None,
                        help="Validate only a specific test (e.g., 'length_function')")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Automatically detect document type and use appropriate template")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Process all extraction files in the specified directory")
    parser.add_argument("--convert", action="store_true",
                        help="Convert extraction format before validation")
    args = parser.parse_args()
    
    # Create output directory if provided
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Process extractions from input directory if provided
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory {input_dir} does not exist or is not a directory")
            sys.exit(1)
            
        # Find all JSON files in the directory
        extraction_files = list(input_dir.glob("*.json"))
        
        # Filter out expected format and validation result files
        extraction_files = [
            f for f in extraction_files 
            if "_expected_format.json" not in f.name
            and "_validation_results.json" not in f.name
            and "_summary.json" not in f.name
        ]
        
        # Filter for specific test if provided
        if args.specific_test:
            extraction_files = [
                f for f in extraction_files
                if args.specific_test in f.stem
            ]
            
        if not extraction_files:
            logger.error(f"No extraction files found in {input_dir}")
            sys.exit(1)
            
        logger.info(f"Found {len(extraction_files)} extraction files in {input_dir}")
        
        # Get default expected format template
        current_dir = Path(__file__).resolve().parent
        default_template = current_dir / "expected_format_template.json"
        
        # Validate each extraction file
        all_results = []
        for extraction_path in extraction_files:
            # If auto-detect is enabled, the template will be selected automatically
            # Otherwise, use the default template
            result = validate_test_pair(
                extraction_path, 
                default_template, 
                output_dir,
                auto_detect=args.auto_detect
            )
            all_results.append(result)
            
    else:
        # Find predefined test pairs
        logger.info("Finding test pairs")
        test_pairs = find_test_pairs()
        
        # Filter for specific test if provided
        if args.specific_test:
            test_pairs = [
                (extraction, expected) for extraction, expected in test_pairs
                if args.specific_test in extraction.stem
            ]
            
            if not test_pairs:
                logger.error(f"No tests found matching '{args.specific_test}'")
                sys.exit(1)
        
        if not test_pairs:
            logger.error("No test pairs found")
            sys.exit(1)
        
        logger.info(f"Found {len(test_pairs)} test pairs")
        for extraction, expected in test_pairs:
            logger.info(f"  {extraction.name} -> {expected.name}")
        
        # Validate each test pair
        all_results = []
        for extraction_path, expected_path in test_pairs:
            # Check if we need to convert the extraction format first
            if args.convert:
                try:
                    from convert_for_validation import load_json_file as convert_load
                    from convert_for_validation import convert_to_validation_format, save_json_file
                    
                    # Load the original extraction
                    original_data = convert_load(extraction_path)
                    if original_data:
                        # Convert to validation format
                        converted_data = convert_to_validation_format(original_data)
                        
                        # Save to a temporary file
                        temp_path = Path(f"{extraction_path}.converted")
                        if save_json_file(converted_data, temp_path):
                            logger.info(f"Converted {extraction_path.name} to validation format")
                            # Use the converted file for validation
                            extraction_path = temp_path
                except ImportError:
                    logger.warning("Could not import conversion module, using original format")
            
            result = validate_test_pair(
                extraction_path, 
                expected_path, 
                output_dir,
                auto_detect=args.auto_detect
            )
            all_results.append(result)
            
            # Clean up temporary converted file if it exists
            if args.convert:
                temp_path = Path(f"{extraction_path}.converted")
                if temp_path.exists():
                    temp_path.unlink()
    
    # Print summary
    print_validation_summary(all_results)
    
    # Save overall results if output directory is provided
    if output_dir:
        overall_results = {
            "results": all_results,
            "timestamp": import_datetime().now().isoformat(),
            "total_tests": len(all_results),
            "passed_tests": sum(1 for r in all_results if r.get("valid", False)),
            "failed_tests": sum(1 for r in all_results if not r.get("valid", False))
        }
        
        overall_path = output_dir / "validation_summary.json"
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2)
        logger.info(f"Saved overall validation results to {overall_path}")
    
    # Determine exit code based on validation results
    all_passed = all(result.get("valid", False) for result in all_results)
    sys.exit(0 if all_passed else 1)
    
    
def import_datetime():
    """Import datetime module on demand."""
    import datetime
    return datetime.datetime


if __name__ == "__main__":
    main()