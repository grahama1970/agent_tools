#!/usr/bin/env python3
"""
Validation Framework Test Script.

This script demonstrates how to use the validation framework to validate
documentation extraction results against expected formats.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

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
logger = logging.getLogger("test_validation_framework")


def load_extraction_result(file_path: Path) -> Any:
    """Load an extraction result from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading extraction result: {e}")
        return None


def main():
    """Main function to demonstrate the validation framework."""
    parser = argparse.ArgumentParser(description="Test the extraction validation framework")
    parser.add_argument("--extraction", type=str, required=True,
                        help="Path to the extraction result JSON file")
    parser.add_argument("--expected", type=str, required=True,
                        help="Path to the expected format JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save validation results (optional)")
    args = parser.parse_args()
    
    # Load the extraction result and expected format
    extraction_path = Path(args.extraction)
    expected_path = Path(args.expected)
    
    logger.info(f"Loading extraction result from {extraction_path}")
    extraction_result = load_extraction_result(extraction_path)
    if extraction_result is None:
        logger.error("Failed to load extraction result")
        sys.exit(1)
        
    logger.info(f"Loading expected format from {expected_path}")
    expected_format = load_expected_format(expected_path)
    if not expected_format:
        logger.error("Failed to load expected format")
        sys.exit(1)
    
    # Validate the extraction result
    logger.info("Validating extraction result")
    validation_results = validate_extraction_result(extraction_result, expected_format)
    
    # Print validation results
    print("\n----- VALIDATION RESULTS -----")
    print(f"Overall validity: {'✅ PASSED' if validation_results['valid'] else '❌ FAILED'}")
    print(f"Overall score: {validation_results.get('overall_score', 0)}%")
    
    # Print format validation
    format_valid = validation_results.get('format_validation', {}).get('valid', False)
    print(f"\nFormat validation: {'✅ PASSED' if format_valid else '❌ FAILED'}")
    
    # Print structure validation
    if 'structure_validation' in validation_results:
        structure = validation_results['structure_validation']
        print(f"\nStructure validation: {'✅ PASSED' if structure.get('valid', False) else '❌ FAILED'}")
        print(f"Structure score: {structure.get('score', 0)}%")
        print(f"Checks passed: {structure.get('passed_checks', 0)}/{structure.get('total_checks', 0)}")
        
        if structure.get('errors'):
            print("\nStructure errors:")
            for error in structure.get('errors', [])[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(structure.get('errors', [])) > 5:
                print(f"  ... and {len(structure.get('errors', [])) - 5} more errors")
    
    # Print content validation
    if 'content_validation' in validation_results:
        content = validation_results['content_validation']
        print(f"\nContent validation: {'✅ PASSED' if content.get('valid', False) else '❌ FAILED'}")
        print(f"Content score: {content.get('score', 0)}%")
        print(f"Checks passed: {content.get('passed_checks', 0)}/{content.get('total_checks', 0)}")
        
        if content.get('errors'):
            print("\nContent errors:")
            for error in content.get('errors', [])[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(content.get('errors', [])) > 5:
                print(f"  ... and {len(content.get('errors', [])) - 5} more errors")
    
    # Print structure consistency validation
    if 'structure_consistency' in validation_results:
        consistency = validation_results['structure_consistency']
        print(f"\nStructure consistency: {'✅ PASSED' if consistency.get('valid', False) else '❌ FAILED'}")
        print(f"Consistency score: {consistency.get('score', 0)}%")
        print(f"Checks passed: {consistency.get('passed_checks', 0)}/{consistency.get('total_checks', 0)}")
        
        if consistency.get('errors'):
            print("\nConsistency errors:")
            for error in consistency.get('errors', [])[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(consistency.get('errors', [])) > 5:
                print(f"  ... and {len(consistency.get('errors', [])) - 5} more errors")
    
    # Save validation results if an output path is provided
    if args.output:
        output_path = Path(args.output)
        logger.info(f"Saving validation results to {output_path}")
        save_validation_results(validation_results, output_path)
    
    # Exit with appropriate status code
    sys.exit(0 if validation_results.get('valid', False) else 1)


if __name__ == "__main__":
    main()