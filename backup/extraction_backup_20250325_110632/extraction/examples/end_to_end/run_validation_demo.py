#!/usr/bin/env python3
"""
Demonstration Script for the Validation Framework.

This script shows how to use the validation framework to validate
extraction outputs against expected formats.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the necessary modules
from convert_for_validation import convert_to_validation_format, load_json_file, save_json_file
from validation import validate_extraction_result, load_expected_format
from detect_doc_type import detect_doc_type, get_template_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_validation_demo")


def main():
    """Main function to demonstrate the validation framework."""
    parser = argparse.ArgumentParser(description="Demonstrate the extraction validation framework")
    parser.add_argument("--extraction", type=str, required=True,
                        help="Path to the extraction result JSON file")
    parser.add_argument("--output-dir", type=str, default="./validation_results",
                        help="Directory to save validation results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load extraction data
    extraction_path = Path(args.extraction)
    extraction_data = load_json_file(extraction_path)
    
    if not extraction_data:
        logger.error(f"Failed to load extraction data from {extraction_path}")
        sys.exit(1)
    
    # Step 1: Convert the extraction to validation format
    logger.info("Converting extraction data to validation format...")
    converted_data = convert_to_validation_format(extraction_data)
    converted_path = output_dir / f"{extraction_path.stem}_converted.json"
    save_json_file(converted_data, converted_path)
    logger.info(f"Saved converted data to {converted_path}")
    
    # Step 2: Detect document type
    logger.info("Detecting document type...")
    doc_type = detect_doc_type(extraction_data)
    logger.info(f"Detected document type: {doc_type}")
    
    # Step 3: Get appropriate template
    template_path = get_template_path(doc_type)
    logger.info(f"Using template: {template_path}")
    expected_format = load_expected_format(template_path)
    
    # Step 4: Validate the converted data
    logger.info("Validating converted data...")
    validation_result = validate_extraction_result(converted_data, expected_format)
    
    # Step 5: Save validation results
    results_path = output_dir / f"{extraction_path.stem}_validation.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(validation_result, f, indent=2)
    logger.info(f"Saved validation results to {results_path}")
    
    # Step 6: Display summary
    print("\n===== VALIDATION SUMMARY =====")
    print(f"Document type: {doc_type}")
    print(f"Template used: {template_path.name}")
    print(f"Overall validity: {'✅ PASSED' if validation_result['valid'] else '❌ FAILED'}")
    print(f"Overall score: {validation_result.get('overall_score', 0)}%")
    
    if 'structure_validation' in validation_result:
        structure = validation_result['structure_validation']
        print(f"Structure score: {structure.get('score', 0)}%")
    
    if 'content_validation' in validation_result:
        content = validation_result['content_validation']
        print(f"Content score: {content.get('score', 0)}%")
    
    if 'structure_consistency' in validation_result:
        consistency = validation_result['structure_consistency']
        print(f"Consistency score: {consistency.get('score', 0)}%")
    
    # Display some errors if validation failed
    if not validation_result['valid']:
        print("\nSample errors:")
        errors = []
        
        if 'structure_validation' in validation_result and not validation_result['structure_validation'].get('valid', True):
            errors.extend(validation_result['structure_validation'].get('errors', [])[:2])
        
        if 'content_validation' in validation_result and not validation_result['content_validation'].get('valid', True):
            errors.extend(validation_result['content_validation'].get('errors', [])[:2])
        
        if 'structure_consistency' in validation_result and not validation_result['structure_consistency'].get('valid', True):
            errors.extend(validation_result['structure_consistency'].get('errors', [])[:2])
        
        for error in errors[:5]:
            print(f"  - {error}")
        
        if len(errors) > 5:
            print(f"  - ... and {len(errors) - 5} more errors")
    
    print("\n===== END OF SUMMARY =====")
    
    # Exit with appropriate status code
    sys.exit(0 if validation_result['valid'] else 1)


if __name__ == "__main__":
    main()