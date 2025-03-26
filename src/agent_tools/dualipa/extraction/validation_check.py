#!/usr/bin/env python3
"""
Validation checker for the output JSON file against QA module requirements.
"""

import json
import sys
from pathlib import Path

def validate_input_json(input_data):
    """Simple validation function based on QA module requirements."""
    error_messages = []
    
    # Check for required top-level fields
    if not all(field in input_data for field in ["sections", "extraction_metadata"]):
        msg = "Missing required fields in input JSON (require 'sections' and 'extraction_metadata')"
        error_messages.append(msg)
        
    # Check sections structure
    sections = input_data.get("sections", [])
    if not sections or not isinstance(sections, list):
        msg = "Sections must be a non-empty list"
        error_messages.append(msg)
    
    # Required fields for sections
    section_required_fields = ["uuid", "type", "content"]
    
    # Validate each section
    for i, section in enumerate(sections):
        if not all(field in section for field in section_required_fields):
            missing = set(section_required_fields) - set(section.keys())
            msg = f"Section {i} missing required fields: {', '.join(missing)}"
            error_messages.append(msg)
    
    # Validate extraction metadata
    metadata = input_data.get("extraction_metadata", {})
    if not isinstance(metadata, dict):
        msg = "extraction_metadata must be a dictionary"
        error_messages.append(msg)
    elif "model_used" not in metadata:
        msg = "Metadata missing required field: model_used"
        error_messages.append(msg)
    
    return error_messages

def main():
    # Get the input file from command line argument
    if len(sys.argv) < 2:
        print("Usage: python validation_check.py <input_json_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Load and validate the input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            input_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {input_file}: {e}")
            sys.exit(1)
    
    # Validate the input data
    errors = validate_input_json(input_data)
    
    # Print validation results
    if errors:
        print(f"Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("✅ Validation successful!")
        print(f"Found {len(input_data['sections'])} sections in the JSON file")
        print(f"Fields in extraction_metadata: {', '.join(input_data['extraction_metadata'].keys())}")
        print("The JSON file is compatible with the QA module requirements.")
        
if __name__ == "__main__":
    main()