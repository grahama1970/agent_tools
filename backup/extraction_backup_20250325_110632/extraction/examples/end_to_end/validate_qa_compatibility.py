#!/usr/bin/env python3
"""
QA Compatibility Validation Tool for Extraction Output.

This script validates that extraction output is compatible with the QA module's
expected input format without requiring the actual QA module to be imported.
It performs structural validation of the JSON format and reports any issues.
It can also convert extraction output to QA-compatible format.

Key Features:
- Validates extraction output against QA module's expected input format
- Handles both standard format and deepseek format
- Adds missing fields required by the QA module
- Converts formats as needed to ensure compatibility
- Minimal dependencies (no need to import QA module directly)

Usage:
    # Validate an extraction output file
    python validate_qa_compatibility.py <path/to/extraction_output.json>
    
    # Convert to QA-compatible format and save to a new file
    python validate_qa_compatibility.py <path/to/extraction_output.json> --convert --output <output_path>

Examples:
    python validate_qa_compatibility.py ./extraction_output.json
    python validate_qa_compatibility.py ./deepseek_output.json --convert --output ./qa_input.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validate_qa_compatibility")


def load_json_file(file_path: Path) -> Any:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def convert_deepseek_to_qa_format(deepseek_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert deepseek format to the QA input format."""
    # Create sections from deepseek items
    sections = []
    for item in deepseek_data:
        section = {
            "uuid": item.get("uuid"),
            "type": "documentation",
            "content": item.get("content", ""),
            "title": item.get("title", ""),
            "extraction_focus": "technical details",
            "summary_instructions": f"Generate QA pairs about '{item.get('title', 'content')}'",
            "breadcrumb": item.get("section_hierarchy_depth", [])
        }
        sections.append(section)
    
    # Create the expected QA input format
    qa_input = {
        "sections": sections,
        "extraction_metadata": {
            "model_used": "extraction-model",
            "timestamp": "2025-03-21T00:00:00Z",
            "statistics": {
                "total_sections": len(sections)
            }
        }
    }
    return qa_input


def adapt_standard_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt standard format to make it QA-compatible."""
    # Get sections and add required fields if missing
    sections = data.get("sections", [])
    updated_sections = []
    
    for section in sections:
        updated_section = section.copy()
        
        # Add required fields if missing
        if "extraction_focus" not in updated_section:
            updated_section["extraction_focus"] = "technical details"
            
        if "summary_instructions" not in updated_section:
            section_name = updated_section.get("name", "content")
            updated_section["summary_instructions"] = f"Generate QA pairs about '{section_name}'"
            
        if "breadcrumb" not in updated_section:
            updated_section["breadcrumb"] = [updated_section.get("name", "untitled")]
            
        updated_sections.append(updated_section)
    
    # Create updated output with sections
    qa_input = {
        "sections": updated_sections,
        "extraction_metadata": data.get("extraction_metadata", {})
    }
    
    # Ensure metadata has required fields
    if "model_used" not in qa_input["extraction_metadata"]:
        qa_input["extraction_metadata"]["model_used"] = "extraction-model"
        
    if "timestamp" not in qa_input["extraction_metadata"]:
        from datetime import datetime
        qa_input["extraction_metadata"]["timestamp"] = datetime.now().isoformat()
    
    return qa_input


def validate_qa_compatibility(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Validate that the data is compatible with the QA module's expected input format.
    
    Args:
        data: The extraction output to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "sections_count": 0,
        "qa_compatible": False
    }
    
    # Convert to QA input format if needed
    qa_input = None
    
    # Check if we have deepseek format or standard format
    if isinstance(data, list):
        logger.info("Detected deepseek.md format output")
        
        # Check for empty data
        if not data:
            validation_results["errors"].append("Empty deepseek.md format data")
            return validation_results
        
        # Convert to QA input format
        qa_input = convert_deepseek_to_qa_format(data)
        
    elif isinstance(data, dict):
        logger.info("Detected standard format output")
        
        # Check for required fields
        if "sections" not in data:
            validation_results["errors"].append("Missing 'sections' field in data")
            return validation_results
        
        if not data.get("sections"):
            validation_results["errors"].append("Empty 'sections' in data")
            return validation_results
        
        # Adapt standard format to QA input format
        qa_input = adapt_standard_format(data)
    else:
        validation_results["errors"].append(f"Invalid data type: {type(data)}")
        return validation_results
    
    # Now validate the QA input format
    sections = qa_input.get("sections", [])
    validation_results["sections_count"] = len(sections)
    
    # Check for required fields in each section
    required_section_fields = ["uuid", "content", "extraction_focus", "summary_instructions"]
    section_errors = 0
    
    for i, section in enumerate(sections):
        for field in required_section_fields:
            if field not in section:
                validation_results["errors"].append(f"Section {i} missing required field: {field}")
                section_errors += 1
    
    # Check extraction_metadata
    if "extraction_metadata" not in qa_input:
        validation_results["errors"].append("Missing 'extraction_metadata' field")
    else:
        metadata = qa_input.get("extraction_metadata", {})
        if "model_used" not in metadata:
            validation_results["warnings"].append("Missing 'model_used' in metadata")
        if "timestamp" not in metadata:
            validation_results["warnings"].append("Missing 'timestamp' in metadata")
    
    # Determine overall result
    validation_results["valid"] = len(validation_results["errors"]) == 0
    validation_results["qa_compatible"] = validation_results["valid"]
    
    return validation_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate extraction output for QA compatibility")
    parser.add_argument("input_file", type=str, help="Path to extraction output JSON file")
    parser.add_argument("--convert", action="store_true", help="Convert to QA-compatible format and save")
    parser.add_argument("--output", type=str, help="Path to save converted output (only with --convert)")
    args = parser.parse_args()
    
    # Load the input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    data = load_json_file(input_path)
    if data is None:
        sys.exit(1)
    
    # Validate QA compatibility
    results = validate_qa_compatibility(data)
    
    # Report results
    if results["valid"]:
        logger.info("✅ Validation successful")
        logger.info(f"Found {results['sections_count']} sections that are QA-compatible")
    else:
        logger.error("❌ Validation failed")
        for error in results["errors"]:
            logger.error(f"Error: {error}")
    
    for warning in results["warnings"]:
        logger.warning(f"Warning: {warning}")
    
    # Create and save the QA-compatible format if requested
    if args.convert:
        if not args.output:
            output_path = input_path.with_name(f"{input_path.stem}_qa_compatible.json")
        else:
            output_path = Path(args.output)
        
        # Convert to QA input format if needed
        qa_input = None
        
        if isinstance(data, list):
            qa_input = convert_deepseek_to_qa_format(data)
        elif isinstance(data, dict):
            qa_input = adapt_standard_format(data)
        
        if qa_input:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_input, f, indent=2)
            logger.info(f"Saved QA-compatible format to {output_path}")
    
    # Exit with appropriate status code
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()