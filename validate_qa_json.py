#\!/usr/bin/env python3
"""
Validate ArangoDB QA JSON Format

This script ensures that the extracted ArangoDB QA format meets the requirements for
question-answering tasks by validating structure, relationships, and content.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def validate_qa_json(json_path: str) -> Dict[str, Any]:
    """
    Validate the QA JSON format.
    
    Args:
        json_path: Path to the QA JSON file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Load the JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        results["errors"].append(f"Failed to load JSON file: {e}")
        return results
    
    # Check for required top-level keys
    required_keys = ["sections", "section_relationships", "extraction_metadata"]
    for key in required_keys:
        if key not in data:
            results["errors"].append(f"Missing required top-level key: {key}")
            
    if results["errors"]:
        return results  # Stop if there are errors in the basic structure
    
    # Validate sections
    sections = data["sections"]
    if not isinstance(sections, list):
        results["errors"].append("'sections' must be a list")
        return results
    
    if len(sections) == 0:
        results["errors"].append("'sections' list is empty")
        return results
    
    # Track section UUIDs for validation
    section_uuids = set()
    section_types = {}
    
    # Validate each section
    for i, section in enumerate(sections):
        # Check required section fields
        if "uuid" not in section:
            results["errors"].append(f"Section {i} is missing 'uuid' field")
        else:
            section_uuids.add(section["uuid"])
            
        if "type" not in section:
            results["errors"].append(f"Section {i} is missing 'type' field")
        else:
            section_type = section["type"]
            section_types[section_type] = section_types.get(section_type, 0) + 1
            
        if "name" not in section:
            results["warnings"].append(f"Section {i} is missing 'name' field")
            
        if "content" not in section:
            results["warnings"].append(f"Section {i} (type: {section.get('type')}) is missing 'content' field")
            
        # Check for required QA fields
        if "extraction_focus" not in section:
            results["warnings"].append(f"Section {i} (name: {section.get('name')}) is missing 'extraction_focus' field")
            
        if "breadcrumb" not in section:
            results["warnings"].append(f"Section {i} (name: {section.get('name')}) is missing 'breadcrumb' field")
    
    # Track statistics
    results["stats"]["total_sections"] = len(sections)
    results["stats"]["section_types"] = section_types
    
    # Validate section relationships
    relationships = data["section_relationships"]
    if not isinstance(relationships, dict):
        results["errors"].append("'section_relationships' must be a dictionary")
    else:
        # Check parent-child relationships
        if "parent_child" not in relationships:
            results["warnings"].append("Missing 'parent_child' in section_relationships")
        else:
            parent_child = relationships["parent_child"]
            for uuid, rel in parent_child.items():
                if uuid not in section_uuids:
                    results["errors"].append(f"Relationship references non-existent section UUID: {uuid}")
                    
                # Check parent reference
                if "parent" in rel and rel["parent"] is not None and rel["parent"] not in section_uuids:
                    results["errors"].append(f"Section {uuid} has non-existent parent: {rel['parent']}")
                    
                # Check children references
                if "children" in rel:
                    for child_uuid in rel["children"]:
                        if child_uuid not in section_uuids:
                            results["errors"].append(f"Section {uuid} references non-existent child: {child_uuid}")
    
    # Validate extraction metadata
    metadata = data["extraction_metadata"]
    if not isinstance(metadata, dict):
        results["errors"].append("'extraction_metadata' must be a dictionary")
    else:
        # Check for statistics
        if "statistics" not in metadata:
            results["warnings"].append("Missing 'statistics' in extraction_metadata")
            
        # Check for model information
        if "model_used" not in metadata:
            results["warnings"].append("Missing 'model_used' in extraction_metadata")
            
        # Check for timestamp
        if "timestamp" not in metadata:
            results["warnings"].append("Missing 'timestamp' in extraction_metadata")
    
    # Set validation result
    results["valid"] = len(results["errors"]) == 0
    
    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Validate ArangoDB QA JSON Format")
    parser.add_argument("json_path", help="Path to the QA JSON file")
    args = parser.parse_args()
    
    # Validate the JSON file
    print(f"Validating QA JSON file: {args.json_path}")
    results = validate_qa_json(args.json_path)
    
    # Print results
    if results["valid"]:
        print("✅ Validation passed\! The QA JSON format is valid.")
    else:
        print("❌ Validation failed\! The QA JSON format has errors.")
        
    # Print errors
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")
            
    # Print warnings
    if results["warnings"]:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"  - {warning}")
            
    # Print statistics
    print("\nStatistics:")
    stats = results["stats"]
    for key, value in stats.items():
        if key == "section_types":
            print(f"  Section Types:")
            for type_name, count in value.items():
                print(f"    - {type_name}: {count}")
        else:
            print(f"  {key}: {value}")
    
    return 0 if results["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
