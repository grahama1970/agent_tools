#!/usr/bin/env python3
"""
Documentation Type Detection Module.

This script analyzes extraction output and determines the appropriate
expected format template to use for validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("detect_doc_type")


def load_json_file(file_path: Path) -> Optional[Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def detect_doc_type(extraction_data: Any) -> str:
    """
    Detect the type of documentation from extraction data.
    
    Args:
        extraction_data: The extraction data (list or dict)
        
    Returns:
        String indicating the detected type: 'arangodb', 'html', 'markdown', or 'generic'
    """
    # Convert to a list if it's a dictionary with sections
    blocks = []
    if isinstance(extraction_data, dict) and "sections" in extraction_data:
        blocks = extraction_data["sections"]
    elif isinstance(extraction_data, list):
        blocks = extraction_data
    else:
        logger.warning("Unknown extraction data format")
        return "generic"
    
    # Check for empty blocks
    if not blocks:
        logger.warning("Empty extraction data")
        return "generic"
    
    # Look for specific indicators
    for block in blocks:
        # Check for ArangoDB-specific metadata
        if block.get("metadata", {}).get("doc_type") == "arangodb" or \
           "arangodb" in block.get("content", "").lower() or \
           "arangodb" in str(block.get("metadata", {})).lower() or \
           "aql" in block.get("content", "").lower():
            return "arangodb"
        
        # Check for language indicators
        language = block.get("language", "").lower()
        if language == "html" or "<html" in block.get("content", "").lower():
            return "html"
        elif language == "markdown" or block.get("type") == "markdown":
            return "markdown"
    
    # Default detection based on content patterns
    markdown_indicators = ["# ", "## ", "```", "---\n", "*italic*", "**bold**"]
    html_indicators = ["<!DOCTYPE html>", "<html", "<body", "<div", "<h1", "<p>"]
    
    markdown_score = 0
    html_score = 0
    
    for block in blocks:
        content = block.get("content", "")
        
        # Count markdown indicators
        for indicator in markdown_indicators:
            if indicator in content:
                markdown_score += 1
        
        # Count HTML indicators
        for indicator in html_indicators:
            if indicator in content:
                html_score += 1
    
    # Determine based on highest score
    if markdown_score > html_score:
        return "markdown"
    elif html_score > markdown_score:
        return "html"
    
    # If still undetermined, use the language metadata
    language_counts = {}
    for block in blocks:
        lang = block.get("language", "").lower()
        if lang:
            language_counts[lang] = language_counts.get(lang, 0) + 1
    
    if language_counts:
        most_common_language = max(language_counts.items(), key=lambda x: x[1])[0]
        if most_common_language == "html":
            return "html"
        elif most_common_language == "markdown":
            return "markdown"
    
    # Default to generic
    return "generic"


def get_template_path(doc_type: str) -> Path:
    """
    Get the path to the appropriate template file.
    
    Args:
        doc_type: The detected document type
        
    Returns:
        Path to the template file
    """
    current_dir = Path(__file__).resolve().parent
    
    # Map doc types to template files
    templates = {
        "arangodb": "arangodb_expected_format.json",
        "html": "html_docs_expected_format.json",
        "markdown": "markdown_docs_expected_format.json",
        "generic": "expected_format_template.json"
    }
    
    template_file = templates.get(doc_type, "expected_format_template.json")
    return current_dir / template_file


def detect_and_load_template(extraction_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Detect document type and load the appropriate template.
    
    Args:
        extraction_path: Path to the extraction data file
        
    Returns:
        Tuple of (doc_type, template_data)
    """
    # Load extraction data
    extraction_data = load_json_file(extraction_path)
    if not extraction_data:
        logger.error(f"Failed to load extraction data from {extraction_path}")
        return "generic", {}
    
    # Detect document type
    doc_type = detect_doc_type(extraction_data)
    logger.info(f"Detected document type: {doc_type}")
    
    # Get template path
    template_path = get_template_path(doc_type)
    logger.info(f"Using template: {template_path}")
    
    # Load template
    template_data = load_json_file(template_path)
    if not template_data:
        logger.error(f"Failed to load template from {template_path}")
        return doc_type, {}
    
    return doc_type, template_data


def main():
    """Main function when script is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect documentation type and recommend template")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the extraction data JSON file")
    args = parser.parse_args()
    
    extraction_path = Path(args.input)
    doc_type, template = detect_and_load_template(extraction_path)
    
    print(f"\nDetected document type: {doc_type}")
    print(f"Recommended template: {get_template_path(doc_type)}")
    
    if template:
        print(f"\nTemplate description: {template.get('description', 'No description')}")
        print(f"Template version: {template.get('version', 'Not specified')}")
        
        structure = template.get('expected_structure', {})
        content = template.get('expected_content_validation', {})
        
        print(f"\nStructure validation threshold: {structure.get('validation_threshold', 'Not specified')}%")
        print(f"Content validation threshold: {content.get('validation_threshold', 'Not specified')}%")
        
        print("\nRequired block types:")
        for block_type in structure.get('required_block_types', []):
            print(f"  - {block_type}")
    else:
        print("\nNo template data available.")


if __name__ == "__main__":
    main()