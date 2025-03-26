#!/usr/bin/env python3
"""
Create a final QA-compatible JSON file from a combined extraction.

This script:
1. Loads combined blocks from a specified directory
2. Processes them into the format required by the QA module
3. Adds all required fields for QA compatibility
4. Validates the output against QA module requirements
5. Saves the final JSON to the specified output file

Usage:
python create_final_qa_json.py --input-dir path/to/extraction/dir --output-file output.json
"""

import os
import sys
import json
import uuid
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import validation functions if available
try:
    from agent_tools.dualipa.qa.utils.validation import validate_input_json
    HAS_QA_VALIDATION = True
except ImportError:
    HAS_QA_VALIDATION = False
    
    def validate_input_json(data):
        """Simple fallback validation."""
        return "sections" in data and "extraction_metadata" in data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("create_final_qa_json")

def ensure_required_fields(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all required fields are present in a block.
    
    Args:
        block: The block to process
        
    Returns:
        Block with all required fields
    """
    # Create a copy to avoid modifying the original
    enhanced = block.copy()
    
    # Ensure UUID field
    if "uuid" not in enhanced:
        enhanced["uuid"] = str(uuid.uuid4())
    if "id" not in enhanced and "uuid" in enhanced:
        enhanced["id"] = enhanced["uuid"]
        
    # Ensure content field
    if "content" not in enhanced:
        enhanced["content"] = f"[{enhanced.get('type', 'unknown')}] {enhanced.get('name', 'Unnamed block')}"
    
    # Ensure type field
    if "type" not in enhanced:
        if "file_path" in enhanced and enhanced["file_path"].endswith((".py", ".js", ".ts")):
            enhanced["type"] = "code"
        elif "file_path" in enhanced and enhanced["file_path"].endswith((".md", ".txt")):
            enhanced["type"] = "documentation"
        else:
            enhanced["type"] = "unknown"
    
    # Ensure language field
    if "language" not in enhanced:
        # Try to determine language from metadata or file extension
        if "metadata" in enhanced and "language" in enhanced["metadata"]:
            enhanced["language"] = enhanced["metadata"]["language"]
        elif enhanced.get("type") in ["section", "text", "doc_section"]:
            enhanced["language"] = "markdown"
        elif enhanced.get("type") == "documentation":
            enhanced["language"] = "markdown"
        elif enhanced.get("type") == "file":
            file_path = enhanced.get("file_path", "")
            if file_path.endswith(".md"):
                enhanced["language"] = "markdown"
            elif file_path.endswith((".py", ".pyx", ".pyw")):
                enhanced["language"] = "python"
            elif file_path.endswith((".js", ".jsx")):
                enhanced["language"] = "javascript"
            elif file_path.endswith((".ts", ".tsx")):
                enhanced["language"] = "typescript"
            else:
                enhanced["language"] = "text"
        else:
            enhanced["language"] = "text"
            
    # Ensure extraction_focus field
    if "extraction_focus" not in enhanced:
        if enhanced.get("type") in ["section", "text", "doc_section", "documentation"]:
            enhanced["extraction_focus"] = ["documentation"]
        else:
            enhanced["extraction_focus"] = ["code"]
            
    # Ensure summary_instructions field
    if "summary_instructions" not in enhanced:
        if enhanced.get("type") in ["section", "text", "doc_section", "documentation"]:
            enhanced["summary_instructions"] = "Extract key points from this documentation section"
        else:
            enhanced["summary_instructions"] = "Explain the purpose of this code"
            
    # Ensure breadcrumb field
    if "breadcrumb" not in enhanced:
        if "metadata" in enhanced and "breadcrumb" in enhanced["metadata"]:
            enhanced["breadcrumb"] = enhanced["metadata"]["breadcrumb"]
        elif "metadata" in enhanced and "section_hierarchy" in enhanced["metadata"]:
            enhanced["breadcrumb"] = enhanced["metadata"]["section_hierarchy"]
        else:
            # Create breadcrumb based on file path and name
            file_path = enhanced.get("file_path", "")
            name = enhanced.get("name", "")
            if file_path:
                file_name = file_path.split("/")[-1] if "/" in file_path else file_path
                enhanced["breadcrumb"] = [file_name]
                if name and name != file_name:
                    enhanced["breadcrumb"].append(name)
            else:
                enhanced["breadcrumb"] = [name or "Unnamed Block"]
    
    # Ensure parent-child relationships
    if "parent_uuid" not in enhanced:
        enhanced["parent_uuid"] = None
    if "child_uuids" not in enhanced:
        enhanced["child_uuids"] = []
        
    return enhanced

def create_section_relationships(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create section relationships for QA format.
    
    Args:
        blocks: List of blocks
        
    Returns:
        Dictionary of section relationships
    """
    relationships = {
        "parent_child": {},
        "imports": {},
        "inheritance": {}
    }
    
    # Build parent-child relationships
    for block in blocks:
        block_uuid = block.get("uuid")
        parent_uuid = block.get("parent_uuid")
        child_uuids = block.get("child_uuids", [])
        
        if block_uuid:
            relationships["parent_child"][block_uuid] = {
                "parent": parent_uuid,
                "children": child_uuids
            }
            
    # Add imports if available
    for block in blocks:
        block_uuid = block.get("uuid")
        
        if "imports" in block and block_uuid:
            imports = block.get("imports", [])
            if imports:
                relationships["imports"][block_uuid] = imports
                
        # Check for imports in metadata
        if "metadata" in block and "imports" in block["metadata"] and block_uuid:
            imports = block["metadata"].get("imports", [])
            if imports:
                relationships["imports"][block_uuid] = imports
                
    # Add inheritance if available
    for block in blocks:
        block_uuid = block.get("uuid")
        
        if "inheritance" in block and block_uuid:
            inheritance = block.get("inheritance", [])
            if inheritance:
                relationships["inheritance"][block_uuid] = inheritance
                
        # Check for inheritance in metadata
        if "metadata" in block and "inheritance" in block["metadata"] and block_uuid:
            inheritance = block["metadata"].get("inheritance", [])
            if inheritance:
                relationships["inheritance"][block_uuid] = inheritance
                
    return relationships

def create_qa_compatible_output(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a QA-compatible output structure.
    
    Args:
        blocks: List of blocks to include
        
    Returns:
        QA-compatible output dictionary
    """
    # Ensure all blocks have required fields
    enhanced_blocks = [ensure_required_fields(block) for block in blocks]
    
    # Create section relationships
    section_relationships = create_section_relationships(enhanced_blocks)
    
    # Build statistics
    block_types = {}
    languages = {}
    file_count = 0
    
    for block in enhanced_blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
        
        if block_type == "file":
            file_count += 1
        
        language = block.get("language", "unknown")
        languages[language] = languages.get(language, 0) + 1
    
    # Create metadata
    metadata = {
        "model_used": "dualipa-extraction",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "statistics": {
            "total_blocks": len(enhanced_blocks),
            "total_files": file_count,
            "block_types": block_types,
            "languages": languages
        }
    }
    
    # Create final output
    output = {
        "sections": enhanced_blocks,
        "section_relationships": section_relationships,
        "extraction_metadata": metadata
    }
    
    return output

def load_combined_blocks(directory: Path) -> List[Dict[str, Any]]:
    """
    Load combined blocks from a directory.
    
    Args:
        directory: Directory containing combined_blocks.json
        
    Returns:
        List of blocks
    """
    combined_file = directory / "combined_blocks.json"
    
    if not combined_file.exists():
        logger.error(f"Combined blocks file not found at {combined_file}")
        # Try looking for other JSON files
        json_files = list(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {directory}")
        
        # Use the first JSON file found
        logger.info(f"Using {json_files[0]} instead")
        combined_file = json_files[0]
    
    with open(combined_file, "r", encoding="utf-8") as f:
        combined_blocks = json.load(f)
    
    return combined_blocks

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a final QA-compatible JSON file")
    parser.add_argument("--input-dir", type=str, 
                      default="/home/grahama/workspace/experiments/agent_tools/src/agent_tools/dualipa/extraction/examples/end_to_end/test_results/ultimate_20250325_152132",
                      help="Directory containing combined blocks")
    parser.add_argument("--output-file", type=str, 
                      default="/home/grahama/workspace/experiments/agent_tools/test_output_dualipa.json",
                      help="Path to output JSON file")
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_dir)
        output_file = Path(args.output_file)
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load blocks
        logger.info(f"Loading combined blocks from {input_dir}")
        blocks = load_combined_blocks(input_dir)
        
        # Create QA-compatible output
        logger.info("Creating QA-compatible output")
        output = create_qa_compatible_output(blocks)
        
        # Validate output
        if HAS_QA_VALIDATION:
            logger.info("Validating output against QA module requirements")
            is_valid = validate_input_json(output)
            if not is_valid:
                logger.warning("Output validation failed")
            else:
                logger.info("Output validation passed")
        else:
            logger.info("QA validation module not available, skipping validation")
        
        # Save output
        logger.info(f"Saving output to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Successfully created QA-compatible JSON at {output_file}")
        logger.info(f"Total sections: {len(output['sections'])}")
        logger.info(f"Block types: {output['extraction_metadata']['statistics']['block_types']}")
        
    except Exception as e:
        logger.error(f"Error creating QA-compatible JSON: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()