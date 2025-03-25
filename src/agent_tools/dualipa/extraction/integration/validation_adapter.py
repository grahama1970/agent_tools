"""
Integration adapter for validation systems.

This module provides interfaces for validating extraction output quality
and compatibility with other systems.
"""

import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple

class ExtractionValidator:
    """
    Interface for validating extraction output.
    
    This class provides methods for validating extraction output against
    expected formats and quality standards.
    """
    
    def __init__(self, schema_file: Optional[str] = None):
        """
        Initialize the extraction validator.
        
        Args:
            schema_file: Optional path to JSON schema file for validation
        """
        self.schema = None
        if schema_file and os.path.exists(schema_file):
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
    
    def validate_extraction(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extraction output against schema and quality standards.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Validation results
        """
        # Initialize validation results
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_blocks": len(extraction_output),
                "blocks_by_type": {},
                "blocks_by_language": {}
            }
        }
        
        # Collect statistics
        for block in extraction_output:
            # Count by type
            block_type = block.get("type", "unknown")
            if block_type not in results["stats"]["blocks_by_type"]:
                results["stats"]["blocks_by_type"][block_type] = 0
            results["stats"]["blocks_by_type"][block_type] += 1
            
            # Count by language
            language = block.get("language", "unknown")
            if language not in results["stats"]["blocks_by_language"]:
                results["stats"]["blocks_by_language"][language] = 0
            results["stats"]["blocks_by_language"][language] += 1
            
            # Validate required fields
            for field in ["uuid", "type", "name", "content", "language", "file_path", "metadata"]:
                if field not in block:
                    results["valid"] = False
                    results["errors"].append(f"Block {block.get('name', 'Unknown')} is missing required field: {field}")
            
            # Validate relationships
            if "parent_uuid" in block and "parent_uuid" is not None:
                parent_exists = any(p["uuid"] == block["parent_uuid"] for p in extraction_output)
                if not parent_exists:
                    results["warnings"].append(f"Block {block.get('name', 'Unknown')} references non-existent parent: {block['parent_uuid']}")
            
            # Validate child_uuids if present
            if "child_uuids" in block:
                for child_uuid in block["child_uuids"]:
                    child_exists = any(c["uuid"] == child_uuid for c in extraction_output)
                    if not child_exists:
                        results["warnings"].append(f"Block {block.get('name', 'Unknown')} references non-existent child: {child_uuid}")
        
        return results
    
    def save_validation_results(self, results: Dict[str, Any], output_file: str) -> None:
        """
        Save validation results to a file.
        
        Args:
            results: Validation results
            output_file: Path to save results
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

class QualityChecker:
    """
    Check extraction quality metrics.
    
    This class provides methods for checking extraction quality metrics
    such as content coverage, hierarchy correctness, and metadata completeness.
    """
    
    def __init__(self):
        """Initialize the quality checker."""
        pass
    
    def check_quality(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check extraction quality metrics.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Quality metrics and issues
        """
        # Initialize quality metrics
        metrics = {
            "content_quality": {
                "empty_blocks": 0,
                "short_blocks": 0,
                "long_blocks": 0
            },
            "hierarchy_quality": {
                "orphaned_blocks": 0,
                "missing_children": 0,
                "circular_references": 0
            },
            "metadata_quality": {
                "missing_metadata": 0,
                "incomplete_metadata": 0
            },
            "issues": []
        }
        
        # Create a map of blocks by UUID for quick lookup
        blocks_by_uuid = {block["uuid"]: block for block in extraction_output}
        
        # Check content quality
        for block in extraction_output:
            content = block.get("content", "")
            
            # Check for empty or short content
            if not content:
                metrics["content_quality"]["empty_blocks"] += 1
                metrics["issues"].append({
                    "type": "empty_content",
                    "block_uuid": block["uuid"],
                    "block_name": block.get("name", "Unknown")
                })
            elif len(content) < 50:
                metrics["content_quality"]["short_blocks"] += 1
            elif len(content) > 10000:
                metrics["content_quality"]["long_blocks"] += 1
            
            # Check hierarchy quality
            if "parent_uuid" in block and block["parent_uuid"] is not None:
                if block["parent_uuid"] not in blocks_by_uuid:
                    metrics["hierarchy_quality"]["orphaned_blocks"] += 1
                    metrics["issues"].append({
                        "type": "orphaned_block",
                        "block_uuid": block["uuid"],
                        "block_name": block.get("name", "Unknown"),
                        "parent_uuid": block["parent_uuid"]
                    })
            
            # Check metadata quality
            if "metadata" not in block or not block["metadata"]:
                metrics["metadata_quality"]["missing_metadata"] += 1
                metrics["issues"].append({
                    "type": "missing_metadata",
                    "block_uuid": block["uuid"],
                    "block_name": block.get("name", "Unknown")
                })
            elif isinstance(block["metadata"], dict):
                # Check for important metadata fields
                important_fields = []
                if block["type"] == "doc_section":
                    important_fields = ["doc_type", "section_hierarchy", "source_url"]
                elif block["type"] == "code_block":
                    important_fields = ["language", "source_file"]
                
                for field in important_fields:
                    if field not in block["metadata"]:
                        metrics["metadata_quality"]["incomplete_metadata"] += 1
                        metrics["issues"].append({
                            "type": "incomplete_metadata",
                            "block_uuid": block["uuid"],
                            "block_name": block.get("name", "Unknown"),
                            "missing_field": field
                        })
                        break
        
        return metrics
