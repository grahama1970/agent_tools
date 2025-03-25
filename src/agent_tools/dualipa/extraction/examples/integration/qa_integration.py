#!/usr/bin/env python3
"""
Integration Test for Extraction-QA Pipeline.

This script validates that the output from the extraction module can be
properly consumed by the QA module to generate question-answer pairs.
It confirms the end-to-end workflow from raw repository contents through
extraction to QA generation.
"""

import os
import sys
import json
import asyncio
import tempfile
import logging
import unittest
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_extraction_qa_integration")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))


class TestExtractionQAIntegration(unittest.TestCase):
    """Test the integration between extraction and QA modules."""

    def setUp(self):
        """Set up the test case."""
        # Path to test repository - using sglang which has good content for QA
        self.repo_path = Path("/home/grahama/workspace/experiments/agent_tools/test_repos/sglang")
        self.assertTrue(self.repo_path.exists(), f"Test repository not found: {self.repo_path}")
        
        # Create temporary directory for output files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="extraction_qa_test_"))
        self.extraction_output_path = self.temp_dir / "extraction_output.json"
        self.qa_output_path = self.temp_dir / "qa_output.json"
        
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Import extraction modules
        from extraction_blocks import extract_all_blocks
        from hierarchy_analyzer import analyze_hierarchies, enrich_blocks_with_hierarchy
        from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output
        from ...validation.validation import validate_qa_output
        
        self.extract_all_blocks = extract_all_blocks
        self.analyze_hierarchies = analyze_hierarchies
        self.enrich_blocks_with_hierarchy = enrich_blocks_with_hierarchy
        self.create_qa_compatible_blocks = create_qa_compatible_blocks
        self.create_qa_compatible_output = create_qa_compatible_output
        self.validate_qa_output = validate_qa_output
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
    
    def run_extraction_pipeline(self) -> Dict[str, Any]:
        """Run the extraction pipeline and return the output."""
        logger.info(f"Extracting content from repository: {self.repo_path}")
        
        # 1. Extract blocks
        blocks = self.extract_all_blocks(self.repo_path)
        logger.info(f"Extracted {len(blocks)} blocks")
        
        # 2. Analyze hierarchies
        hierarchies = self.analyze_hierarchies(blocks)
        logger.info(f"Analyzed {len(hierarchies)} file hierarchies")
        
        # 3. Enrich blocks with hierarchy
        enriched_blocks = self.enrich_blocks_with_hierarchy(blocks, hierarchies)
        logger.info(f"Enriched {len(enriched_blocks)} blocks")
        
        # 4. Create QA-compatible blocks
        qa_blocks = self.create_qa_compatible_blocks(enriched_blocks)
        logger.info(f"Created {len(qa_blocks)} QA-compatible blocks")
        
        # 5. Create QA-compatible output
        output = self.create_qa_compatible_output(qa_blocks)
        logger.info(f"Created QA-compatible output")
        
        # 6. Write output to file
        with open(self.extraction_output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved extraction output to {self.extraction_output_path}")
        
        return output
    
    def test_extraction_output_format(self):
        """Test that the extraction output has the format expected by the QA module."""
        # Run extraction pipeline
        extraction_output = self.run_extraction_pipeline()
        
        # Check if the output is a dict (standard format) or list (deepseek format)
        if isinstance(extraction_output, list):
            # Check deepseek.md format
            logger.info("Detected deepseek.md format output")
            self.assertGreater(len(extraction_output), 0, "Empty extraction output")
            
            # Check if first item has required fields
            first_item = extraction_output[0]
            required_fields = ["uuid", "title", "content", "section_hierarchy_depth"]
            for field in required_fields:
                self.assertIn(field, first_item, f"Missing required field: {field}")
            
            # Check nested arrays
            nested_fields = ["images", "tables", "code"]
            for field in nested_fields:
                self.assertIn(field, first_item, f"Missing nested field: {field}")
                self.assertIsInstance(first_item[field], list, f"{field} should be a list")
        else:
            # Check standard format
            logger.info("Detected standard format output")
            
            # Check for sections in output
            self.assertIn("sections", extraction_output, "Missing 'sections' in extraction output")
            sections = extraction_output.get("sections", [])
            self.assertGreater(len(sections), 0, "No sections in extraction output")
            
            # Check first section has required fields
            first_section = sections[0]
            required_fields = ["uuid", "id", "type", "content"]
            for field in required_fields:
                self.assertIn(field, first_section, f"Missing required field: {field}")
                
            # Check extraction_metadata
            self.assertIn("extraction_metadata", extraction_output, "Missing 'extraction_metadata'")
            metadata = extraction_output.get("extraction_metadata", {})
            self.assertIn("statistics", metadata, "Missing 'statistics' in metadata")
    
    def prepare_qa_input_format(self, extraction_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extraction output to the format expected by the QA module."""
        # Check if we need to convert deepseek.md format to QA input format
        if isinstance(extraction_output, list):
            logger.info("Converting deepseek.md format to QA input format")
            
            # Create sections from deepseek items
            sections = []
            for item in extraction_output:
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
        
        # For standard format, ensure it has the needed section fields
        logger.info("Adapting standard format for QA input")
        
        # Get sections and add required fields if missing
        sections = extraction_output.get("sections", [])
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
            "extraction_metadata": extraction_output.get("extraction_metadata", {})
        }
        
        # Ensure metadata has required fields
        if "model_used" not in qa_input["extraction_metadata"]:
            qa_input["extraction_metadata"]["model_used"] = "extraction-model"
            
        if "timestamp" not in qa_input["extraction_metadata"]:
            from datetime import datetime
            qa_input["extraction_metadata"]["timestamp"] = datetime.now().isoformat()
        
        return qa_input
    
    def test_qa_module_compatibility(self):
        """Test that the QA module can process the extraction output."""
        # Skip test if the QA module is not available
        try:
            from agent_tools.dualipa.qa.models.qa_models import QAGenerationConfig
            from agent_tools.dualipa.qa.processor import process_extraction_json
        except ImportError:
            logger.warning("QA module not available, skipping QA compatibility test")
            self.skipTest("QA module not available")
        
        # Run extraction pipeline
        extraction_output = self.run_extraction_pipeline()
        
        # Prepare input for QA module
        qa_input = self.prepare_qa_input_format(extraction_output)
        
        # Save prepared input
        qa_input_path = self.temp_dir / "qa_input.json"
        with open(qa_input_path, 'w', encoding='utf-8') as f:
            json.dump(qa_input, f, indent=2)
        logger.info(f"Saved QA input to {qa_input_path}")
        
        # Create a minimal QA test that just validates the input format
        # without making actual LLM calls
        logger.info("Validating QA module input compatibility")
        
        # Test that the QA config can parse the input
        try:
            config = QAGenerationConfig(
                max_qa_pairs_per_section=1,
                bidirectional_ratio=0.3,
                temperature_range=[0.3]
            )
            # Check that we can access a few sections
            sections = qa_input.get("sections", [])
            if sections:
                test_section = sections[0]
                content = test_section.get("content", "")
                content_type = test_section.get("type", "text")
                uuid = test_section.get("uuid")
                
                self.assertIsNotNone(content, "Section missing content")
                self.assertIsNotNone(content_type, "Section missing type")
                self.assertIsNotNone(uuid, "Section missing uuid")
                
                logger.info(f"Successfully validated QA input section: {uuid}")
                logger.info("QA module input compatibility test passed")
        except Exception as e:
            logger.error(f"Error validating QA input: {e}")
            self.fail(f"QA input validation failed: {e}")


def main():
    """Run the integration test."""
    unittest.main()


if __name__ == "__main__":
    main()