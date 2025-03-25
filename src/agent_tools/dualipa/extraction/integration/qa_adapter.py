"""
Integration adapter for QA systems.

This module provides interfaces for integrating the extraction output
with question-answering systems.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import uuid

class QAIntegration:
    """
    Interface for integrating extraction output with QA systems.
    
    This class provides methods for converting extraction output to
    formats suitable for QA systems.
    """
    
    def __init__(self):
        """Initialize the QA integration interface."""
        pass
    
    def format_for_qa(self, extraction_output: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format extraction output for QA system consumption.
        
        Args:
            extraction_output: List of extracted blocks
            
        Returns:
            Formatted data suitable for QA system
        """
        # Create a dictionary mapping UUIDs to blocks for easy lookup
        blocks_by_uuid = {block["uuid"]: block for block in extraction_output}
        
        # Get root blocks (those without parent_uuid or with parent_uuid not in blocks)
        root_blocks = [
            block for block in extraction_output 
            if "parent_uuid" not in block or block["parent_uuid"] not in blocks_by_uuid
        ]
        
        # Format for QA
        qa_data = {
            "documents": [],
            "metadata": {
                "total_blocks": len(extraction_output),
                "root_blocks": len(root_blocks)
            }
        }
        
        # Process each root block and its children
        for root_block in root_blocks:
            qa_data["documents"].append(self._process_block_hierarchy(root_block, blocks_by_uuid))
        
        return qa_data
    
    def _process_block_hierarchy(self, block: Dict[str, Any], 
                                blocks_by_uuid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a block and its children for QA format.
        
        Args:
            block: The current block
            blocks_by_uuid: Dictionary mapping UUIDs to blocks
            
        Returns:
            Processed block with children for QA
        """
        qa_block = {
            "id": block["uuid"],
            "text": block["content"],
            "metadata": {
                "name": block["name"],
                "type": block["type"],
                "language": block["language"],
                "file_path": block["file_path"]
            },
            "children": []
        }
        
        # Include additional metadata if available
        if "metadata" in block:
            qa_block["metadata"].update(block["metadata"])
        
        # Process children
        if "child_uuids" in block:
            for child_uuid in block["child_uuids"]:
                if child_uuid in blocks_by_uuid:
                    child_block = blocks_by_uuid[child_uuid]
                    qa_block["children"].append(
                        self._process_block_hierarchy(child_block, blocks_by_uuid)
                    )
        
        return qa_block
    
    def save_qa_format(self, extraction_output: List[Dict[str, Any]], 
                      output_file: str) -> None:
        """
        Format extraction output and save to a file for QA system.
        
        Args:
            extraction_output: List of extracted blocks
            output_file: Path to save the formatted output
        """
        qa_data = self.format_for_qa(extraction_output)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)

class QuestionGenerator:
    """
    Generate sample questions from extraction output.
    
    This class provides methods for generating sample questions from
    extracted content to test QA system integration.
    """
    
    def __init__(self):
        """Initialize the question generator."""
        pass
    
    def generate_questions(self, extraction_output: List[Dict[str, Any]], 
                          num_questions: int = 10) -> List[Dict[str, Any]]:
        """
        Generate sample questions from extraction output.
        
        Args:
            extraction_output: List of extracted blocks
            num_questions: Number of questions to generate
            
        Returns:
            List of questions with expected answers
        """
        questions = []
        
        # Extract content sections for question generation
        content_sections = []
        for block in extraction_output:
            if len(block.get("content", "")) > 100:
                content_sections.append({
                    "content": block["content"],
                    "name": block["name"],
                    "type": block["type"],
                    "uuid": block["uuid"]
                })
        
        # Generate questions (simplified example approach)
        for i in range(min(num_questions, len(content_sections))):
            section = content_sections[i]
            
            # Generate a simple question (in a real implementation, this would use NLP)
            question = f"What is described in the section '{section['name']}'?"
            
            # Truncate content for answer preview
            answer_preview = section["content"][:100] + "..." if len(section["content"]) > 100 else section["content"]
            
            questions.append({
                "id": str(uuid.uuid4()),
                "question": question,
                "answer_source_id": section["uuid"],
                "expected_answer_preview": answer_preview,
                "type": "content_summary"
            })
        
        return questions
