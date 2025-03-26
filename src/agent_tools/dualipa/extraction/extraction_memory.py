#!/usr/bin/env python3
"""
Extraction Memory Helper

Utility functions for managing memory during extraction operations.
This module provides a simplified interface to the AI memory system
specifically designed for extraction workflows.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction_memory.log")
    ]
)
logger = logging.getLogger("extraction_memory")

# Import memory system
try:
    from .ai_memory_system import (
        initialize_memory_system,
        remember,
        recall,
        save_docs,
        find_docs,
        log_error,
        suggest_recovery,
        batch_operation,
        get_system_status
    )
except ImportError:
    logger.error("AI memory system not available. Falling back to mock functions.")
    
    # Mock functions for when memory system is not available
    def initialize_memory_system(db_path=None, load_docs=True):
        logger.warning("Using mock memory system")
        return "Mock memory system initialized"
        
    def remember(task, goal, progress, next_steps, notes=None, priority=5, tags=None):
        logger.warning(f"Mock memory: Remembered task '{task}' (no persistence)")
        return f"Mock memory: Remembered task '{task}'"
        
    def recall(search_term=None, last=False, tag=None, days=None, priority_min=None, semantic=False):
        logger.warning("Mock memory: Recall called (no persistence)")
        return "No contexts found (mock memory)"
        
    def save_docs(topic, content, summary=None, importance=7, source=None, tags=None, related=None):
        logger.warning(f"Mock memory: Saved docs topic '{topic}' (no persistence)")
        return f"Mock memory: Saved docs '{topic}'"
        
    def find_docs(topic=None, search=None, tag=None, importance_min=0, semantic=True, max_results=10):
        logger.warning("Mock memory: Find docs called (no persistence)")
        return "No documentation found (mock memory)"
        
    def log_error(error_type, details, recovery_action=None, severity=5, tags=None):
        logger.warning(f"Mock memory: Logged error '{error_type}' (no persistence)")
        return f"Mock memory: Logged error '{error_type}'"
        
    def suggest_recovery(error_type=None, details=None, use_semantic=True):
        logger.warning("Mock memory: Suggest recovery called (no persistence)")
        return "No recovery suggestions (mock memory)"
        
    def batch_operation(operations):
        logger.warning("Mock memory: Batch operation called (no persistence)")
        return ["Mock results for batch operations"]
        
    def get_system_status():
        logger.warning("Mock memory: Status check called (no persistence)")
        return {"status": "mock", "message": "No real memory system available"}


# Default database path for extraction operations
DEFAULT_DB_PATH = "extraction_memory.db"


def init_extraction_memory(db_path: Optional[str] = None) -> str:
    """
    Initialize memory for extraction operations.
    
    Args:
        db_path: Optional path to the memory database
        
    Returns:
        Initialization result message
    """
    actual_path = db_path or DEFAULT_DB_PATH
    
    # Initialize the memory system
    result = initialize_memory_system(actual_path, True)
    
    # Record initialization
    remember(
        task="Extraction initialization",
        goal="Set up memory for extraction operations",
        progress="Memory system initialized",
        next_steps="Begin extraction process",
        tags=["extraction", "initialization"]
    )
    
    return result


def track_extraction_start(repo_name: str, extraction_type: str = "code", config: Optional[Dict[str, Any]] = None) -> str:
    """
    Record the start of an extraction operation.
    
    Args:
        repo_name: Name of the repository being extracted
        extraction_type: Type of extraction (code, docs, etc.)
        config: Optional extraction configuration
        
    Returns:
        Confirmation message
    """
    config_str = json.dumps(config, indent=2) if config else "Default configuration"
    
    return remember(
        task=f"Extracting {extraction_type} from {repo_name}",
        goal=f"Extract structured {extraction_type} information for QA generation",
        progress="Starting extraction process",
        next_steps="Parse repository structure and identify extraction targets",
        notes=f"Configuration:\n{config_str}",
        priority=7,
        tags=["extraction", extraction_type, repo_name]
    )


def track_extraction_progress(repo_name: str, stage: str, progress_details: str, next_step: str, 
                              stats: Optional[Dict[str, Any]] = None) -> str:
    """
    Update extraction progress in memory.
    
    Args:
        repo_name: Name of the repository being extracted
        stage: Current extraction stage
        progress_details: Details of current progress
        next_step: Next step in the extraction process
        stats: Optional statistics about extraction progress
        
    Returns:
        Confirmation message
    """
    stats_str = json.dumps(stats, indent=2) if stats else "No statistics available"
    
    return remember(
        task=f"Extracting from {repo_name}",
        goal="Complete extraction process successfully",
        progress=f"Stage: {stage} - {progress_details}",
        next_steps=next_step,
        notes=f"Statistics:\n{stats_str}",
        priority=6,
        tags=["extraction", "progress", repo_name]
    )


def track_extraction_completion(repo_name: str, result_summary: str, stats: Dict[str, Any]) -> str:
    """
    Record the completion of an extraction operation.
    
    Args:
        repo_name: Name of the repository being extracted
        result_summary: Summary of extraction results
        stats: Statistics about the extraction process
        
    Returns:
        Confirmation message
    """
    stats_str = json.dumps(stats, indent=2)
    
    return remember(
        task=f"Extraction from {repo_name} completed",
        goal="Successfully extract and process repository content",
        progress=f"Completed: {result_summary}",
        next_steps="Proceed to QA generation or additional processing",
        notes=f"Final Statistics:\n{stats_str}",
        priority=8,
        tags=["extraction", "completion", repo_name]
    )


def record_extraction_error(error_type: str, details: str, file_path: Optional[str] = None, 
                           recovery_action: Optional[str] = None, severity: int = 5) -> str:
    """
    Record an error encountered during extraction.
    
    Args:
        error_type: Type of error encountered
        details: Detailed error information
        file_path: Optional path to file where error occurred
        recovery_action: Optional recovery action suggestion
        severity: Error severity (1-10)
        
    Returns:
        Confirmation message
    """
    # Enhance details with file path if available
    enhanced_details = details
    if file_path:
        enhanced_details = f"File: {file_path}\n\n{details}"
    
    tags = ["extraction", "error"]
    if file_path:
        # Add extension as tag if available
        ext = os.path.splitext(file_path)[1].lstrip('.')
        if ext:
            tags.append(ext)
    
    return log_error(
        error_type=error_type,
        details=enhanced_details,
        recovery_action=recovery_action,
        severity=severity,
        tags=tags
    )


def find_similar_errors(error_details: str) -> List[Dict[str, Any]]:
    """
    Find similar extraction errors in memory.
    
    Args:
        error_details: Details of the current error
        
    Returns:
        List of similar error records with recovery suggestions
    """
    results = suggest_recovery(details=error_details, use_semantic=True)
    
    if isinstance(results, str):
        logger.warning(f"No similar errors found: {results}")
        return []
        
    return results


def save_extraction_knowledge(topic: str, content: str, summary: Optional[str] = None,
                             tags: Optional[List[str]] = None) -> str:
    """
    Save extraction-related knowledge for future reference.
    
    Args:
        topic: Knowledge topic
        content: Detailed knowledge content
        summary: Optional short summary
        tags: Optional tags for categorization
        
    Returns:
        Confirmation message
    """
    # Add extraction tag
    all_tags = ["extraction"]
    if tags:
        all_tags.extend(tags)
    
    return save_docs(
        topic=topic,
        content=content,
        summary=summary,
        importance=7,
        tags=all_tags
    )


def find_extraction_knowledge(search_term: str) -> Union[List[Dict[str, Any]], str]:
    """
    Find extraction-related knowledge using semantic search.
    
    Args:
        search_term: Search term to find in knowledge base
        
    Returns:
        List of relevant knowledge entries or error message
    """
    results = find_docs(
        search=search_term,
        tag="extraction",
        semantic=True,
        max_results=5
    )
    
    return results


def get_extraction_context(repo_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the current extraction context.
    
    Args:
        repo_name: Optional repository name to filter by
        
    Returns:
        Current extraction context information
    """
    if repo_name:
        # Try to find context for this repository
        results = recall(tag=repo_name, last=True)
        if isinstance(results, dict) and 'task' in results:
            return results
    
    # Fall back to most recent extraction context
    results = recall(tag="extraction", last=True)
    
    if isinstance(results, dict) and 'task' in results:
        return results
    
    # Nothing found, return empty context
    return {
        "task": "No active extraction context",
        "goal": "",
        "progress": "",
        "next_steps": "Initialize extraction process"
    }


if __name__ == "__main__":
    # Example usage
    print("Initializing extraction memory...")
    init_extraction_memory("example_extraction.db")
    
    print("\nTracking extraction start...")
    track_extraction_start("example-repo", "code", {"language": "python", "max_files": 100})
    
    print("\nTracking progress...")
    track_extraction_progress(
        "example-repo",
        "parsing",
        "Parsed 50 Python files successfully",
        "Extract class structures from parsed files",
        {"files_processed": 50, "classes_found": 15, "functions_found": 120}
    )
    
    print("\nRecording error...")
    record_extraction_error(
        "parsing_error",
        "Failed to parse nested class structure",
        "example-repo/src/complex_module.py",
        "Use a more robust parser for complex class hierarchies",
        severity=7
    )
    
    print("\nCompleting extraction...")
    track_extraction_completion(
        "example-repo",
        "Successfully extracted 15 classes and 120 functions",
        {"files_processed": 100, "classes_found": 15, "functions_found": 120, "errors": 2}
    )
    
    print("\nFinding relevant knowledge...")
    results = find_extraction_knowledge("nested class extraction")
    if isinstance(results, list):
        print(f"Found {len(results)} relevant documents")
        
    print("\nCurrent extraction context:")
    print(get_extraction_context("example-repo"))