#!/usr/bin/env python3
"""
Extraction Example with Memory Integration

This script demonstrates how to use the AI memory system
in a real extraction workflow with proper error handling
and context management.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction_example.log")
    ]
)
logger = logging.getLogger("extraction_example")

# Import memory helpers
try:
    from src.agent_tools.dualipa.extraction.extraction_memory import (
        init_extraction_memory,
        track_extraction_start,
        track_extraction_progress,
        track_extraction_completion,
        record_extraction_error,
        find_similar_errors,
        save_extraction_knowledge,
        find_extraction_knowledge,
        get_extraction_context
    )
    MEMORY_AVAILABLE = True
except ImportError:
    logger.error("Failed to import extraction memory modules. Continuing without memory.")
    MEMORY_AVAILABLE = False


def simulate_extraction(repo_path: str, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Simulate an extraction process with memory integration.
    
    Args:
        repo_path: Path to repository
        config: Extraction configuration
        
    Returns:
        Tuple of success flag and results dictionary
    """
    # Initialize statistics
    stats = {
        "files_processed": 0,
        "files_extracted": 0,
        "extraction_errors": 0,
        "start_time": time.time()
    }
    
    # Phase 1: Repository scan
    if MEMORY_AVAILABLE:
        repo_name = os.path.basename(repo_path)
        track_extraction_start(repo_name, "complete", config)
    
    logger.info(f"Starting extraction from {repo_path}")
    
    try:
        # Simulate repository scanning
        logger.info("Scanning repository structure...")
        time.sleep(1)  # Simulate work
        
        # Update progress
        if MEMORY_AVAILABLE:
            track_extraction_progress(
                repo_name,
                "scanning",
                "Repository structure scanned",
                "Identify extraction targets",
                {"files_found": 150}
            )
        
        # Phase 2: File identification
        logger.info("Identifying target files...")
        time.sleep(1)  # Simulate work
        
        # Simulate an error during file identification
        if config.get("simulate_errors", False):
            error_msg = "Failed to access directory with insufficient permissions"
            logger.error(f"Error: {error_msg}")
            
            if MEMORY_AVAILABLE:
                record_extraction_error(
                    "permission_error",
                    error_msg,
                    f"{repo_path}/restricted/module.py",
                    "Run with elevated permissions or skip restricted directories",
                    severity=6
                )
                
                # Find similar errors
                similar_errors = find_similar_errors(error_msg)
                if similar_errors:
                    logger.info(f"Found {len(similar_errors)} similar errors with recovery suggestions")
                    for err in similar_errors:
                        logger.info(f"  - {err.get('recovery_action', 'No recovery action')}")
        
        # Update progress
        stats["files_processed"] = 120
        if MEMORY_AVAILABLE:
            track_extraction_progress(
                repo_name,
                "identification",
                "Identified 120 Python files for extraction",
                "Extract code structures from files",
                {"files_processed": 120, "python_files": 80, "js_files": 40}
            )
        
        # Phase 3: Code extraction
        logger.info("Extracting code structures...")
        time.sleep(2)  # Simulate work
        
        # Simulate another error
        if config.get("simulate_errors", False):
            error_msg = "Complex nested class structure could not be parsed"
            logger.error(f"Error: {error_msg}")
            
            if MEMORY_AVAILABLE:
                record_extraction_error(
                    "parsing_error",
                    error_msg,
                    f"{repo_path}/src/complex_module.py",
                    "Use tree-sitter parser instead of regex-based parser",
                    severity=7
                )
                
                # Store knowledge about tree-sitter
                save_extraction_knowledge(
                    "tree-sitter-for-complex-parsing",
                    """# Using Tree-Sitter for Complex Code Structures
                    
When encountering complex nested code structures like deeply nested classes
or complex decorators, the regex-based parser may fail. In these cases, 
switch to tree-sitter which provides robust AST-based parsing.

Example:
```python
from tree_sitter import Parser, Language

# Initialize parser
parser = Parser()
parser.set_language(Language('/path/to/python.so', 'python'))

# Parse code
tree = parser.parse(bytes(code, 'utf8'))
```
                    """,
                    summary="How to use tree-sitter to parse complex code structures",
                    tags=["parsing", "tree-sitter", "python"]
                )
        
        # Update progress
        stats["files_extracted"] = 110
        stats["extraction_errors"] = 10
        if MEMORY_AVAILABLE:
            track_extraction_progress(
                repo_name,
                "extraction",
                "Extracted code from 110 files (10 failures)",
                "Format extraction results for output",
                {"files_processed": 120, "files_extracted": 110, "errors": 10}
            )
        
        # Phase 4: Result formatting
        logger.info("Formatting extraction results...")
        time.sleep(1)  # Simulate work
        
        # Finalize statistics
        stats["end_time"] = time.time()
        stats["duration_seconds"] = stats["end_time"] - stats["start_time"]
        
        # Complete extraction
        if MEMORY_AVAILABLE:
            track_extraction_completion(
                repo_name,
                "Extraction completed with 110 successful files and 10 errors",
                stats
            )
        
        logger.info("Extraction completed successfully")
        return True, stats
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        
        if MEMORY_AVAILABLE:
            record_extraction_error(
                "critical_error",
                f"Unexpected error during extraction: {str(e)}",
                severity=9
            )
        
        stats["end_time"] = time.time()
        stats["duration_seconds"] = stats["end_time"] - stats["start_time"]
        stats["failure_reason"] = str(e)
        
        return False, stats


def main():
    """Main function to run the extraction example."""
    parser = argparse.ArgumentParser(description="Example extraction with memory integration")
    
    parser.add_argument(
        "--repo-path",
        help="Path to repository to extract",
        default="/home/grahama/workspace/experiments/agent_tools/test_repos/python-sample"
    )
    
    parser.add_argument(
        "--memory-db",
        help="Path to memory database",
        default="example_extraction.db"
    )
    
    parser.add_argument(
        "--simulate-errors",
        action="store_true",
        help="Simulate extraction errors"
    )
    
    args = parser.parse_args()
    
    # Initialize extraction memory
    if MEMORY_AVAILABLE:
        init_extraction_memory(args.memory_db)
    
    # Configure extraction
    config = {
        "languages": ["python", "javascript"],
        "max_files": 1000,
        "simulate_errors": args.simulate_errors
    }
    
    # Run extraction process
    success, stats = simulate_extraction(args.repo_path, config)
    
    # Print results
    print("\nExtraction Results:")
    print(f"Success: {success}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files extracted: {stats['files_extracted']}")
    print(f"Errors: {stats['extraction_errors']}")
    print(f"Duration: {stats['duration_seconds']:.2f} seconds")
    
    # Show context from memory
    if MEMORY_AVAILABLE:
        context = get_extraction_context()
        print("\nCurrent Extraction Context:")
        print(f"Task: {context.get('task', 'None')}")
        print(f"Progress: {context.get('progress', 'None')}")
        print(f"Next steps: {context.get('next_steps', 'None')}")


if __name__ == "__main__":
    main()