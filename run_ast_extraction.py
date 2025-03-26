#!/usr/bin/env python3
"""
Run AST extraction with memory integration and convert to QA-compatible format.

This script performs extraction on specified files or directories using tree-sitter
and the memory system, then converts the results to a format compatible with the QA module.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ast_extraction.log")
    ]
)
logger = logging.getLogger("ast_extraction")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import extraction modules
from src.agent_tools.dualipa.extraction.extractors.code.ast_extractor import AstExtractor
from src.agent_tools.dualipa.extraction.extraction_memory import (
    init_extraction_memory,
    track_extraction_start,
    track_extraction_completion
)
sys.path.append(str(Path(__file__).parent))
from qa_convert import ast_to_qa_format

# Import agent memory
try:
    from src.agent_tools.agent_memory.ast_memory import get_ast_memory
    AGENT_MEMORY_AVAILABLE = True
except ImportError:
    AGENT_MEMORY_AVAILABLE = False
    logger.warning("Agent memory system not available. Agent will not maintain state between runs.")

def run_extraction(source_path: str, memory_db_path: str = "extraction_memory.db", 
                  output_file: str = "extraction_output.json",
                  qa_output_file: str = "qa_compatible_output.json") -> bool:
    """
    Run extraction on a source file or directory with memory integration.
    
    Args:
        source_path: Path to the source file or directory
        memory_db_path: Path to the memory database
        output_file: Path to save the extraction output
        qa_output_file: Path to save the QA-compatible output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize memory
        logger.info(f"Initializing memory with database: {memory_db_path}")
        init_extraction_memory(memory_db_path)
        
        # Initialize extractor with memory
        extractor = AstExtractor(memory_db_path=memory_db_path)
        
        source_path = os.path.abspath(source_path)
        repo_name = os.path.basename(source_path)
        
        # Track extraction start
        track_extraction_start(
            repo_name,
            "ast_extraction",
            {
                "source_path": source_path,
                "timestamp": time.time()
            }
        )
        
        result = None
        start_time = time.time()
        
        if os.path.isfile(source_path):
            # Extract single file
            logger.info(f"Extracting from file: {source_path}")
            result = extractor.extract_file(source_path)
            
            # Wrap in a list for consistency
            if result:
                result = {"results": [result]}
        else:
            # Extract directory
            logger.info(f"Extracting from directory: {source_path}")
            result = extractor.extract_directory(source_path)
        
        duration = time.time() - start_time
        logger.info(f"Extraction completed in {duration:.2f} seconds")
        
        # Track extraction completion
        track_extraction_completion(
            repo_name,
            f"Completed extraction of {repo_name}",
            {
                "duration": duration,
                "statistics": extractor.get_statistics()
            }
        )
        
        # Save raw extraction output
        if result:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved extraction output to {output_file}")
            
            # Convert to QA-compatible format
            logger.info("Converting to QA-compatible format")
            qa_compatible = ast_to_qa_format(output_file, qa_output_file)
            logger.info(f"Saved QA-compatible output to {qa_output_file}")
            
            # Update agent memory if available
            if AGENT_MEMORY_AVAILABLE:
                try:
                    agent_memory = get_ast_memory("ast_agent_memory.db")
                    file_name = os.path.basename(source_path)
                    agent_memory.record_file_processed(source_path, result.get("language", "unknown"), result)
                    logger.info(f"Updated agent memory for {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to update agent memory: {e}")
            
            return True
        else:
            logger.error("Extraction returned no results")
            return False
    
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AST extraction with memory integration")
    
    parser.add_argument(
        "source_path",
        help="Path to the source file or directory to extract"
    )
    
    parser.add_argument(
        "--memory-db",
        default="extraction_memory.db",
        help="Path to the memory database"
    )
    
    parser.add_argument(
        "--output",
        default="extraction_output.json",
        help="Path to save the extraction output"
    )
    
    parser.add_argument(
        "--qa-output",
        default="qa_compatible_output.json",
        help="Path to save the QA-compatible output"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_path):
        logger.error(f"Source path does not exist: {args.source_path}")
        return 1
    
    if run_extraction(args.source_path, args.memory_db, args.output, args.qa_output):
        logger.info("Extraction completed successfully")
        return 0
    else:
        logger.error("Extraction failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())