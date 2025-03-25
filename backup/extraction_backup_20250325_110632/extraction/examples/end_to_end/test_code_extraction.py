#!/usr/bin/env python3
"""
Test script for extracting code blocks from individual files.

This script tests the extraction capabilities of our module on different
types of files (JavaScript, Python, etc.) and validates the results.
"""

import os
import sys
import json
import uuid
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_code_extraction")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions
from extraction_blocks import extract_all_blocks
try:
    from agent_tools.dualipa.extraction.extractors.utils.language_utils import detect_language
except ImportError:
    # Fallback function if we can't import the real one
    def detect_language(file_path):
        """Simple language detection based on file extension."""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.js'):
            return 'javascript'
        elif file_path.endswith('.ts'):
            return 'typescript'
        elif file_path.endswith('.md'):
            return 'markdown'
        else:
            return 'unknown'
from qa_formatter import create_qa_compatible_blocks, create_qa_compatible_output

def extract_single_file(file_path):
    """Extract blocks from a single file."""
    logger.info(f"Extracting blocks from {file_path}")
    
    # Create a temporary directory containing just this file
    parent_dir = Path(file_path).parent
    
    # Extract blocks from the entire directory
    blocks = extract_all_blocks(parent_dir)
    
    # Filter blocks for just this file
    file_blocks = [b for b in blocks if b.get("file_path", "") == str(file_path)]
    
    logger.info(f"Extracted {len(file_blocks)} blocks from {file_path}")
    return file_blocks

def test_javascript_file(file_path):
    """Test extraction on a JavaScript file."""
    blocks = extract_single_file(file_path)
    
    # Validate basic structure
    if not blocks:
        logger.error(f"No blocks extracted from {file_path}")
        return False
    
    # Check if we have a file block
    file_blocks = [b for b in blocks if b.get("type") == "file"]
    if not file_blocks:
        logger.error(f"No file block found for {file_path}")
        return False
    
    # Check language detection
    language = file_blocks[0].get("language", "")
    if language not in ["javascript", "typescript"]:
        logger.error(f"Wrong language detected: {language} (expected javascript/typescript)")
        return False
    
    # Count different block types
    block_types = {}
    for block in blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
    
    logger.info(f"Block types: {block_types}")
    
    # Check for expected block types in JavaScript
    expected_types = ["file"]
    if len(blocks) > 1:  # If there are blocks beyond the file itself
        expected_types.extend(["function", "class", "method"])
    
    for expected_type in expected_types:
        if expected_type not in block_types and expected_type != "file":
            logger.warning(f"Expected block type '{expected_type}' not found")
    
    return True

def test_python_file(file_path):
    """Test extraction on a Python file."""
    blocks = extract_single_file(file_path)
    
    # Validate basic structure
    if not blocks:
        logger.error(f"No blocks extracted from {file_path}")
        return False
    
    # Check if we have a file block
    file_blocks = [b for b in blocks if b.get("type") == "file"]
    if not file_blocks:
        logger.error(f"No file block found for {file_path}")
        return False
    
    # Check language detection
    language = file_blocks[0].get("language", "")
    if language != "python":
        logger.error(f"Wrong language detected: {language} (expected python)")
        return False
    
    # Count different block types
    block_types = {}
    for block in blocks:
        block_type = block.get("type", "unknown")
        block_types[block_type] = block_types.get(block_type, 0) + 1
    
    logger.info(f"Block types: {block_types}")
    
    # Check for expected block types in Python
    expected_types = ["file"]
    if len(blocks) > 1:  # If there are blocks beyond the file itself
        expected_types.extend(["function", "class", "method"])
    
    for expected_type in expected_types:
        if expected_type not in block_types and expected_type != "file":
            logger.warning(f"Expected block type '{expected_type}' not found")
    
    return True

def run_tests():
    """Run tests on selected files."""
    # JavaScript test files
    js_files = [
        "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/analyzers.js",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/arango-database.js",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/aql/functions.js",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/foxx/router/router.js"
    ]
    
    # Python test files
    py_files = [
        "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/frontend_language/quick_start/openai_example_chat.py",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/embedding.py",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/custom_server.py",
        "/home/grahama/workspace/experiments/agent_tools/test_repos/sglang/examples/runtime/engine/offline_batch_inference_async.py"
    ]
    
    # Test JavaScript files
    for file_path in js_files:
        logger.info(f"\n========== Testing JavaScript file: {Path(file_path).name} ==========")
        success = test_javascript_file(file_path)
        if success:
            logger.info(f"✅ Test passed for {file_path}")
        else:
            logger.error(f"❌ Test failed for {file_path}")
    
    # Test Python files
    for file_path in py_files:
        logger.info(f"\n========== Testing Python file: {Path(file_path).name} ==========")
        success = test_python_file(file_path)
        if success:
            logger.info(f"✅ Test passed for {file_path}")
        else:
            logger.error(f"❌ Test failed for {file_path}")

if __name__ == "__main__":
    run_tests()