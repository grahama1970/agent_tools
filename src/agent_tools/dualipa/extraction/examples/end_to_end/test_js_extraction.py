#!/usr/bin/env python3
"""
Test script for extracting JavaScript blocks.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_js_extraction")

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

# Import extraction functions directly
from extraction_blocks import extract_all_blocks

# Test a single JavaScript file
def test_single_js_file(file_path):
    """Test extracting blocks from a single JavaScript file."""
    js_file = Path(file_path)
    
    logger.info(f"Testing JavaScript extraction on {js_file.name}")
    
    # Create file block directly
    with open(js_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Print content for debugging
    logger.info(f"File content (first 200 chars): {content[:200]}...")
    
    # Check for functions using regex
    import re
    # More comprehensive function patterns for JavaScript
    func_patterns = [
        # Regular function declarations
        r'function\s+(\w+)\s*\(',
        # Variable assignments with functions
        r'(?:var|let|const)\s+(\w+)\s*=\s*function',
        # Arrow functions
        r'(?:var|let|const)\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
        # Exports
        r'exports\.(\w+)\s*=',
        # ArangoAnalyzer prototype methods
        r'ArangoAnalyzer\.prototype\.(\w+)\s*=',
    ]
    
    all_matches = []
    for pattern in func_patterns:
        compiled = re.compile(pattern, re.MULTILINE)
        matches = list(compiled.finditer(content))
        logger.info(f"Pattern '{pattern}': Found {len(matches)} matches")
        all_matches.extend(matches)
        
        # Print matches for debugging
        for match in matches:
            func_name = match.group(1)
            context = content[max(0, match.start()-10):min(len(content), match.end()+30)]
            logger.info(f"Match: {func_name} - Context: {context}")
    
    # Check for classes
    class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', re.MULTILINE)
    class_matches = list(class_pattern.finditer(content))
    logger.info(f"Found {len(class_matches)} class matches")
    
    # Success
    return True

if __name__ == "__main__":
    # Test a sample JavaScript file
    test_file = "/home/grahama/workspace/experiments/agent_tools/test_repos/arangodb/js/server/modules/@arangodb/analyzers.js"
    test_single_js_file(test_file)