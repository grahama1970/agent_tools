#!/usr/bin/env python3
"""
Verify code blocks functionality.

This script verifies code blocks extracted from various sources
by testing their validity and structure.
"""

import os
import sys
import tempfile
from pathlib import Path

# Import the required modules
from agent_tools.dualipa.code_extractor import _verify_code_block

def verify_code_block(block, language=None):
    """
    Verify if a code block is valid.
    
    This is a wrapper around the _verify_code_block function in the code_extractor module,
    ensuring consistent verification functionality from the verification subpackage.
    
    Args:
        block (dict): Code block to verify
        language (str, optional): Language to verify against, defaults to block's language
        
    Returns:
        bool: True if the block is valid, False otherwise
    """
    return _verify_code_block(block, language)

def verify_code_blocks(blocks, language=None):
    """
    Verify a list of code blocks.
    
    Args:
        blocks (list): List of code blocks to verify
        language (str, optional): Language to verify against
        
    Returns:
        dict: Dictionary with verification results
    """
    results = {
        "total": len(blocks),
        "valid": 0,
        "invalid": 0,
        "invalid_blocks": []
    }
    
    for block in blocks:
        if verify_code_block(block, language):
            results["valid"] += 1
        else:
            results["invalid"] += 1
            results["invalid_blocks"].append(block)
    
    return results

def main():
    """Run verification of code blocks."""
    # Example usage
    print("Verifying sample code blocks...")
    
    # Create a valid Python block
    valid_block = {
        "language": "python",
        "content": """
def hello_world():
    print("Hello, world!")
    return "Hello, world!"
""",
        "file": "sample.py"
    }
    
    # Create an invalid Python block with syntax error
    invalid_block = {
        "language": "python",
        "content": """
def hello_world()
    print("Hello, world!")
    return "Hello, world!"
""",  # Missing colon
        "file": "sample.py"
    }
    
    # Test verification
    print(f"Valid block verification: {verify_code_block(valid_block)}")
    print(f"Invalid block verification: {verify_code_block(invalid_block)}")
    
    # Test bulk verification
    blocks = [valid_block, invalid_block]
    results = verify_code_blocks(blocks)
    print(f"Bulk verification results: {results}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 