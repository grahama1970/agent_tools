#!/usr/bin/env python3
"""Test script for indented code block extraction"""

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent_tools.dualipa.markdown_parser import extract_code_blocks

def test_indented_blocks():
    """Test that indented code blocks are properly extracted"""
    # Test markdown with indented code blocks
    markdown = """# Test Markdown

This is a test with indented code blocks.

## Python Example

    # Python code block
    def hello():
        print("Hello, World!")

## JavaScript Example

    // JavaScript code block
    function greet() {
        console.log("Hello, World!");
    }
"""
    
    # Extract the code blocks
    code_blocks = extract_code_blocks(markdown)
    
    # Print the results
    print(f"Found {len(code_blocks)} code blocks:")
    for lang, content in code_blocks.items():
        print(f"\nLanguage: {lang}")
        print(f"Content:\n{content}")
    
    # Verify results
    if len(code_blocks) == 2:
        print("\n✓ Found the expected number of code blocks")
    else:
        print(f"\n✗ Expected 2 code blocks, found {len(code_blocks)}")
    
    # Check Python block
    python_block_found = False
    for lang, content in code_blocks.items():
        if lang == "python" or lang.startswith("python_"):
            python_block_found = True
            if "def hello():" in content and "print(\"Hello, World!\")" in content:
                print("✓ Python block correctly detected")
            else:
                print("✗ Python block content incorrect")
    
    if not python_block_found:
        print("✗ Python block not detected")
    
    # Check JavaScript block
    js_block_found = False
    for lang, content in code_blocks.items():
        if lang == "javascript" or lang.startswith("javascript_"):
            js_block_found = True
            if "function greet()" in content and "console.log(\"Hello, World!\");" in content:
                print("✓ JavaScript block correctly detected")
            else:
                print("✗ JavaScript block content incorrect")
    
    if not js_block_found:
        print("✗ JavaScript block not detected")

if __name__ == "__main__":
    test_indented_blocks() 