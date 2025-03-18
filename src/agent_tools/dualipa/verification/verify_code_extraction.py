#!/usr/bin/env python3
"""
Verify code extraction functionality.

This script tests the code extraction functionality in the DuaLipa library
by extracting code blocks from sample files and repositories.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Import the required modules
from agent_tools.dualipa.code_extractor import (
    extract_blocks_from_file,
    extract_blocks_from_text,
    extract_repository,
    create_block_from_text
)

def print_header(text, underline='='):
    """Print a header with underline."""
    print(f"\n{text}")
    print(underline * len(text))

def get_test_code():
    """Return test code with multiple code blocks for testing."""
    return """
# This is a sample Python file with multiple code blocks

def hello_world():
    \"\"\"This is a simple function that prints Hello World\"\"\"
    print("Hello, World!")
    return "Hello, World!"

# Here's another function
def add(a, b):
    \"\"\"Add two numbers and return the result\"\"\"
    return a + b

class Calculator:
    \"\"\"A simple calculator class\"\"\"
    
    def __init__(self):
        self.result = 0
    
    def add(self, a, b=None):
        \"\"\"Add a number to the result or add two numbers\"\"\"
        if b is None:
            self.result += a
            return self.result
        return a + b
        
    def subtract(self, a, b=None):
        \"\"\"Subtract a number from the result or subtract b from a\"\"\"
        if b is None:
            self.result -= a
            return self.result
        return a - b
"""

def verify_block_extraction_from_text():
    """Verify extracting code blocks from text."""
    print_header("Testing code block extraction from text", "-")
    
    test_code = get_test_code()
    
    try:
        print("Extracting code blocks from text...")
        blocks = extract_blocks_from_text(test_code, "sample.py")
        
        # Print block information
        print(f"Extracted {len(blocks)} blocks:")
        for i, block in enumerate(blocks):
            print(f"\nBlock {i+1}:")
            print(f"  Type: {block.get('type', 'unknown')}")
            print(f"  Language: {block.get('language', 'unknown')}")
            print(f"  Length: {len(block.get('content', ''))}")
            print(f"  First line: {block.get('content', '').splitlines()[0] if block.get('content', '') else ''}")
        
        return len(blocks) > 0
    except Exception as e:
        print(f"❌ Error extracting blocks from text: {str(e)}")
        return False

def verify_create_block_from_text():
    """Verify creating a single code block from text."""
    print_header("Testing create block from text", "-")
    
    test_code = """
def hello_world():
    print("Hello, World!")
    return "Hello, World!"
"""
    
    try:
        print("Creating code block from text...")
        block = create_block_from_text(test_code, "python", "function")
        
        # Print block information
        if block:
            print(f"Created block successfully:")
            print(f"  Type: {block.get('type', 'unknown')}")
            print(f"  Language: {block.get('language', 'unknown')}")
            print(f"  Length: {len(block.get('content', ''))}")
            print(f"  Content: {block.get('content', '')[:50]}...")
            return True
        else:
            print("❌ Failed to create block")
            return False
    except Exception as e:
        print(f"❌ Error creating block from text: {str(e)}")
        return False

def verify_repository_extraction():
    """Verify code extraction from a repository."""
    print_header("Testing repository extraction", "-")
    
    # Create a test repository structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        test_files = {
            "main.py": get_test_code(),
            "README.md": "# Test Repository\n\nThis is a test repository for code extraction.",
            "src/utils.py": "def util_function():\n    return 'Utility function'"
        }
        
        # Create the files
        for file_path, content in test_files.items():
            full_path = temp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Create output directory
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Extracting code from test repository at {temp_dir}...")
            stats = extract_repository(
                source=str(temp_path),
                output_path=str(output_dir),
                extract_documentation=True,
                extract_code=True,
                extract_blocks=True
            )
            
            # Check if files were created
            blocks_file = output_dir / "blocks.json"
            code_file = output_dir / "code.json"
            docs_file = output_dir / "documentation.json"
            
            files_exist = blocks_file.exists() and code_file.exists() and docs_file.exists()
            
            # Print statistics
            print(f"\nExtraction statistics:")
            print(f"  Files processed: {stats.get('files_processed', 0)}")
            print(f"  Files with code: {stats.get('files_with_code', 0)}")
            print(f"  Files with docs: {stats.get('files_with_documentation', 0)}")
            print(f"  Code blocks: {stats.get('code_blocks', 0)}")
            
            # Print blocks content
            if blocks_file.exists():
                blocks = json.loads(blocks_file.read_text())
                print(f"\nExtracted {len(blocks)} blocks")
                
                # Print information about a few blocks
                for i, block in enumerate(blocks[:3]):
                    print(f"\nBlock {i+1}:")
                    print(f"  Type: {block.get('type', 'unknown')}")
                    print(f"  Language: {block.get('language', 'unknown')}")
                    print(f"  File: {block.get('file', 'unknown')}")
                    print(f"  Length: {len(block.get('content', ''))}")
            
            return files_exist and stats.get('code_blocks', 0) > 0
        except Exception as e:
            print(f"❌ Error during repository extraction: {str(e)}")
            return False

def main():
    """Run all verification tests."""
    print_header("Code Extraction Verification")
    
    # Run all verification tests
    block_extraction_success = verify_block_extraction_from_text()
    create_block_success = verify_create_block_from_text()
    repo_extraction_success = verify_repository_extraction()
    
    # Calculate overall success
    all_success = (
        block_extraction_success and
        create_block_success and
        repo_extraction_success
    )
    
    # Print summary
    print_header("Verification Summary")
    print(f"Block Extraction from Text: {'✅' if block_extraction_success else '❌'}")
    print(f"Create Block from Text: {'✅' if create_block_success else '❌'}")
    print(f"Repository Extraction: {'✅' if repo_extraction_success else '❌'}")
    print(f"\nOverall: {'✅ All tests passed!' if all_success else '❌ Some tests failed!'}")
    
    # Return exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 