#!/usr/bin/env python3
"""
End-to-end test for AST extraction with memory integration.

This script runs the AST extraction on complex test files and demonstrates
the memory integration capabilities.
"""

import os
import sys
import time
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).parent))

# Import AST tester
from test_ast_extraction import AstTester

# Define test files
COMPLEX_PYTHON_DIR = "test_repos/complex_python_sample"
COMPLEX_TS_DIR = "test_repos/complex_typescript_sample"

# Test files by category
NESTED_CLASS_FILES = [
    f"{COMPLEX_PYTHON_DIR}/nested_classes/nested_example.py"
]

INHERITANCE_FILES = [
    f"{COMPLEX_PYTHON_DIR}/inheritance/complex_inheritance.py" 
]

DECORATOR_FILES = [
    f"{COMPLEX_PYTHON_DIR}/decorator_patterns.py"
]

TYPESCRIPT_FILES = [
    f"{COMPLEX_TS_DIR}/src/interfaces/models.ts",
    f"{COMPLEX_TS_DIR}/src/components/UserProfile.tsx"
]

def run_tests():
    """Run the end-to-end test with memory integration."""
    print("=" * 80)
    print("AST EXTRACTION END-TO-END TEST WITH MEMORY INTEGRATION")
    print("=" * 80)
    
    # Create memory database path
    memory_db_path = "ast_extraction_memory.db"
    if os.path.exists(memory_db_path):
        print(f"Using existing memory database: {memory_db_path}")
    else:
        print(f"Creating new memory database: {memory_db_path}")
    
    # Initialize the tester with memory
    tester = AstTester(memory_db_path=memory_db_path, verbose=True)
    
    # Test nested class examples
    print("\n" + "=" * 80)
    print("TESTING NESTED CLASS EXTRACTION")
    print("=" * 80)
    
    for file_path in NESTED_CLASS_FILES:
        tester.test_file(file_path)
        time.sleep(1)  # Small delay to make output more readable
    
    # Test inheritance examples
    print("\n" + "=" * 80)
    print("TESTING INHERITANCE EXTRACTION")
    print("=" * 80)
    
    for file_path in INHERITANCE_FILES:
        tester.test_file(file_path)
        time.sleep(1)
    
    # Test decorator examples
    print("\n" + "=" * 80)
    print("TESTING DECORATOR PATTERN EXTRACTION")
    print("=" * 80)
    
    for file_path in DECORATOR_FILES:
        tester.test_file(file_path)
        time.sleep(1)
    
    # Test TypeScript files
    print("\n" + "=" * 80)
    print("TESTING TYPESCRIPT EXTRACTION")
    print("=" * 80)
    
    for file_path in TYPESCRIPT_FILES:
        tester.test_file(file_path)
        time.sleep(1)
    
    # Generate visualization if matplotlib is available
    try:
        output_path = "ast_extraction_results.png"
        print(f"\nGenerating visualization to {output_path}")
        tester.visualize_results(output_path)
    except Exception as e:
        print(f"Failed to generate visualization: {e}")
    
    print("\n" + "=" * 80)
    print("END-TO-END TEST COMPLETE")
    print("=" * 80)
    
    # Print summary
    print("\nTEST SUMMARY:")
    success_count = sum(1 for result in tester.results.values() if result["success"])
    total_count = len(tester.results)
    print(f"Tests run: {total_count}")
    print(f"Successful extractions: {success_count}")
    print(f"Failed extractions: {total_count - success_count}")
    print(f"Success rate: {success_count/total_count*100:.1f}%")

if __name__ == "__main__":
    run_tests()