#!/usr/bin/env python3

"""
Simple test script for the _get_language_for_file_ext function.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agent_tools.dualipa.code_extractor import _get_language_for_file_ext
    
    # Test the function
    print("Testing _get_language_for_file_ext function...")
    
    test_cases = [
        (".py", "python"),
        (".js", "javascript"),
        (".ts", "typescript"),
        (".tsx", "typescript"),
        (".jsx", "javascript"),
        (".md", "markdown"),
        (".rst", "rst"),
        (".txt", "text"),
        (".c", "c"),
        (".cpp", "cpp"),
        (".java", "java"),
        (".unknown", "text"),
        ("", "text")
    ]
    
    all_passed = True
    
    for ext, expected in test_cases:
        result = _get_language_for_file_ext(ext)
        if result == expected:
            print(f"✅ {ext} -> {result}")
        else:
            print(f"❌ {ext} -> {result} (expected {expected})")
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed! The _get_language_for_file_ext function is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. The _get_language_for_file_ext function is not working as expected.")
        sys.exit(1)
        
except ImportError as e:
    print(f"Error importing _get_language_for_file_ext: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 