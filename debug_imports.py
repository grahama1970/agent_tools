#!/usr/bin/env python3

"""Debug imports for the code_extractor module."""

import sys
import traceback

print("Python path:")
for path in sys.path:
    print(f"  - {path}")

print("\nAttempting to import code_extractor components...")
try:
    from agent_tools.dualipa.code_extractor import (
        extract_repository, 
        _extract_python_blocks, 
        _extract_with_tree_sitter,
        _extract_js_ts_blocks, 
        _extract_markdown_blocks, 
        _extract_generic_blocks,
        _get_language_for_file_ext
    )
    print("Success! All imports found.")
    
    # Check if tree_sitter is available
    try:
        import tree_sitter
        print("tree_sitter is available.")
    except ImportError as e:
        print(f"tree_sitter is NOT available: {e}")
        
except ImportError as e:
    print(f"Import failed: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    
    # Try individual imports to narrow down the issue
    print("\nTrying individual imports:")
    
    functions = [
        "extract_repository", 
        "_extract_python_blocks", 
        "_extract_with_tree_sitter",
        "_extract_js_ts_blocks", 
        "_extract_markdown_blocks", 
        "_extract_generic_blocks",
        "_get_language_for_file_ext"
    ]
    
    for func in functions:
        try:
            exec(f"from agent_tools.dualipa.code_extractor import {func}")
            print(f"  - {func}: Success")
        except ImportError as e:
            print(f"  - {func}: Failed - {e}")
        except Exception as e:
            print(f"  - {func}: Failed with non-import error - {e}") 