#!/usr/bin/env python3
"""
Direct test runner for multilanguage extraction tests.

This script runs the tests directly without involving pytest's command-line interface.
"""

import os
import sys
import tempfile
from pathlib import Path
import importlib.util
import traceback

def import_module_from_path(module_path, module_name):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """Run the tests directly."""
    # Add the project root to the Python path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    sys.path.append(str(project_root))
    
    # Path to the js_ts_extractor.py file
    js_ts_extractor_path = script_dir / "js_ts_extractor.py"
    if not js_ts_extractor_path.exists():
        print(f"Error: js_ts_extractor.py not found at {js_ts_extractor_path}")
        return 1
    
    # Import the extractor module
    try:
        js_ts_extractor = import_module_from_path(js_ts_extractor_path, "js_ts_extractor")
        process_file = js_ts_extractor.process_file
        TREE_SITTER_LANGUAGES = js_ts_extractor.TREE_SITTER_LANGUAGES
        detect_language = js_ts_extractor.detect_language
    except Exception as e:
        print(f"Error importing js_ts_extractor: {e}")
        traceback.print_exc()
        return 1
    
    print(f"Successfully imported js_ts_extractor from {js_ts_extractor_path}")
    print(f"Tree-sitter languages available: {len(TREE_SITTER_LANGUAGES)}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"Created temporary directory: {temp_path}")
        
        # Sample JavaScript code
        js_code = """
        /**
         * Sample JavaScript function
         */
        function greet(name) {
            return `Hello, ${name}!`;
        }

        class Person {
            constructor(name) {
                this.name = name;
            }
            
            getGreeting() {
                return `Hello, my name is ${this.name}`;
            }
        }
        """
        
        # Create a JavaScript file
        js_file = temp_path / "test.js"
        with open(js_file, "w") as f:
            f.write(js_code)
        
        print(f"Created test JavaScript file: {js_file}")
        
        # Process the file
        output_dir = temp_path / "output"
        output_dir.mkdir()
        
        print(f"Processing JavaScript file...")
        try:
            declarations, output_files = process_file(js_file, {}, output_dir)
            
            # Check results
            print(f"Extracted {len(declarations)} declarations:")
            for i, decl in enumerate(declarations):
                print(f"  {i+1}. {decl.get('type', 'unknown')}: {decl.get('name', 'unnamed')}")
            
            if len(declarations) >= 2:
                print("✅ Test passed: Successfully extracted JavaScript declarations")
            else:
                print("❌ Test failed: Not enough declarations extracted")
                return 1
                
            # Check output files
            print(f"Created {len(output_files)} output files:")
            for name, file_path in output_files.items():
                if os.path.exists(file_path):
                    print(f"  ✅ {name}: {file_path} (exists)")
                else:
                    print(f"  ❌ {name}: {file_path} (missing)")
                    return 1
            
            print("\nAll tests passed! Multilanguage extraction is working properly.")
            return 0
            
        except Exception as e:
            print(f"Error processing file: {e}")
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    sys.exit(main()) 